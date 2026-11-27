"""基线模型接入层：配置 HiMe 模型请求与日志，不参与规划、触发或记忆管理。"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from openai import APIConnectionError
from websockets.sync.client import connect

from benchmark.libero._vendor.openpi_client import msgpack_numpy

CONNECTION_RETRY_DELAYS_S = (2, 4)
PLANNER_FORMAT_REVISION = "xml-summary-status-v1"


def normalize_planner_format(text: str) -> str:
    """接入层只修复已知标签格式，不生成计划、翻译动作或执行记忆操作。"""
    candidate = text
    # 部分响应漏关 summary，或把后续两字段都放进 summary；只处理唯一字段布局。
    if all(candidate.count(f"<{tag}>") == 1 for tag in ("summary", "memory_operations", "plan_list")):
        start = candidate.index("<summary>")
        memory = candidate.index("<memory_operations>")
        plan_end = candidate.find("</plan_list>")
        close = candidate.find("</summary>")
        if start < memory < plan_end and (close == -1 or close > plan_end):
            if close != -1:
                candidate = candidate[:close] + candidate[close + len("</summary>"):]
            candidate = candidate[:memory] + "</summary>\n" + candidate[memory:]

    markers = {"当前": "current", "待处理": "pending", "已完成": "done"}

    def normalize_plan(match):
        body = re.sub(r"\[(当前|待处理|已完成)\]", lambda m: "[" + markers[m[1]] + "]", match[1])
        return "<plan_list>" + body + "</plan_list>"

    candidate = re.sub(r"<plan_list>(.*?)</plan_list>", normalize_plan, candidate, flags=re.DOTALL)
    if candidate != text:
        # 不尝试猜测其他损坏标签；不把部分修复结果交给原解析器。
        try:
            ET.fromstring("<root>" + candidate + "</root>")
        except ET.ParseError:
            return text
    return candidate


class ModelBudgetExceededError(RuntimeError):
    """模型请求达到预先声明的预算。"""


class ModelOutputLimitError(RuntimeError):
    """模型用完本次响应的输出 token 预算。"""


class HiMePolicyClient:
    """连接常驻 π0.5，并按消息成功发出的时间记录 VLA 调用。"""

    def __init__(self, host: str, port: int, *, task: int, seed: int) -> None:
        self.socket = connect(
            f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=60,
            additional_headers={"X-MAEX-Episode-Seed": str(seed), "X-MAEX-Task": str(task)},
        )
        try:
            self.metadata = msgpack_numpy.unpackb(self.socket.recv(timeout=60))
        except Exception:
            self.socket.close()
            raise
        self.packer = msgpack_numpy.Packer()
        self.sent_at: list[float] = []

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.socket.send(self.packer.pack(observation))
        self.sent_at.append(time.monotonic())
        response = self.socket.recv(timeout=45)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference failed: {response}")
        return msgpack_numpy.unpackb(response)

    def get_server_metadata(self) -> dict:
        return self.metadata

    def close(self) -> None:
        self.socket.close()


class HiMeChatClient:
    """实现 HiMe BaseVLMClient 契约，记录实际请求及 token 用量。"""

    def __init__(
        self, client, *, model: str, role: str, log_path: Path,
        max_tokens: int, max_calls: int, extra_body: dict | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.role = role
        self.log_path = log_path
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.extra_body = extra_body
        self.calls = 0

    @staticmethod
    def encode_image_to_data_url(image_path: str) -> dict:
        """读取 HiMe 当前帧或检索附件；缺失证据应使试验显式失败。"""
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        """提交原有 HiMe 对话，仅在连接失败时有限重试相同请求。"""
        attempt = 1
        while True:
            try:
                return self._chat_once(messages, attempt=attempt)
            except APIConnectionError:
                if attempt > len(CONNECTION_RETRY_DELAYS_S):
                    raise
                delay = CONNECTION_RETRY_DELAYS_S[attempt - 1]
                print(f"[HiMe] {self.role} connection failed; retry {attempt}/2 in {delay}s")
                # 非流式完整响应返回前，调用方尚未执行记忆操作或新的机器人动作。
                time.sleep(delay)
                attempt += 1

    def _chat_once(self, messages: list[dict], *, attempt: int) -> str:
        if self.calls >= self.max_calls:
            raise ModelBudgetExceededError(f"{self.role} reached {self.max_calls} requests")
        self.calls += 1
        # HiMe 为图像块附加本地 meta；不将实现字段发送给模型服务。
        wire_messages, logged_messages = [], []
        for message in messages:
            content = message["content"]
            logged = content
            if isinstance(content, list):
                logged, clean = [], []
                for block in content:
                    clean.append({key: value for key, value in block.items() if key != "meta"})
                    logged.append(
                        {"type": "image_url", "meta": block.get("meta", {})}
                        if block.get("type") == "image_url" else block
                    )
                content = clean
            wire_messages.append({"role": message["role"], "content": content})
            logged_messages.append({"role": message["role"], "content": logged})
        record = {
            "role": self.role, "call": self.calls, "model": self.model,
            "attempt": attempt,
            "max_tokens": self.max_tokens, "extra_body": self.extra_body,
            "messages": logged_messages,
        }
        started = time.monotonic()
        record["started_at"] = started
        try:
            kwargs = {"extra_body": self.extra_body} if self.extra_body is not None else {}
            response = self.client.chat.completions.create(
                model=self.model, messages=wire_messages, max_tokens=self.max_tokens, **kwargs,
            )
            ended = time.monotonic()
            text = response.choices[0].message.content or ""
            record.update(
                response=text, response_model=getattr(response, "model", None),
                finish_reason=response.choices[0].finish_reason, ended_at=ended,
                usage=response.usage.model_dump() if response.usage else None,
            )
            if response.choices[0].finish_reason == "length":
                raise ModelOutputLimitError(f"{self.role} response truncated at the output token limit")
            if self.role == "planner":
                normalized = normalize_planner_format(text)
                if normalized != text:
                    record["normalized_response"] = normalized
                    record["format_revision"] = PLANNER_FORMAT_REVISION
                    print(f"[HiMe] Normalized planner format: {PLANNER_FORMAT_REVISION}")
                text = normalized
            return text
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record.setdefault("ended_at", time.monotonic())
            record["elapsed_s"] = record["ended_at"] - started
            with self.log_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")


class LocalMiniLMEncoder:
    """接入试跑的本地 embedding；使用模型官方的 masked mean pooling。"""

    def __init__(self, model_path: Path) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().cpu()
        self.embedding_dim = int(self.model.config.hidden_size)

    def encode_text(self, text: str) -> list[float]:
        """在 CPU 上生成归一化文本向量，不加载或微调 VLA。"""
        import torch

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256,
        )
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vector = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return vector[0].tolist()

    def encode_obj(self, obj_name: str) -> list[float]:
        """以相同文本编码器表示 HiMe 检索标签。"""
        return self.encode_text(obj_name)


class RecordedEncoder:
    """记录 embedding 调用失败，避免原 HiMe 的容错将其伪装成无匹配记忆。"""

    def __init__(self, encoder, log_path: Path) -> None:
        self.encoder = encoder
        self.log_path = log_path
        self.failed = False

    def encode_obj(self, text: str) -> list[float]:
        """编码检索标签。"""
        return self._encode("encode_obj", text)

    def encode_text(self, text: str) -> list[float]:
        """编码记忆描述。"""
        return self._encode("encode_text", text)

    def _encode(self, method: str, text: str) -> list[float]:
        started = time.monotonic()
        record = {"method": method, "text": text}
        try:
            vector = getattr(self.encoder, method)(text)
            values = np.asarray(vector)
            if values.ndim != 1 or not np.isfinite(values).all() or not np.any(values):
                raise ValueError("Embedding must be a finite, nonzero vector")
            record["dimension"] = len(vector)
            return vector
        except Exception as exc:
            self.failed = True
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - started
            with self.log_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
