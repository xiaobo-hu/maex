"""论文评测的 VoLo 原生代理，提供双模型接入、逐局 VLA 种子和请求时序。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from time import monotonic

from openai import OpenAI
from vlm_orchestrator.protocols.openpi_ws import OpenpiWsBackendConnection
from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
from vlm_orchestrator.strategies.base import StrategyContext
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from vlm_orchestrator.utils import codec
from vlm_orchestrator.vlm import chat_create
from websockets.sync.client import connect


class EpisodeRequests:
    """通过原生代理转发角色请求，记录实际下游调用及回合终止条件。"""

    def __init__(self, directory, plan, observer_port):
        budgets = plan["budgets"]
        self.directory = directory
        self.models = {"planner": plan["planner"]["id"], "observer": plan["observer"]["id"]}
        self.clients = {
            "planner": OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=plan["planner"]["base_url"],
                              timeout=budgets["planner_timeout_s"], max_retries=0),
            "observer": OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{observer_port}/v1",
                               timeout=budgets["observer_timeout_s"], max_retries=0),
        }
        self.limits = {role: budgets[f"{role}_calls"] for role in self.models}
        self.tokens = {role: budgets[f"{role}_tokens"] for role in self.models}
        self.calls = dict.fromkeys(self.models, 0)
        self.stop = None

    def log(self, event, at=None, **fields):
        """以相同的主机 monotonic 时钟记录 Planner、Observer 与 VLA 请求。"""
        entry = {"event": event, "at": monotonic() if at is None else at, **fields}
        with (self.directory / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def call(self, role, *, messages, audit_dir):
        """原生策略决定调用时机，模型输出沿原有解析和回退路径处理。"""
        if self.stop is None and self.calls[role] >= self.limits[role]:
            self.stop = {"reason": "model_budget", "role": role}
        if self.stop is not None:
            raise RuntimeError(f"VoLo episode stopped: {self.stop['reason']}")
        self.calls[role] += 1
        call = self.calls[role]

        def log_request(phase, at):
            self.log(f"{role}_{phase}", at, call=call, model=self.models[role])

        try:
            return chat_create(self.clients[role], model=self.models[role], messages=messages,
                               max_tokens=self.tokens[role], temperature=0, audit_dir=audit_dir,
                               on_request=log_request)
        except Exception as error:
            self.stop = {"reason": "infrastructure_error", "role": role, "error_type": type(error).__name__}
            raise

    def close(self):
        """关闭本回合的两个模型客户端。"""
        for client in self.clients.values():
            client.close()


class PaperSubgoalStrategy(SubgoalStrategy):
    """只替换任务分解和视觉监控的模型调用接口。"""

    def __init__(self, context, requests, check_interval):
        super().__init__(context, SubgoalConfig(check_interval=check_interval),
                         failure_monitor="vlm", recovery_mode="replan", env_mode="libero", motion_planner="linear")
        self.requests = requests

    def request_vlm(self, role, *, messages, max_tokens, audit_dir):
        """按原生 VoLo 传入的角色路由，采用评测配置的统一输出预算。"""
        return self.requests.call(role, messages=messages, audit_dir=audit_dir)


class SeededPolicyConnection(OpenpiWsBackendConnection):
    """在 VoLo 后端 socket 上记录实际 π₀.₅ 发包，并向评测端传递停机原因。"""

    def __init__(self, websocket, requests, timeout):
        super().__init__(websocket)
        self.requests = requests
        self.timeout = timeout

    async def recv_metadata(self):
        """读取常驻策略元数据，确保当前服务采用固定的逐局 RNG。"""
        raw = await asyncio.to_thread(self._ws.recv, timeout=60)
        metadata = codec.unpackb(raw)
        if metadata.get("maex_episode_rng") != "jax_fold_in_seed_task":
            raise ValueError("The resident policy service does not advertise the expected episode RNG")
        return metadata

    async def infer(self, canonical_obs):
        """转发一个动作请求，记录一次真实发送。"""
        if self.requests.stop is not None:
            return {"orchestrator_control_stop": self.requests.stop}
        try:
            await asyncio.to_thread(self._ws.send, self._packer.pack(canonical_obs))
            self.requests.log("vla_sent")
            raw = await asyncio.to_thread(self._ws.recv, timeout=self.timeout)
            if isinstance(raw, str):
                raise RuntimeError(f"OpenPI inference failed: {raw}")
            result = codec.unpackb(raw)
            self.requests.log("vla_end")
            return result
        except TimeoutError:
            self.requests.log("vla_end")
            return {"orchestrator_control_stop": {
                "reason": "infrastructure_error", "role": "vla", "error_type": "TimeoutError",
            }}


class SeededPolicyBackend:
    """每局用相同 task 和 seed 请求与 MAEX 相同的随机流。"""

    def __init__(self, port, task, seed, requests, timeout):
        self.port = port
        self.headers = {"X-MAEX-Task": str(task), "X-MAEX-Episode-Seed": str(seed)}
        self.requests = requests
        self.timeout = timeout

    async def connect(self):
        """启动当前回合的专属 π₀.₅ 连接。"""
        websocket = await asyncio.to_thread(
            connect, f"ws://127.0.0.1:{self.port}", additional_headers=self.headers,
            compression=None, max_size=None, open_timeout=60, ping_interval=None,
        )
        return SeededPolicyConnection(websocket, self.requests, self.timeout)


def main():
    """以原生 VoLo RPC 代理服务单个评测回合。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--observer-port", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.experiment.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    requests = EpisodeRequests(args.run_dir, plan, args.observer_port)
    image_key = "observation/image"
    extra_images = ["observation/wrist_image"]
    context = StrategyContext(vlm=None, image_key=image_key, extra_image_keys=extra_images,
                              vla_host="127.0.0.1", vla_port=args.policy_port)
    strategy = PaperSubgoalStrategy(context, requests, plan["volo"]["check_interval_steps"])
    backend = SeededPolicyBackend(args.policy_port, args.task, args.seed, requests, plan["budgets"]["policy_timeout_s"])
    config = ProxyConfig(host="127.0.0.1", port=args.proxy_port, vla_host="127.0.0.1", vla_port=args.policy_port,
                         strategy=strategy, image_key=image_key, extra_image_keys=extra_images,
                         backend=backend, log_dir=str(args.run_dir / "orchestrator"))
    try:
        OrchestratorProxy(config).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        requests.close()


if __name__ == "__main__":
    main()
