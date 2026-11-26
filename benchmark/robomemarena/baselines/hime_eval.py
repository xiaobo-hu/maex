"""基线评测入口：组合 HiMe、冻结 OpenPI 与 RoboMemArena，独立保存官方评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from openai import OpenAI

from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient
from benchmark.robomemarena.baselines.hime_bridge import HiMeSession
from benchmark.robomemarena.baselines.hime_clients import (
    CONNECTION_RETRY_DELAYS_S,
    PLANNER_FORMAT_REVISION,
    HiMeChatClient,
    LocalMiniLMEncoder,
    ModelBudgetExceededError,
    RecordedEncoder,
)
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.observation import build_roboagent_observation
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.client.video_recording import RoboMemArenaVideoRecorder
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.preprocessing import build_policy_input
from roboagent.prompts.robomemarena import task_primitive_reference

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = Path(__file__).resolve().parent / "hime_prompts"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _load_env(path: Path) -> None:
    # 与项目现有启动脚本一样读取简单 KEY=value；不执行 shell，也不保存密钥快照。
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key.startswith(("LLM_", "EMBEDDING_")):
                os.environ.setdefault(key, value.strip().strip("\"'"))


def execute_episode(
    env, policy, session, recorder, *, max_steps: int, action_chunk_size: int,
    sentry_interval: int, trace_path: Path,
) -> dict:
    """执行 HiMe 闭环；评分仅在本函数返回后读取，不进入规划上下文。"""
    raw = env.reset()
    recorder.publish(build_roboagent_observation(raw))
    instruction = env.task_description
    reference = task_primitive_reference(instruction)
    if reference:
        instruction += "\n\n" + reference
    state = session.start(instruction, raw)
    steps, chunks, done = 0, 0, False
    frames = []
    with trace_path.open("a") as trace:
        def record(kind: str, **extra) -> None:
            trace.write(json.dumps({
                "event": kind, "policy_action_steps": steps, "policy_calls": chunks,
                "plan_list": state.plan_list, "subtask": state.current_subtask_description,
                "hime_done": state.is_done, **extra,
            }, ensure_ascii=False) + "\n")
            trace.flush()

        record("initial_plan")
        while not done and not state.is_done and steps < max_steps:
            subtask = state.current_subtask_description
            actions = np.asarray(policy.infer(build_policy_input(raw, subtask))["actions"])
            if (actions.ndim != 2 or actions.shape[1] != 7
                    or len(actions) == 0 or not np.isfinite(actions).all()):
                raise ValueError("OpenPI must return nonempty finite actions [N, 7]")
            chunks += 1
            chunk_start = steps
            for action in actions[:min(action_chunk_size, max_steps - steps)]:
                result = env.step(action)
                raw, done = result.observation, result.done
                steps += 1
                recorder.publish(build_roboagent_observation(raw))
                if done:
                    break
            record(
                "action_chunk", executed_subtask=subtask, returned_actions=len(actions),
                executed_actions=steps - chunk_start,
            )
            frames.append(session.frames(raw))
            # 终态后不再调用模型；未终止时保持原 HiMe 周期观察与事件触发。
            if not done and steps < max_steps and chunks % sentry_interval == 0:
                state = session.update(frames)
                frames.clear()
                record("sentry_step")
        stop_reason = "environment_done" if done else "hime_done" if state.is_done else "action_budget"
        if state.is_done and not done:
            result = env.settle()
            done = result.done
            recorder.publish(build_roboagent_observation(result.observation))
            record("terminal_settle", environment_done=done)
        record("stop", stop_reason=stop_reason)
    return {
        "stop_reason": stop_reason, "policy_calls": chunks, "policy_action_steps": steps,
        "environment_done": done, "hime_done": state.is_done, "plan_list": state.plan_list,
    }


def main() -> int:
    """运行一次真实 HiMe episode；模型/API 错误保留产物并以非零状态退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hime-root", type=Path, default=PROJECT_ROOT / "HiMe")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--action-chunk-size", type=int, default=20)
    parser.add_argument("--sentry-interval", type=int, default=5)
    parser.add_argument("--sentry-window", type=int, default=8)
    parser.add_argument("--fallback-images", type=int, default=50)
    parser.add_argument("--max-planner-calls", type=int, default=80)
    parser.add_argument("--max-sentry-calls", type=int, default=125)
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8021)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8020)
    parser.add_argument("--sentry-base-url", default="http://127.0.0.1:8022/v1")
    parser.add_argument("--sentry-model", default="robo-observer")
    parser.add_argument("--local-embedding-model", type=Path)
    args = parser.parse_args()
    for name in ("action_chunk_size", "sentry_interval", "sentry_window", "fallback_images",
                 "max_planner_calls", "max_sentry_calls"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    _load_env(args.env_file)
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME"):
        if not os.environ.get(key):
            parser.error(f"Missing {key}")
    if args.local_embedding_model is None:
        for key in ("EMBEDDING_OPENAI_API_KEY", "EMBEDDING_OPENAI_BASE_URL", "EMBEDDING_MODEL"):
            if not os.environ.get(key):
                parser.error(f"Missing {key}; alternatively specify --local-embedding-model")
    root, run_dir = args.hime_root.resolve(), args.run_dir.resolve()
    if not (root / "server/task_manager.py").is_file():
        parser.error("--hime-root must point to the original HiMe checkout")
    config = RoboMemArenaConfig(task_id=args.task, seed=args.seed, max_steps=args.max_steps)
    local_embedding = args.local_embedding_model.resolve() if args.local_embedding_model else None
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PROMPT_DIR, run_dir / "prompts")
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py") if ".venv" not in path.parts
    }
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
    if max_tokens < 1:
        parser.error("LLM_MAX_TOKENS must be positive")
    thinking_disabled = os.environ.get("LLM_DISABLE_THINKING", "true").lower() in {"true", "1", "yes"}
    extra_body = {"thinking": {"type": "disabled" if thinking_disabled else "enabled"}}
    _write_json(run_dir / "run_config.json", {
        **vars(args), "baseline": "HiMe-Same-Backbone", "hime_revision": revision,
        "planner_model": os.environ["LLM_MODEL_NAME"], "planner_max_tokens": max_tokens,
        "planner_extra_body": extra_body, "sentry_max_tokens": 512,
        "connection_retry_delays_s": CONNECTION_RETRY_DELAYS_S,
        "planner_format_revision": PLANNER_FORMAT_REVISION,
        "embedding_model": str(local_embedding) if local_embedding else os.environ["EMBEDDING_MODEL"],
        "embedding_variant": "local-MiniLM-adaptation" if local_embedding else "OpenAI-compatible",
        "memory_initialization": "fresh-per-episode", "primitive_reference": "same-as-RoboAgent",
        "policy": "pi05_robomemarena", "source_sha256": hashes,
        "budget_protocol": "physical steps and separate model request caps; not token-matched",
    })
    # 原 HiMe 的相对日志路径落到本次 run 中，不写入上游源码目录。
    os.chdir(run_dir)
    sys.path.insert(0, str(root))
    resources = ExitStack()
    env = policy = session = None
    recorder = RoboMemArenaVideoRecorder(run_dir / "videos", task_id=args.task)
    outcome, evaluation = {}, None
    started = time.monotonic()
    exit_code = 0
    try:
        if local_embedding:
            encoder = LocalMiniLMEncoder(local_embedding)
        else:
            from memory.encoder import OpenAIEmbeddingEncoder

            encoder = OpenAIEmbeddingEncoder(
                model=os.environ["EMBEDDING_MODEL"], embedding_dim=0,
                api_key=os.environ["EMBEDDING_OPENAI_API_KEY"],
                base_url=os.environ["EMBEDDING_OPENAI_BASE_URL"],
            )
            resources.callback(encoder._client.close)
        recorded_encoder = RecordedEncoder(encoder, run_dir / "embedding_calls.jsonl")
        recorded_encoder.encode_text("RoboMemArena memory connectivity check")
        planner_api = OpenAI(
            api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"],
            timeout=180, max_retries=0,
        )
        resources.callback(planner_api.close)
        sentry_api = OpenAI(
            api_key=os.environ.get("OBSERVER_OPENAI_API_KEY", "EMPTY"),
            base_url=args.sentry_base_url, timeout=180, max_retries=0,
        )
        resources.callback(sentry_api.close)
        session = HiMeSession(
            root, run_dir / "prompts",
            HiMeChatClient(
                planner_api, model=os.environ["LLM_MODEL_NAME"], role="planner",
                log_path=run_dir / "model_calls.jsonl", max_tokens=max_tokens,
                max_calls=args.max_planner_calls, extra_body=extra_body,
            ),
            HiMeChatClient(
                sentry_api, model=args.sentry_model, role="sentry",
                log_path=run_dir / "model_calls.jsonl", max_tokens=512,
                max_calls=args.max_sentry_calls,
            ),
            recorded_encoder, window_size=args.sentry_window, fallback_images=args.fallback_images,
        )
        resources.callback(session.close)
        env = RemoteRoboMemArenaEnv(config, host=args.env_host, port=args.env_port)
        resources.callback(env.close)
        policy = WebsocketPolicyClient(args.policy_host, args.policy_port)
        resources.callback(policy.close)
        _write_json(run_dir / "policy_metadata.json", policy.get_server_metadata())
        outcome = execute_episode(
            env, policy, session, recorder, max_steps=args.max_steps,
            action_chunk_size=args.action_chunk_size, sentry_interval=args.sentry_interval,
            trace_path=run_dir / "trace.jsonl",
        )
    except ModelBudgetExceededError as exc:
        outcome = {"stop_reason": "model_budget", "reason": str(exc)}
    except Exception as exc:
        outcome = {"stop_reason": "infrastructure_or_protocol_error", "error_type": type(exc).__name__}
        print(f"HiMe run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            if session is not None:
                session.memory.save_to_json(str(run_dir / "memory.json"))
            if env is not None:
                evaluation = env.metrics()
                outcome["environment_steps"] = evaluation.get("step")
                write_official_artifacts(
                    run_dir, task_id=args.task, seed=args.seed, prompt=env.task_description,
                    evaluation=evaluation, video_dir=run_dir / "videos",
                )
        except Exception as exc:
            outcome["finalization_error"] = type(exc).__name__
            exit_code = 1
        finally:
            try:
                outcome["videos"] = recorder.close(success=bool(evaluation and evaluation.get("TSR")))
            finally:
                resources.close()
        outcome["elapsed_s"] = time.monotonic() - started
        outcome["valid_rollout"] = exit_code == 0
        _write_json(run_dir / "result.json", outcome)
    print(json.dumps({"run_dir": str(run_dir), "result": outcome, "evaluation": evaluation}, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
