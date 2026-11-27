"""基线评测层：连接原生 VoLo 代理，复用 RoboMemArena 客户端、评分与视频。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import numpy as np

from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient, msgpack_numpy
from benchmark.robomemarena.baselines.volo_observation import build_volo_observation
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.observation import build_roboagent_observation
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.client.video_recording import RoboMemArenaVideoRecorder
from benchmark.robomemarena.config import RoboMemArenaConfig
from roboagent.prompts.robomemarena import task_primitive_reference


class EpisodeClock:
    """以首个有效观测为起点，执行墙钟和动作频率预算。"""

    def __init__(self, wall_seconds, control_hz):
        self.wall_seconds = wall_seconds
        self.period = 1 / control_hz
        self.first_observation_at = None
        self.next_action = 0.0

    def remaining(self):
        """返回本回合的剩余墙钟。"""
        left = self.first_observation_at + self.wall_seconds - time.monotonic()
        if left <= 0:
            raise TimeoutError("episode wall budget")
        return left

    def step(self, env, action):
        """在动作边界检查剩余预算，并将动作派发频率限制为 20 Hz。"""
        time.sleep(min(max(0.0, self.next_action - time.monotonic()), self.remaining()))
        self.remaining()
        self.next_action = time.monotonic() + self.period
        return env.step(action)


class TimedVoLoClient(WebsocketPolicyClient):
    """在原协议上限制等待本地 VoLo 代理的时长。"""

    def infer_before(self, observation, clock):
        """发送观测后，最多等待到本局墙钟截止时间。"""
        clock.remaining()
        self._socket.send(self._packer.pack(observation))
        response = self._socket.recv(timeout=clock.remaining())
        if isinstance(response, str):
            raise RuntimeError(f"VoLo proxy failed: {response}")
        return msgpack_numpy.unpackb(response)


def execute_episode(env, policy, recorder, *, max_steps, action_chunk_size, trace_path,
                    enable_depth=False, enable_task_reference=False, clock=None) -> dict:
    """执行单回合；发送显式回合标识与物理步数，不把官方阶段评分传给 VoLo。"""
    raw = env.reset()
    if clock is not None:
        clock.first_observation_at = time.monotonic()
    reference = task_primitive_reference(env.task_description) if enable_task_reference else None
    if reference:
        trace_path.with_name("task_reference.txt").write_text(reference + "\n", encoding="utf-8")
    recorder.publish(build_roboagent_observation(raw))
    episode_id = str(uuid4())
    steps, calls, done = 0, 0, False
    stop = None
    with trace_path.open("x") as trace:
        while not done and steps < max_steps:
            payload = build_volo_observation(raw, env.task_description, enable_depth=enable_depth)
            payload.update({"__episode_id": episode_id, "__step": steps})
            if reference:
                # 静态参考进入任务分解，VLA prompt 仍由 VoLo 的当前子目标决定。
                payload["__task_primitive_reference"] = reference
            response = policy.infer_before(payload, clock) if clock else policy.infer(payload)
            stop = response.get("orchestrator_control_stop")
            if stop is not None:
                trace.write(json.dumps({"event": "control_stop", "step": steps, "at": time.monotonic(), **stop}) + "\n")
                break
            actions = np.asarray(response["actions"])
            if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0
                    or not np.isfinite(actions).all()):
                raise ValueError("VoLo must return nonempty finite actions [N, 7]")
            calls += 1
            selected = actions[:min(action_chunk_size, max_steps - steps)]
            # 每次推理只使用新返回的 chunk，没有跨推理缓存需要 flush。
            trace.write(json.dumps({
                "event": "inference", "episode_id": episode_id, "step": steps, "at": time.monotonic(),
                "policy_call": calls, "actions": selected.tolist(),
                **{key: value for key, value in response.items()
                   if key.startswith("orchestrator_") and not key.endswith("_jpg")},
            }, ensure_ascii=False, default=str) + "\n")
            trace.flush()
            for action in selected:
                result = clock.step(env, action) if clock else env.step(action)
                raw, done = result.observation, result.done
                steps += 1
                recorder.publish(build_roboagent_observation(raw))
                # done 后必须立刻停止，避免额外动作破坏终态。
                if done:
                    break
            trace.write(json.dumps({"event": "chunk_end", "step": steps, "done": done, "at": time.monotonic()}) + "\n")
            trace.flush()
    return {
        "episode_id": episode_id, "policy_action_steps": steps, "policy_calls": calls,
        "stop_reason": stop["reason"] if stop is not None else "environment_done" if done else "action_budget",
        "proxy_stop": stop,
        "task_reference_sha256": hashlib.sha256(reference.encode()).hexdigest() if reference else None,
    }


def main() -> int:
    """运行一个 VoLo episode；错误独立记录，不将连接异常当成有效失败实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--action-chunk-size", type=int, default=5)
    parser.add_argument("--enable-depth", action="store_true", help="Enable calibrated RGB-D for VoLo grasp tools")
    parser.add_argument("--task-reference", action="store_true", help="Give decomposition the shared RoboMemArena primitive reference")
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=8040)
    parser.add_argument("--experiment", type=Path, help="Use the paper plan and its shared evaluation budgets")
    parser.add_argument("--policy-port", type=int)
    parser.add_argument("--observer-port", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action_chunk_size < 1:
        parser.error("--action-chunk-size must be positive")
    config = RoboMemArenaConfig(task_id=args.task, seed=args.seed, max_steps=args.max_steps,
                               enable_depth=args.enable_depth)
    plan = json.loads(args.experiment.read_text()) if args.experiment is not None else None
    clock = EpisodeClock(plan["budgets"]["wall_seconds"], plan["budgets"]["control_hz"]) if plan else None
    args.run_dir.mkdir(parents=True, exist_ok=plan is not None)
    _write_json(args.run_dir / "run_config.json", {
        **vars(args), "baseline": "VoLo-RGBD-RoboMemArena-adaptation" if args.enable_depth else "VoLo-Only-VLA-RoboMemArena-adaptation",
        "environment_config": config.to_reset_params(), "external_skill": None,
        "instruction": ("official task description plus shared static primitive reference for decomposition"
                        if args.task_reference else "official task description without added primitive reference"),
        "action_chunk_protocol": "shared MAEX prefix cap" if plan else "5 by default, matching upstream LIBERO client",
        "paper_config": {
            "planner": plan["planner"]["id"], "observer": plan["observer"]["id"], "policy": plan["policy"]["id"],
            "volo": plan["volo"], "budgets": plan["budgets"],
        } if plan else None,
    })
    env = None
    resources = ExitStack()
    recorder = RoboMemArenaVideoRecorder(args.run_dir / "videos", task_id=args.task,
                                         fps=plan["budgets"]["video_fps"] if plan else 10)
    outcome, evaluation, exit_code = {}, None, 0
    started = time.monotonic()
    try:
        policy_client = TimedVoLoClient if plan else WebsocketPolicyClient
        policy = policy_client(args.proxy_host, args.proxy_port)
        resources.callback(policy.close)
        metadata = policy.get_server_metadata()
        if metadata.get("orchestrator") is not True:
            raise ValueError("--proxy-port must point to VoLo, not directly to a VLA")
        _write_json(args.run_dir / "policy_metadata.json", metadata)
        env = RemoteRoboMemArenaEnv(config, host=args.env_host, port=args.env_port)
        resources.callback(env.close)
        outcome = execute_episode(
            env, policy, recorder, max_steps=args.max_steps,
            action_chunk_size=args.action_chunk_size, trace_path=args.run_dir / "trace.jsonl",
            enable_depth=args.enable_depth,
            enable_task_reference=args.task_reference,
            clock=clock,
        )
        if outcome["stop_reason"] == "infrastructure_error":
            exit_code = 1
    except TimeoutError as exc:
        exit_code = 0 if clock is not None and clock.first_observation_at is not None else 1
        outcome = {"stop_reason": "wall_budget" if exit_code == 0 else "startup_timeout", "error_type": type(exc).__name__}
    except KeyboardInterrupt:
        exit_code = 130
        outcome = {"stop_reason": "interrupted", "error_type": "KeyboardInterrupt"}
    except Exception as exc:
        exit_code = 1
        outcome = {"stop_reason": "infrastructure_or_protocol_error", "error_type": type(exc).__name__}
        print(f"VoLo run failed: {type(exc).__name__}: {exc}")
    finally:
        terminated_at = time.monotonic()
        try:
            if env is not None:
                evaluation = env.metrics()
                write_official_artifacts(
                    args.run_dir, task_id=args.task, seed=args.seed, prompt=env.task_description,
                    evaluation=evaluation, video_dir=args.run_dir / "videos",
                )
        except Exception as exc:
            outcome["finalization_error"] = type(exc).__name__
            exit_code = 1
        finally:
            try:
                outcome["videos"] = recorder.close(success=bool(evaluation and evaluation.get("TSR")))
            finally:
                resources.close()
        outcome.update({"elapsed_s": time.monotonic() - started, "valid_rollout": exit_code == 0})
        _write_json(args.run_dir / "result.json", outcome)
        if clock is not None and clock.first_observation_at is not None and evaluation is not None:
            status = "error" if exit_code else "success" if evaluation["TSR"] else "failure"
            if exit_code == 0 and outcome["stop_reason"] in {"wall_budget", "control_timeout"}:
                status = "timeout"
            journal = args.run_dir / "trace.jsonl"
            _write_json(args.run_dir / "trace.json", {
                "episode": {"status": status, "summary": outcome["stop_reason"], "details": {
                    "journal": [json.loads(line) for line in journal.read_text().splitlines()],
                }},
                "evaluation": evaluation,
                "timing": {"first_observation_at": clock.first_observation_at, "terminated_at": terminated_at},
            })
    print(json.dumps(outcome, ensure_ascii=False))
    return exit_code


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
