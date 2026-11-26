"""使用现有 RoboMemArena 服务运行独立的 MAEX agent。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from benchmark.robomemarena.client.adapter import RoboMemArenaRobotPort
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.observation import build_roboagent_observation
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.client.video_recording import RoboMemArenaVideoRecorder
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.scripts.roboagent_eval import (
    _create_planner,
    _create_planner_stream_options,
)
from openai import AsyncOpenAI
from pydantic import BaseModel
from websockets.asyncio.client import connect

from roboagent.observer.types import ObservationEvent
from roboagent.utils.ee_pose import EePoseAdjustmentConfig

from . import MAEX_VERSION
from .executive import Executive, Limits
from .executor import PI05, Executor, OpenPiPolicy
from .memory import ProcedureLibrary
from .models import Observer, Planner
from .procedural import ProcedureStore, TaskContext
from .reflector import publish_trace


def parse_args() -> argparse.Namespace:
    """读取官方任务、模型服务和 MAEX 的预算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument("--observer-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--observer-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--planner", choices=("deepseek", "qwen", "glm"), default="deepseek")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--planner-retries", type=int, default=3)
    parser.add_argument("--action-chunk-size", type=int, default=20)
    parser.add_argument("--observer-timesteps", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--wall-seconds", type=float, default=900)
    parser.add_argument("--max-planner-calls", type=int, default=20)
    parser.add_argument("--max-observer-calls", type=int, default=100)
    parser.add_argument("--verify-stride", type=int, default=45)
    parser.add_argument("--attempt-chunks", type=int, default=Limits().attempt_chunks)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--procedures", type=Path, default=Path(__file__).with_name("procedures.json"))
    parser.add_argument("--experience-db", type=Path, help="Read a persistent procedural memory")
    parser.add_argument("--experience-revision", type=int, help="Pin the procedural memory revision for this episode")
    parser.add_argument("--learn-procedures", action="store_true", help="Publish replayed experience after the episode")
    args = parser.parse_args()
    if args.experience_db is None and (args.experience_revision is not None or args.learn_procedures):
        parser.error("--experience-revision and --learn-procedures require --experience-db")
    return args


async def run(args: argparse.Namespace) -> int:
    """连接冻结模型，执行后在独立评测通道读取官方分数。"""
    config = RoboMemArenaConfig(task_id=args.task, seed=args.seed, max_steps=args.max_steps)
    limit = Limits(
        max_steps=args.max_steps, wall_seconds=args.wall_seconds,
        planner_calls=args.max_planner_calls, observer_calls=args.max_observer_calls,
        verify_stride=args.verify_stride, attempt_chunks=args.attempt_chunks,
        timesteps=args.observer_timesteps, completion_wait_steps=config.post_goal_steps,
    )
    if not 1 <= args.action_chunk_size <= 20 or min(
        limit.wall_seconds, limit.planner_calls, limit.observer_calls,
        limit.verify_stride, limit.attempt_chunks,
    ) <= 0:
        raise ValueError("MAEX requires positive budgets and an action prefix of at most 20")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "trace.json").exists():
        raise FileExistsError(f"Episode output already exists in {args.run_dir}")
    library = ProcedureLibrary.load(args.procedures)
    store = ProcedureStore(args.experience_db) if args.experience_db is not None else None
    environment = RemoteRoboMemArenaEnv(config, port=args.env_port)
    video = RoboMemArenaVideoRecorder(args.run_dir / "videos", task_id=args.task, fps=args.video_fps)
    workspace = EePoseAdjustmentConfig(
        workspace_min_m=(-0.45, -0.40, 0.30), workspace_max_m=(0.35, 0.65, 1.20),
    )
    robot = RoboMemArenaRobotPort(environment, build_roboagent_observation,
                                 observation_recorder=video.publish, home_workspace=workspace)
    success = False
    try:
        first = ObservationEvent(0, monotonic(), await robot.reset(seed=args.seed))
        instruction = environment.task_description
        procedures = None
        if store is not None:
            procedures = await asyncio.to_thread(
                store.snapshot, TaskContext(task_id=args.task, instruction=instruction, seed=args.seed),
                library.executor, library.version, args.experience_revision,
            )
        model, stream = _create_planner(args.planner, args.model, instruction)
        options = _create_planner_stream_options(
            args.planner, max_retries=args.planner_retries, max_retry_delay_ms=5_000,
        )
        started_at = datetime.now(UTC).isoformat()
        control_hz = 20
        settings = {
            "agent": "maex", "maex_version": MAEX_VERSION,
            "task": args.task, "seed": args.seed, "task_description": instruction,
            "planner": args.planner, "model": model.id, "observer_model": args.observer_model,
            "observer_base_url": args.observer_base_url, "executor": PI05,
            "policy_port": args.policy_port, "env_port": args.env_port,
            "control_hz": control_hz, "video_fps": args.video_fps,
            "action_chunk_size": args.action_chunk_size, "workspace": asdict(workspace),
            "procedures": library.model_dump(mode="json"), "limits": asdict(limit),
            "started_at": started_at,
        }
        if procedures is not None:
            settings["procedural_memory"] = {
                "database": str(store.path.resolve()), "revision": procedures.revision, "learn": args.learn_procedures,
                "plan_source": procedures.exemplar.episode_id if procedures.exemplar is not None else None,
                "recovery_candidates": len(procedures.recoveries),
            }
        write_json(args.run_dir / "run_config.json", settings)
        async with (
            connect(
                f"ws://127.0.0.1:{args.policy_port}", max_size=None, compression=None, open_timeout=60,
            ) as socket,
            AsyncOpenAI(base_url=args.observer_base_url, api_key="EMPTY", timeout=35, max_retries=0) as client,
        ):
            await asyncio.wait_for(socket.recv(), timeout=60)
            executor = Executor(robot, OpenPiPolicy(socket, args.action_chunk_size, 45),
                                first, workspace, control_hz)
            exemplar = procedures.exemplar if procedures is not None else None
            planner = Planner(model, stream, options, timeout=60,
                              task_example=exemplar.plan if exemplar is not None else None)
            executive = Executive(planner, Observer(client, args.observer_model), executor, library, limit,
                                  procedures=procedures)
            print(f"MAEX task {args.task} seed {args.seed} · {instruction}", flush=True)
            result = await executive.run(instruction)
        evaluation = dict(environment.metrics())
        success = int(evaluation["TSR"]) == 1
        trace_path = args.run_dir / "trace.json"
        write_json(trace_path, {"episode": asdict(result), "evaluation": evaluation})
        write_official_artifacts(
            args.run_dir, task_id=args.task, seed=args.seed, prompt=instruction,
            evaluation=evaluation, video_dir=args.run_dir / "videos",
        )
        if store is not None and args.learn_procedures:
            revision = await asyncio.to_thread(publish_trace, store, trace_path)
            print(f"Procedural memory · revision {revision}" if revision is not None
                  else "Procedural memory · no reusable case")
        print(f"{result.status} · {result.summary} · TSR {evaluation['TSR']} · CSR {evaluation['CSR']:.2f}")
        print(args.run_dir / "trace.json")
        return int(result.status == "error")
    finally:
        video.close(success=success)
        environment.close()


def write_json(path: Path, value: object) -> None:
    """用项目的 JSON 轨迹口径输出结构化事件。"""
    def encode(item: object) -> dict:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"{type(item).__name__} is not trace-serializable")

    path.write_text(json.dumps(value, default=encode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """执行一次 RoboMemArena episode。"""
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
