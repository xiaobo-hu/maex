"""Run RoboAgent against pi05_robomemarena and the isolated RoboMemArena service."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ai import SimpleStreamOptions, create_models
from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient
from benchmark.robomemarena.client.adapter import RoboMemArenaRobotPort
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.observation import (
    build_openpi_input,
    build_roboagent_observation,
)
from benchmark.robomemarena.client.observer_recording import RoboMemArenaObserverRecorder
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.client.video_recording import RoboMemArenaVideoRecorder
from benchmark.robomemarena.config import RoboMemArenaConfig
from roboagent import (
    EePoseAdjustmentConfig,
    OpenPiPolicyAdapter,
    RoboAgent,
    RoboAgentOptions,
    TaskExecutionSkill,
    TaskStep,
    load_task_execution_skill,
)
from roboagent.observer import (
    ObservationSink,
    ObservationSource,
    ObserverPort,
    ObserverRunner,
    ObserverSamplingOptions,
    OpenAiObserverAdapter,
    create_observation_stream,
)

_EEF_WORKSPACE_MIN_M = (-0.45, -0.40, 0.30)
_EEF_WORKSPACE_MAX_M = (0.35, 0.65, 1.20)
_DEFAULT_PLAN_REJECTION = (
    "The current step is not complete. Re-evaluate the physical state."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--action-chunk-size", type=int, default=20)
    parser.add_argument("--max-vla-chunks-per-call", type=int, default=24)
    parser.add_argument("--max-agent-turns", type=int, default=24)
    parser.add_argument("--planner-retries", type=int, default=3)
    parser.add_argument("--planner-retry-delay-ms", type=int, default=5_000)
    parser.add_argument(
        "--planner",
        choices=("faux", "glm", "deepseek", "qwen", "openai", "anthropic"),
        default="faux",
    )
    parser.add_argument("--model")
    parser.add_argument("--runtime-revision")
    parser.add_argument("--evaluator-revision")
    parser.add_argument(
        "--data-role",
        choices=("experience_collection", "candidate_gate", "final_evaluation"),
    )
    parser.add_argument(
        "--observer-base-url",
        help="Enable the read-only Observer through this OpenAI-compatible /v1 endpoint.",
    )
    parser.add_argument("--observer-model", default="robo-observer")
    parser.add_argument("--observer-timesteps", type=int, default=5)
    parser.add_argument("--observer-min-new-observations", type=int, default=10)
    parser.add_argument("--observer-gripper-threshold", type=float)
    parser.add_argument(
        "--observer-record-dir",
        type=Path,
        help="Optionally save Observer image windows and assessments locally.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Save the episode, Observer trace and official evaluation as JSON.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Save trace, official-style summaries, and dual-camera videos in one directory.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--eef-workspace-min-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=_EEF_WORKSPACE_MIN_M,
        help="Hard minimum EEF xyz bounds in RoboMemArena world coordinates.",
    )
    parser.add_argument(
        "--eef-workspace-max-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=_EEF_WORKSPACE_MAX_M,
        help="Hard maximum EEF xyz bounds in RoboMemArena world coordinates.",
    )
    parser.add_argument(
        "--task-execution-skill",
        type=Path,
        help="Load one validated global Task-Execution Skill from SKILL.md.",
    )
    parser.add_argument(
        "--interactive-plan-approval",
        action="store_true",
        help="Ask for terminal approval before each valid advance_task_plan call.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if (
        args.action_chunk_size < 1
        or args.max_vla_chunks_per_call < 1
        or args.max_agent_turns < 1
    ):
        raise ValueError("VLA action, chunk and Agent turn budgets must be at least 1")
    if args.planner_retries < 0 or args.planner_retry_delay_ms < 0:
        raise ValueError("planner retry settings must not be negative")
    if args.observer_record_dir is not None and not args.observer_base_url:
        raise ValueError("observer-record-dir requires observer-base-url")
    if args.observer_timesteps < 1 or args.observer_min_new_observations < 1:
        raise ValueError("observer sampling sizes must be at least 1")
    if args.video_fps < 1:
        raise ValueError("video fps must be at least 1")

    config = RoboMemArenaConfig(
        task_id=args.task,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    environment = RemoteRoboMemArenaEnv(config, host=args.env_host, port=args.env_port)
    observer_sink: ObservationSink | None = None
    observer: ObserverRunner | None = None
    if args.observer_base_url:
        observer_sink, observer_source = create_observation_stream()
        observer = _create_observer(args, observer_source)
    video_recorder = (
        RoboMemArenaVideoRecorder(
            args.run_dir / "videos",
            task_id=args.task,
            fps=args.video_fps,
        )
        if args.run_dir is not None
        else None
    )
    ee_pose_adjustment = EePoseAdjustmentConfig(
        workspace_min_m=tuple(args.eef_workspace_min_m),
        workspace_max_m=tuple(args.eef_workspace_max_m),
    )
    robot = RoboMemArenaRobotPort(
        environment,
        build_roboagent_observation,
        observation_sink=observer_sink,
        observation_recorder=video_recorder.publish if video_recorder is not None else None,
        home_workspace=ee_pose_adjustment,
    )
    policy = WebsocketPolicyClient(args.policy_host, args.policy_port)
    video_finalized = False
    try:
        await robot.reset(seed=args.seed)
        instruction = environment.task_description
        if not instruction:
            raise RuntimeError("RoboMemArena environment did not return a task description")
        total_chunks = math.ceil(config.max_steps / args.action_chunk_size)
        faux_execute_calls = math.ceil(total_chunks / args.max_vla_chunks_per_call)
        model, stream_fn = _create_planner(
            args.planner,
            args.model,
            instruction,
            faux_execute_calls,
        )
        task_execution_skill = (
            load_task_execution_skill(args.task_execution_skill)
            if args.task_execution_skill is not None
            else None
        )
        if args.run_dir is not None:
            _write_run_config(
                args.run_dir,
                args,
                instruction,
                model.id,
                task_execution_skill,
            )
            if task_execution_skill is not None:
                _snapshot_task_execution_skill(args.run_dir, task_execution_skill)
        vla = OpenPiPolicyAdapter(
            policy,
            build_openpi_input,
            max_actions=args.action_chunk_size,
        )
        agent = RoboAgent(
            RoboAgentOptions(
                model=model,
                robot=robot,
                vla=vla,
                stream_fn=stream_fn,
                max_turns=(
                    faux_execute_calls + 3
                    if args.planner == "faux"
                    else args.max_agent_turns
                ),
                max_vla_chunks=args.max_vla_chunks_per_call,
                stream_options=_create_planner_stream_options(
                    args.planner,
                    max_retries=args.planner_retries,
                    max_retry_delay_ms=args.planner_retry_delay_ms,
                ),
                observer=observer,
                finalizer=robot,
                ee_pose_adjustment=ee_pose_adjustment,
                task_execution_skill=task_execution_skill,
                plan_advance_approval=(
                    _confirm_plan_advance if args.interactive_plan_approval else None
                ),
            )
        )
        result = await agent.run(instruction)
        # 官方评分只在 episode 结束后通过 evaluation 通道读取，不进入 Agent feedback。
        evaluation = dict(environment.metrics())
        if video_recorder is not None:
            video_recorder.close(success=_metric(evaluation.get("TSR")) > 0)
            video_finalized = True
        output: dict[str, object] = {
            "episode": asdict(result),
            "evaluation": evaluation,
        }
        trace_file = args.trace_file
        if trace_file is None and args.run_dir is not None:
            trace_file = args.run_dir / "trace.json"
        _output_result(output, trace_file)
        if args.run_dir is not None:
            write_official_artifacts(
                args.run_dir,
                task_id=args.task,
                seed=args.seed,
                prompt=instruction,
                evaluation=evaluation,
                video_dir=args.run_dir / "videos",
            )
        if result.status == "error":
            raise SystemExit(1)
    finally:
        if video_recorder is not None and not video_finalized:
            video_recorder.close(success=False)
        policy.close()
        environment.close()


def _create_observer(
    args: argparse.Namespace,
    source: ObservationSource,
) -> ObserverRunner:
    """Compose the read-only Observer and optional local window recorder."""
    observer: ObserverPort = OpenAiObserverAdapter(
        base_url=args.observer_base_url,
        model=args.observer_model,
    )
    if args.observer_record_dir is not None:
        observer = RoboMemArenaObserverRecorder(observer, args.observer_record_dir)
    return ObserverRunner(
        observer,
        source,
        sampling=ObserverSamplingOptions(
            max_timesteps=args.observer_timesteps,
            min_new_observations=args.observer_min_new_observations,
            gripper_transition_threshold=args.observer_gripper_threshold,
        ),
    )


async def _confirm_plan_advance(
    current_step: TaskStep,
    next_step: TaskStep | None,
    latest_reflection: str,
) -> tuple[bool, str]:
    """在显式探索模式中等待终端用户确认一次合法计划推进。"""
    return await asyncio.to_thread(
        _read_plan_advance_approval,
        current_step,
        next_step,
        latest_reflection,
    )


def _read_plan_advance_approval(
    current_step: TaskStep,
    next_step: TaskStep | None,
    latest_reflection: str,
) -> tuple[bool, str]:
    """读取一次终端批准；stdin 不可用时安全拒绝。"""
    next_label = (
        f"{next_step.id}: {next_step.instruction}"
        if next_step is not None
        else "<terminal>"
    )
    print(
        "\n[interactive plan approval]\n"
        f"Current step: {current_step.id}: {current_step.instruction}\n"
        f"Next step: {next_label}\n"
        f"Latest Reflection: {latest_reflection or '<none>'}\n"
        "Agent requests completion of the current step.",
        flush=True,
    )
    while True:
        try:
            answer = input("Approve? [y/n]: ").strip().lower()
        except (EOFError, OSError):
            return False, _DEFAULT_PLAN_REJECTION
        if answer in {"y", "yes"}:
            return True, ""
        if answer in {"n", "no"}:
            try:
                feedback = input("Feedback (Enter for default): ").strip()
            except (EOFError, OSError):
                feedback = ""
            return False, feedback or _DEFAULT_PLAN_REJECTION
        print("Please enter y or n.", flush=True)


def _create_planner(
    planner: str,
    model_id: str | None,
    instruction: str,
    faux_execute_calls: int = 1,
):
    models = create_models()
    if planner == "faux":
        from ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call

        handle = faux_provider()
        create_plan = faux_assistant_message(
            faux_tool_call(
                "create_task_plan",
                {
                    "steps": [
                        {
                            "id": "benchmark_task",
                            "instruction": instruction,
                            "status": "running",
                        }
                    ]
                },
            ),
            stop_reason="toolUse",
        )
        execute_calls = [
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "benchmark_task"}),
                stop_reason="toolUse",
            )
            for _ in range(faux_execute_calls)
        ]
        advance_plan = faux_assistant_message(
            faux_tool_call(
                "advance_task_plan",
                {"completed_step_id": "benchmark_task", "next_step_id": None},
            ),
            stop_reason="toolUse",
        )
        finish = faux_assistant_message(
            faux_tool_call(
                "finish",
                {"success": True, "summary": "VLA rollout completed."},
            ),
            stop_reason="toolUse",
        )
        handle.set_responses([create_plan, *execute_calls, advance_plan, finish])
        provider = handle.provider
        selected = handle.get_model(model_id)
    elif planner == "glm":
        from ai.providers import create_openai_compatible_model, openai_compatible_provider

        selected = create_openai_compatible_model(
            model_id=model_id or "glm-4.6v-flash",
            provider_id="zhipu",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            name="GLM-4.6V-Flash",
            context_window=128_000,
            max_tokens=8_192,
            reasoning=True,
        )
        provider = openai_compatible_provider(
            provider_id="zhipu",
            name="智谱 GLM",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            models=[selected],
            api_key_envs=("ZHIPU_API_KEY",),
        )
    elif planner == "deepseek":
        from ai.providers import create_openai_compatible_model, openai_compatible_provider

        selected = create_openai_compatible_model(
            model_id=model_id
            or os.environ.get("LLM_MODEL_NAME")
            or "deepseek-v4-flash-vision-exp",
            provider_id="deepseek",
            base_url="https://api.deepseek.com",
            name="DeepSeek Vision Planner",
            context_window=1_000_000,
            max_tokens=8_192,
        )
        provider = openai_compatible_provider(
            provider_id="deepseek",
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            models=[selected],
            api_key_envs=("LLM_API_KEY",),
        )
    elif planner == "qwen":
        from ai.providers import create_openai_compatible_model, openai_compatible_provider

        base_url = os.environ.get(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        selected = create_openai_compatible_model(
            model_id=model_id or os.environ.get("LLM_MODEL_NAME") or "qwen3.8-max",
            provider_id="qwen",
            base_url=base_url,
            name="Qwen Vision Planner",
            context_window=128_000,
            max_tokens=8_192,
        )
        provider = openai_compatible_provider(
            provider_id="qwen",
            name="Qwen",
            base_url=base_url,
            models=[selected],
            api_key_envs=("LLM_API_KEY",),
        )
    elif planner == "openai":
        from ai.providers.openai import openai_provider

        provider = openai_provider()
        selected = _select_model(provider, model_id or "gpt-4o")
    else:
        from ai.providers.anthropic import anthropic_provider

        provider = anthropic_provider()
        selected = _select_model(provider, model_id or "claude-sonnet-4-5")
    if selected is None:
        raise ValueError(f"Unknown {planner} model: {model_id}")
    models.set_provider(provider)
    return selected, models.stream_simple


def _create_planner_stream_options(
    planner: str,
    *,
    max_retries: int,
    max_retry_delay_ms: int,
) -> SimpleStreamOptions | None:
    """Build provider-specific streaming options for a visual planner."""
    if planner not in {"glm", "deepseek", "qwen"}:
        return None

    thinking_disabled = os.environ.get("LLM_DISABLE_THINKING", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    extra_body: dict[str, object] | None = None
    if planner == "deepseek":
        extra_body = {"thinking": {"type": "disabled" if thinking_disabled else "enabled"}}
    elif planner == "qwen":
        extra_body = {"enable_thinking": not thinking_disabled}

    return SimpleStreamOptions(
        max_retries=max_retries,
        max_retry_delay_ms=max_retry_delay_ms,
        tool_choice="required",
        extra_body=extra_body,
    )


def _select_model(provider, model_id: str):
    return next((model for model in provider.get_models() if model.id == model_id), None)


def _output_result(output: dict[str, object], trace_file: Path | None) -> None:
    """Print and optionally persist the same episode and evaluation payload."""
    result_json = json.dumps(output, ensure_ascii=False, indent=2)
    print(result_json)
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(f"{result_json}\n", encoding="utf-8")
        print(f"[trace] saved to {trace_file}")


def _write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    instruction: str,
    model_id: str,
    task_execution_skill: TaskExecutionSkill | None = None,
) -> None:
    """Persist reproducibility metadata without copying credentials."""
    run_dir.mkdir(parents=True, exist_ok=True)
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    values.update(
        {
            "task_description": instruction,
            "resolved_model": model_id,
            "started_at": datetime.now(UTC).isoformat(),
            "task_execution_skill_sha256": (
                task_execution_skill.content_sha256
                if task_execution_skill is not None
                else None
            ),
            "task_execution_skill_version": (
                task_execution_skill.version if task_execution_skill is not None else None
            ),
        }
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _snapshot_task_execution_skill(
    run_dir: Path,
    skill: TaskExecutionSkill,
) -> None:
    """把本次在线使用的 Skill 原样复制到运行目录供复现实验。"""
    source = Path(skill.source_path)
    target = run_dir / "task_execution_skill"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("SKILL.md", "PURPOSE.md", "manifest.json"):
        path = source.parent / name
        if path.is_file():
            snapshot = target / name
            shutil.copy2(path, snapshot)
            snapshot.chmod(0o444)


def _metric(value: object) -> float:
    return float(value) if isinstance(value, (int, float, str)) else 0.0


def main() -> None:
    """Run one Agent-controlled RoboMemArena episode."""
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
