"""Run RoboAgent's execute_vla tool against isolated LIBERO and pi05_libero services."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path

from ai import SimpleStreamOptions, create_models
from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient
from benchmark.libero.adapter import LiberoRobotPort
from benchmark.libero.config import LiberoEvalConfig, LiberoTaskSpec
from benchmark.libero.observation import build_openpi_input, build_roboagent_observation
from benchmark.libero.observer_recording import LiberoObserverRecorder
from benchmark.libero.remote_env import RemoteLiberoEnv
from roboagent import EpisodeResult, OpenPiPolicyAdapter, RoboAgent, RoboAgentOptions
from roboagent.observer import (
    ObservationSink,
    ObservationSource,
    ObserverPort,
    ObserverRunner,
    ObserverSamplingOptions,
    OpenAiObserverAdapter,
    create_observation_stream,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-vla-chunks-per-call", type=int, default=24)
    parser.add_argument("--planner-retries", type=int, default=3)
    parser.add_argument("--planner-retry-delay-ms", type=int, default=5_000)
    parser.add_argument(
        "--planner",
        choices=("faux", "glm", "openai", "anthropic"),
        default="faux",
    )
    parser.add_argument("--model")
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
        help="Save the complete episode result and tool trace as JSON.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if args.replan_steps < 1 or args.max_vla_chunks_per_call < 1:
        raise ValueError("VLA action and chunk budgets must be at least 1")
    if args.planner_retries < 0 or args.planner_retry_delay_ms < 0:
        raise ValueError("planner retry settings must not be negative")
    if args.observer_record_dir is not None and not args.observer_base_url:
        raise ValueError("observer-record-dir requires observer-base-url")
    if args.observer_timesteps < 1 or args.observer_min_new_observations < 1:
        raise ValueError("observer sampling sizes must be at least 1")
    config = LiberoEvalConfig(
        task=LiberoTaskSpec(args.suite, args.task, args.seed, args.init_state),
        warmup_steps=args.warmup_steps,
    )
    environment = RemoteLiberoEnv(config, host=args.env_host, port=args.env_port)
    observer_sink: ObservationSink | None = None
    observer: ObserverRunner | None = None
    if args.observer_base_url:
        observer_sink, observer_source = create_observation_stream()
        observer = _create_observer(args, observer_source)
    robot = LiberoRobotPort(
        environment,
        build_roboagent_observation,
        observation_sink=observer_sink,
    )
    policy = WebsocketPolicyClient(args.policy_host, args.policy_port)
    try:
        await robot.reset(seed=args.seed)
        instruction = environment.task_description
        if not instruction:
            raise RuntimeError("LIBERO environment did not return a task description")
        total_chunks = math.ceil(config.resolved_max_steps() / args.replan_steps)
        faux_execute_calls = math.ceil(total_chunks / args.max_vla_chunks_per_call)
        model, stream_fn = _create_planner(
            args.planner,
            args.model,
            instruction,
            faux_execute_calls,
        )
        vla = OpenPiPolicyAdapter(policy, build_openpi_input, max_actions=args.replan_steps)
        agent = RoboAgent(
            RoboAgentOptions(
                model=model,
                robot=robot,
                vla=vla,
                stream_fn=stream_fn,
                max_turns=faux_execute_calls + 3 if args.planner == "faux" else 12,
                max_vla_chunks=args.max_vla_chunks_per_call,
                stream_options=(
                    SimpleStreamOptions(
                        max_retries=args.planner_retries,
                        max_retry_delay_ms=args.planner_retry_delay_ms,
                        tool_choice="required",
                    )
                    if args.planner == "glm"
                    else None
                ),
                observer=observer,
            )
        )
        result = await agent.run(instruction)
        _output_result(result, args.trace_file)
        if result.status == "error":
            raise SystemExit(1)
    finally:
        policy.close()
        environment.close()


def _create_observer(
    args: argparse.Namespace,
    source: ObservationSource,
) -> ObserverRunner:
    observer: ObserverPort = OpenAiObserverAdapter(
        base_url=args.observer_base_url,
        model=args.observer_model,
    )
    if args.observer_record_dir is not None:
        observer = LiberoObserverRecorder(observer, args.observer_record_dir)
    return ObserverRunner(
        observer,
        source,
        sampling=ObserverSamplingOptions(
            max_timesteps=args.observer_timesteps,
            min_new_observations=args.observer_min_new_observations,
            gripper_transition_threshold=args.observer_gripper_threshold,
        ),
    )


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


def _select_model(provider, model_id: str):
    return next((model for model in provider.get_models() if model.id == model_id), None)


def _output_result(result: EpisodeResult, trace_file: Path | None) -> None:
    """格式化展示 episode，并按需保存同一份 JSON trace。"""
    result_json = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    print(result_json)
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(f"{result_json}\n", encoding="utf-8")
        print(f"[trace] saved to {trace_file}")


def main() -> None:
    """Run one Agent-controlled LIBERO episode."""
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
