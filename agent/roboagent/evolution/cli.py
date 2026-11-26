"""提供 RoboAgent-Evolve V3 的单一命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from ..skills import load_task_execution_skill
from .compiler import (
    compile_batch,
    compile_paired_comparisons,
    compile_reviewed_events,
    compile_run,
)
from .gate import compare_rollouts, promote_candidate, record_gate_report
from .replay import create_model_decider, evaluate_replay, read_replay_cases
from .workspace import initialize_workspace, validate_workspace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboAgent-Evolve V3 offline workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize an evolution workspace")
    _workspace_argument(init_parser)

    compile_parser = subparsers.add_parser("compile", help="Compile rollout traces")
    _workspace_argument(compile_parser)
    compile_source = compile_parser.add_mutually_exclusive_group(required=True)
    compile_source.add_argument("--run-dir", action="append", type=Path)
    compile_source.add_argument("--batch-manifest", type=Path)
    compile_parser.add_argument(
        "--policy-role",
        choices=("baseline", "candidate", "active"),
        help="Required with --batch-manifest.",
    )

    pairs_parser = subparsers.add_parser(
        "compile-pairs", help="Compile reviewed paired Episode comparisons"
    )
    _workspace_argument(pairs_parser)
    pairs_parser.add_argument("--manifest", type=Path, required=True)

    events_parser = subparsers.add_parser(
        "compile-events", help="Compile reviewed high-value Experience events"
    )
    _workspace_argument(events_parser)
    events_parser.add_argument("--manifest", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate Wiki and Skill links")
    _workspace_argument(validate_parser)

    replay_parser = subparsers.add_parser("replay", help="Run historical decision replay")
    _workspace_argument(replay_parser)
    replay_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate Skill directory or SKILL.md.",
    )
    replay_parser.add_argument(
        "--planner",
        choices=("glm", "deepseek", "qwen", "openai", "anthropic"),
        required=True,
    )
    replay_parser.add_argument("--model", required=True)
    replay_parser.add_argument(
        "--baseline-skill",
        type=Path,
        help="Skill directory or SKILL.md used as the replay baseline; defaults to Active.",
    )
    replay_parser.add_argument(
        "--cases",
        type=Path,
        help="Versioned replay JSONL; defaults to workspace replay/cases.jsonl.",
    )
    replay_parser.add_argument("--planner-retries", type=int, default=3)
    replay_parser.add_argument("--planner-retry-delay-ms", type=int, default=5000)

    compare_parser = subparsers.add_parser(
        "compare-rollouts", help="Compare paired held-out rollouts"
    )
    _workspace_argument(compare_parser)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument(
        "--phase",
        choices=("candidate-gate", "final-evaluation"),
        default="candidate-gate",
    )

    promote_parser = subparsers.add_parser("promote", help="Promote a validated candidate")
    _workspace_argument(promote_parser)
    promote_parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args()


def _workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, required=True)


async def _run_replay(args: argparse.Namespace) -> dict[str, object]:
    from benchmark.robomemarena.scripts.roboagent_eval import (
        _create_planner,
        _create_planner_stream_options,
    )

    root = args.workspace.expanduser().resolve()
    errors = validate_workspace(root)
    if errors:
        raise ValueError("invalid evolution workspace: " + "; ".join(errors))
    candidate_file = _skill_file(args.candidate)
    candidate_path = candidate_file.parent
    candidate = load_task_execution_skill(candidate_file)
    active_path = root / "skills/active/SKILL.md"
    baseline_path = _skill_file(args.baseline_skill) if args.baseline_skill else active_path
    baseline = load_task_execution_skill(baseline_path) if baseline_path.is_file() else None
    cases_path = (args.cases or root / "replay/cases.jsonl").expanduser().resolve()
    cases = read_replay_cases(cases_path)
    model, stream_fn = _create_planner(args.planner, args.model, "")
    stream_options = _create_planner_stream_options(
        args.planner,
        max_retries=args.planner_retries,
        max_retry_delay_ms=args.planner_retry_delay_ms,
    )
    report = await evaluate_replay(
        cases,
        baseline_skill=baseline,
        candidate_skill=candidate,
        decide=create_model_decider(model, stream_fn, stream_options),
    )
    report["case_source"] = str(cases_path)
    report["baseline_source"] = str(baseline_path) if baseline is not None else None
    record_gate_report(root, candidate_path, "replay", report)
    return cast(dict[str, object], report)


def _skill_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved if resolved.name == "SKILL.md" else resolved / "SKILL.md"


def main() -> None:
    """执行一个离线 evolution 子命令。"""
    args = _parse_args()
    if args.command == "init":
        result: object = {"workspace": str(initialize_workspace(args.workspace))}
    elif args.command == "compile":
        initialize_workspace(args.workspace)
        if args.batch_manifest is not None:
            if args.policy_role is None:
                raise ValueError("--policy-role is required with --batch-manifest")
            result = compile_batch(
                args.batch_manifest,
                args.workspace,
                policy_role=args.policy_role,
            )
        else:
            if args.policy_role is not None:
                raise ValueError("--policy-role requires --batch-manifest")
            result = [compile_run(run_dir, args.workspace) for run_dir in args.run_dir]
    elif args.command == "validate":
        errors = validate_workspace(args.workspace)
        result = {"valid": not errors, "errors": errors}
    elif args.command == "compile-pairs":
        initialize_workspace(args.workspace)
        result = compile_paired_comparisons(args.manifest, args.workspace)
    elif args.command == "compile-events":
        initialize_workspace(args.workspace)
        result = compile_reviewed_events(args.manifest, args.workspace)
    elif args.command == "replay":
        result = asyncio.run(_run_replay(args))
    elif args.command == "compare-rollouts":
        result = compare_rollouts(
            args.workspace,
            args.candidate,
            args.manifest,
            phase=args.phase,
        )
        record_gate_report(args.workspace, args.candidate, args.phase, result)
    elif args.command == "promote":
        result = promote_candidate(args.workspace, args.candidate)
    else:
        raise AssertionError(f"Unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("valid") is False:
        raise SystemExit(1)


__all__ = ["main"]
