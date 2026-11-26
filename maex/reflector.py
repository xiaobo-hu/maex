"""从已归档的 MAEX 窗口生成待闭环核验的程序模板候选。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from benchmark.robomemarena.scripts.roboagent_eval import (
    _create_planner,
    _create_planner_stream_options,
)
from pydantic import BaseModel, ConfigDict, Field

from roboagent.types import Observation

from .models import Planner
from .run_eval import write_json
from .schema import Predicate


class Proposal(BaseModel):
    """在既有谓词下补充可复用的视觉核验问题。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predicate: Predicate
    focus: str = Field(min_length=1, max_length=240)
    source_cases: tuple[str, ...] = Field(min_length=1, max_length=6)


class Proposals(BaseModel):
    """本次由冻结 VLM 归纳的少量候选。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    templates: tuple[Proposal, ...] = Field(min_length=1, max_length=5)


REFLECTION_PROMPT = """Propose reusable visual verification questions for MAEX from executed
episodes. Each proposal must cite existing source_cases with the same predicate. Generalize
object names into the target and destination roles. Use failure and uncertain judgments
to disambiguate the positive case. Keep the question short. The registered effect and
required evidence remain authoritative. Do not invent execution, feedback or evidence IDs."""


def collect_cases(paths: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    """重放轨迹的引用关系，提取真实事件及相邻的反例。"""
    cases = {}
    counterexamples = []
    for path in paths:
        episode = json.loads(path.read_text())["episode"]
        details = episode["details"]
        if details["agent"] != "maex":
            raise ValueError(f"Expected a MAEX trace: {path}")
        journal = details["journal"]
        requests = {event["id"]: event["frames"] for event in journal
                    if event["event"] == "verification_requested"}
        returned = {event["id"]: event["judgments"]["judgments"] for event in journal
                    if event["event"] == "verification_returned"}
        for fact in details["facts"]:
            request_id = fact["request_id"]
            if not fact["refs"] or not set(fact["refs"]) <= requests[request_id].keys():
                raise ValueError(f"Event references cannot be replayed in {path}")
            case_id = f"{details['episode_id']}:{fact['step']}"
            cases[case_id] = {
                "trace": str(path.resolve()), "goal": fact["goal"], "refs": fact["refs"],
                "first": fact["first"], "last": fact["last"], "episode_status": episode["status"],
                "judgments": returned[request_id],
            }
        counterexamples.extend(
            {"goal_step": event["goal_step"], "outcome": event["outcome"], "reason": event["reason"]}
            for event in journal if event["event"] == "evidence_gate" and event["outcome"] != "true"
        )
    if not cases:
        raise ValueError("At least one verified event is required to propose a procedure")
    return cases, counterexamples


async def generate(args: argparse.Namespace) -> None:
    """提议模板并核对来源，保留供独立仿真实验复核的候选。"""
    cases, counterexamples = collect_cases(args.traces)
    model, stream = _create_planner("deepseek", args.model, "")
    options = _create_planner_stream_options("deepseek", max_retries=3, max_retry_delay_ms=5_000)
    planner = Planner(model, stream, options, timeout=60)
    proposals, usage = await planner.ask(
        "propose_procedure_templates", Proposals, REFLECTION_PROMPT,
        json.dumps({"verified_cases": cases, "counterexamples": counterexamples}, ensure_ascii=False),
        Observation(),
    )
    for proposal in proposals.templates:
        if any(case not in cases or cases[case]["goal"]["predicate"] != proposal.predicate
               for case in proposal.source_cases):
            raise ValueError(f"The proposed {proposal.predicate} template cites an incompatible case")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "review_status": "closed_loop_review_required", "model": model.id, "usage": usage,
        "proposals": proposals.templates, "verified_cases": cases, "counterexamples": counterexamples,
    })


def main() -> None:
    """运行一次离线经验整理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
