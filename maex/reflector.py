"""重放 MAEX 轨迹，发布有执行证据的经验及待审查的 VLM 候选。"""

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
from .procedural import (
    REUSABLE_ACTIONS,
    CaseOutcome,
    EpisodeExperience,
    ProcedureStore,
    RecoveryCase,
    RecoveryCondition,
    TaskContext,
    goal_roles,
)
from .schema import Goal, Outcome, Plan, Predicate


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


def replay_facts(details: dict, path: Path) -> dict[int, dict]:
    """从事件账本重放准入，核对现有回复及窗口，不读取离线评测标签。"""
    journal = details["journal"]
    requests = {event["id"]: event for event in journal if event["event"] == "verification_requested"}
    replies = {event["id"]: event for event in journal if event["event"] == "verification_returned"}
    admitted = {
        (event["fact"]["step"], event["fact"]["request_id"]): event["fact"]
        for event in journal if event["event"] == "evidence_gate" and event["admitted"]
    }
    facts = {}
    for fact in details["facts"]:
        step, request_id = fact["step"], fact["request_id"]
        request = requests.get(request_id)
        reply = replies.get(request_id)
        supported = {
            ref for item in reply["judgments"]["judgments"] if item["step"] == step and item["outcome"] == "true"
            for support in item["support"] for ref in support["refs"]
        } if reply is not None else set()
        if (admitted.get((step, request_id)) != fact or request is None or reply is None
                or request["epoch"] != fact["execution_id"] or step not in request["goal_steps"]
                or not fact["refs"] or not set(fact["refs"]) <= request["frames"].keys()
                or not set(fact["refs"]) <= supported):
            raise ValueError(f"Cannot replay MAEX fact {step}:{request_id} in {path}")
        facts[step] = fact
    return facts


def recoveries_from_trace(journal: list[dict], facts: dict[int, dict]) -> tuple[RecoveryCase, ...]:
    """按恢复请求分段，匹配真正派发的动作、反馈和随后同一步的结果。"""
    active = {}
    checks = {}
    boundaries = []
    for index, event in enumerate(journal):
        event_type = event["event"]
        if event_type == "intent_committed":
            active[event["epoch"]] = event
        elif event_type in {"effect_reopened", "pickup_reopened"}:
            checks.pop(event["goal_step"], None)
        elif event_type == "evidence_gate" and event["outcome"] != "unknown":
            step = event["goal_step"]
            if step not in checks or checks[step]["observed_at_step"] <= event["observed_at_step"]:
                checks[step] = event
        elif event_type == "recovery_requested":
            step = event["goal_step"]
            check = checks.get(step)
            outcome = (Outcome(check["outcome"]) if check and check["observed_at_step"] == event["physical_step"]
                       else Outcome.UNKNOWN)
            boundaries.append((index, active[event["epoch"]], outcome))

    cases = []
    for position, (index, intent, outcome) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(journal)
        request = journal[index]
        step = intent["step"]
        start = request["physical_step"]
        goal = Goal.model_validate(intent["goal"])
        window = journal[index + 1:end]
        prepared = next((event for event in window if event["event"] == "recovery_prepared"
                         and event["epoch"] == intent["epoch"] and event["physical_step"] == start), None)
        if prepared is None:
            continue
        choices = {prepared["decision"]["action"]}
        choices.update(event["action"] for event in window if event["event"] == "recovery_adjusted"
                       and event["physical_step"] == start)
        commit_index = next((i for i, e in enumerate(window) if e["event"] == "intent_committed"), len(window))
        if commit_index == len(window):
            continue
        committed = window[commit_index]
        if (committed["mode"] not in REUSABLE_ACTIONS & choices or committed["step"] != step
                or committed["start_step"] != start or committed["goal"] != intent["goal"]):
            continue
        after = window[commit_index + 1:]
        # 同一目标的策略可接续恢复。其他准备或意图开始后不再归到本次恢复。
        stop = next((i for i, event in enumerate(after) if event["event"] == "intent_committed"
                     and (event["mode"] != "policy" or event["step"] != step or event["goal"] != intent["goal"])),
                    len(after))
        after = after[:stop]
        executed = next((event for event in after if event["event"] == "chunk_finished"
                         and event["task"]["epoch"] == committed["epoch"] and event["end_step"] > start), None)
        if executed is None:
            continue
        condition = RecoveryCondition.observe(
            goal, intent["mode"], Observation(robot_state=request["robot_state"]), outcome,
            request["lost_pickup"] is not None, request["blocked_by"],
        )
        known = facts.get(step)
        epochs = {committed["epoch"]} | {event["epoch"] for event in after if event["event"] == "intent_committed"}
        proof = next((
            event for event in after if known is not None and known["execution_id"] in epochs
            and event["event"] == "evidence_gate" and event["goal_step"] == step
            and event["admitted"] and event["fact"] == known and known["first"] > start
            and goal_roles(Goal.model_validate(known["goal"])) == goal_roles(goal)
        ), None)
        contradiction = next((
            event for event in reversed(after) if event["event"] == "evidence_gate" and event["goal_step"] == step
            and event["outcome"] == "false" and event["observed_at_step"] >= executed["end_step"]
        ), None)
        result: CaseOutcome = "unresolved"
        if executed["status"] == "safety_stop":
            result = "contradicted"
        elif proof:
            result = "verified"
        elif contradiction:
            result = "contradicted"
        elif executed["status"] in {"timeout", "cancelled"} or any(
            event["event"] == "terminate" and event["status"] in {"timeout", "cancelled"} for event in after
        ):
            result = "interrupted"
        cases.append(RecoveryCase(
            step=step, goal=goal, condition=condition, action=committed["mode"], start=start, end=executed["end_step"],
            outcome=result,
            result_step=proof["observed_at_step"] if proof else contradiction["observed_at_step"] if contradiction else None,
            evidence_request=known["request_id"] if proof else None,
            evidence_refs=tuple(known["refs"]) if proof else (),
        ))
    return tuple(cases)


def confirmed_plan(plan: Plan, facts: dict[int, dict]) -> bool:
    """核对已执行的有序角色，保留依赖本局观测的计划槽位。"""
    if set(facts) != set(range(len(plan.goals))):
        return False
    drawers = plan.drawer_selection.candidates if plan.drawer_selection is not None else ()
    for step, goal in enumerate(plan.goals):
        observed = Goal.model_validate(facts[step]["goal"])
        targets = drawers if goal.target == "$drawer" else (goal.target,)
        if goal.destination_from is not None:
            destinations = (facts[goal.destination_from]["goal"]["destination"],)
        else:
            destinations = drawers if goal.destination == "$drawer" else (goal.destination,)
        if (goal.predicate != observed.predicate or observed.target not in targets
                or observed.destination not in destinations or goal.occurrence != observed.occurrence):
            return False
    return True


def publish_trace(store: ProcedureStore, path: Path) -> int | None:
    """将已执行的候选及反例写入后续 episode 可读取的版本。"""
    episode = json.loads(path.read_text())["episode"]
    details = episode["details"]
    if details["agent"] != "maex":
        raise ValueError(f"Expected a MAEX trace: {path}")
    if episode["status"] == "error":
        return None
    settings = json.loads(path.with_name("run_config.json").read_text())
    if settings["procedures"]["version"] != details["procedure_version"]:
        raise ValueError(f"Procedure version does not match the trace in {path}")
    facts = replay_facts(details, path)
    journal = details["journal"]
    recoveries = recoveries_from_trace(journal, facts)
    proposal = next((event["plan"] for event in journal if event["event"] == "global_plan_prepared"), None)
    plan = Plan.model_validate(proposal) if episode["status"] == "success" and proposal is not None else None
    if plan is not None and (
        not confirmed_plan(plan, facts)
        or not any(event["event"] == "chunk_finished" and event["end_step"] > event["start_step"]
                   for event in journal)
    ):
        plan = None
    if plan is None and not recoveries:
        return None
    return store.publish(EpisodeExperience(
        episode_id=details["episode_id"], task=TaskContext(
            task_id=settings["task"], instruction=settings["task_description"], seed=settings["seed"],
        ),
        executor=settings["executor"], library_version=details["procedure_version"],
        trace=str(path.resolve()), plan=plan, recoveries=recoveries,
    ))


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
        returned = {event["id"]: event["judgments"]["judgments"] for event in journal
                    if event["event"] == "verification_returned"}
        for fact in replay_facts(details, path).values():
            request_id = fact["request_id"]
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
    candidate_report = {
        "review_status": "closed_loop_review_required", "model": model.id, "usage": usage,
        "proposals": [item.model_dump(mode="json") for item in proposals.templates],
        "verified_cases": cases, "counterexamples": counterexamples,
    }
    args.output.write_text(json.dumps(candidate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """运行一次离线经验整理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--output", type=Path, help="Write VLM suggestions for offline review")
    parser.add_argument("--experience-db", type=Path, help="Publish replayed cases to the procedure store")
    args = parser.parse_args()
    if args.output is None and args.experience_db is None:
        parser.error("Provide --experience-db to publish cases or --output for VLM suggestions")
    if args.experience_db is not None:
        store = ProcedureStore(args.experience_db)
        for path in args.traces:
            revision = publish_trace(store, path)
            print(f"{path} · revision {revision}" if revision is not None else f"{path} · no reusable case")
    if args.output is not None:
        asyncio.run(generate(args))


if __name__ == "__main__":
    main()
