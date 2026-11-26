"""在不执行机器人动作的情况下重放历史 Planner 决策。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent_core import (
    Agent,
    AgentOptions,
    AgentState,
    AgentToolResult,
    ToolExecutionStartEvent,
)
from agent_core.types import AgentToolUpdateCallback, StreamFn
from ai import AssistantMessage, Model, SimpleStreamOptions, TextContent, UserMessage

from ..prompts import DEFAULT_SYSTEM_PROMPT
from ..skills import TaskExecutionSkill, apply_task_execution_skill
from ..tools import (
    AdjustEePoseTool,
    AdvanceTaskPlanTool,
    CloseGripperTool,
    CreateTaskPlanTool,
    ExecuteVlaTool,
    FinishTool,
    OpenGripperTool,
    RepairTaskPlanTool,
    RestageVlaTool,
    ReturnToHomeTool,
)
from .schema import AllowedAction, ReplayCase, ReplayDecision

ReplayDecider = Callable[
    [ReplayCase, TaskExecutionSkill | None],
    Awaitable[ReplayDecision],
]


@dataclass(slots=True)
class _ReplayTool:
    """保留在线工具 schema、但不产生任何物理副作用的 Replay stub。"""

    name: str
    label: str
    description: str
    parameters: dict[str, object]
    prepare_arguments = None
    execution_mode = "sequential"

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, params, cancel_event, on_update
        return AgentToolResult(content=[TextContent(text="Replay tool was not executed.")])


def replay_tools() -> list[_ReplayTool]:
    """返回与当前 RoboAgent 决策回合一致的工具描述和参数 schema。"""
    classes = (
        CreateTaskPlanTool,
        ExecuteVlaTool,
        RestageVlaTool,
        AdjustEePoseTool,
        CloseGripperTool,
        OpenGripperTool,
        AdvanceTaskPlanTool,
        RepairTaskPlanTool,
        ReturnToHomeTool,
        FinishTool,
    )
    return [
        _ReplayTool(
            name=tool.name,
            label=tool.label,
            description=tool.description,
            parameters=tool.parameters,
        )
        for tool in classes
    ]


def create_model_decider(
    model: Model,
    stream_fn: StreamFn,
    stream_options: SimpleStreamOptions | None,
) -> ReplayDecider:
    """用在线同款 Planner 和工具 schema 创建一个无物理执行的 decider。"""

    async def decide(
        case: ReplayCase,
        skill: TaskExecutionSkill | None,
    ) -> ReplayDecision:
        selected: list[ReplayDecision] = []
        agent: Agent[object] = Agent(
            AgentOptions(
                initial_state=AgentState(
                    model=model,
                    system_prompt=apply_task_execution_skill(
                        DEFAULT_SYSTEM_PROMPT,
                        skill,
                    ),
                    tools=replay_tools(),
                ),
                stream_fn=stream_fn,
                stream_options=stream_options,
                tool_execution="sequential",
                should_stop_after_turn=lambda _context: True,
            )
        )

        def record(event: object) -> None:
            if isinstance(event, ToolExecutionStartEvent) and not selected:
                selected.append(
                    ReplayDecision(tool=event.tool_name, arguments=dict(event.args))
                )

        agent.subscribe(record)
        await agent.prompt(UserMessage(content=_case_prompt(case)))
        if not selected:
            if agent.state.error_message:
                raise RuntimeError(
                    f"Planner request failed for replay case {case.case_id}: "
                    f"{agent.state.error_message}"
                )
            response = next(
                (
                    message
                    for message in reversed(agent.state.messages)
                    if isinstance(message, AssistantMessage)
                ),
                None,
            )
            response_text = "\n".join(
                block.text for block in response.content if isinstance(block, TextContent)
            ).strip() if response is not None else ""
            return ReplayDecision(
                tool="__no_tool_call__",
                arguments={
                    "stop_reason": response.stop_reason if response is not None else "unknown",
                    "text": response_text,
                },
            )
        return selected[0]

    return decide


async def evaluate_replay(
    cases: list[ReplayCase],
    *,
    baseline_skill: TaskExecutionSkill | None,
    candidate_skill: TaskExecutionSkill,
    decide: ReplayDecider,
) -> dict[str, object]:
    """比较 baseline/candidate，并执行严格改善和零回归 Gate。"""
    if not cases:
        raise ValueError("Replay Gate requires at least one case")
    rows: list[dict[str, object]] = []
    baseline_score = 0
    candidate_score = 0
    regressions: list[str] = []
    for case in cases:
        baseline = await decide(case, baseline_skill)
        candidate = await decide(case, candidate_skill)
        baseline_ok = decision_matches(baseline, case.allowed_actions)
        candidate_ok = decision_matches(candidate, case.allowed_actions)
        baseline_score += int(baseline_ok)
        candidate_score += int(candidate_ok)
        if baseline_ok and not candidate_ok:
            regressions.append(case.case_id)
        rows.append(
            {
                "case_id": case.case_id,
                "baseline": baseline.summary(),
                "baseline_correct": baseline_ok,
                "candidate": candidate.summary(),
                "candidate_correct": candidate_ok,
            }
        )
    passed = candidate_score > baseline_score and not regressions
    return {
        "schema_version": 1,
        "candidate": candidate_skill.summary(),
        "baseline": baseline_skill.summary() if baseline_skill is not None else None,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "case_count": len(cases),
        "regressions": regressions,
        "passed": passed,
        "cases": rows,
    }


def decision_matches(
    decision: ReplayDecision,
    allowed_actions: tuple[AllowedAction, ...],
) -> bool:
    """忽略 decision_summary，匹配工具名和必要参数子集。"""
    arguments = {
        key: value for key, value in decision.arguments.items() if key != "decision_summary"
    }
    return any(
        decision.tool == allowed.tool
        and _contains(arguments, allowed.args_contains)
        for allowed in allowed_actions
    )


def read_replay_cases(path: str | Path) -> list[ReplayCase]:
    """读取 JSONL replay cases。"""
    cases: list[ReplayCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(ReplayCase.from_dict(json.loads(line)))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid replay case at line {number}: {error}") from error
    return cases


def _case_prompt(case: ReplayCase) -> str:
    return (
        "This is a historical decision replay. The physical state is fixed; choose exactly "
        "one next tool from the supplied evidence.\n\n"
        f"Task:\n{case.task}\n\n"
        f"Current TaskPlan:\n{json.dumps(case.task_plan, ensure_ascii=False)}\n\n"
        "EpisodeMemory:\n"
        f"{json.dumps(case.episode_memory, ensure_ascii=False)}\n\n"
        f"Latest Grounded State Reflection:\n{case.latest_reflection}"
    )


def _contains(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _contains(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


__all__ = [
    "create_model_decider",
    "decision_matches",
    "evaluate_replay",
    "read_replay_cases",
    "replay_tools",
]
