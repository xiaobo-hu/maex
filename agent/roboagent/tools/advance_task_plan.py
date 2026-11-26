"""应用上层 Agent 显式选择的 TaskPlan 原子推进。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from .common import result_with_observation


class AdvanceTaskPlanTool:
    """完成当前步骤并激活 Agent 明确指定的下一步骤。"""

    name = "advance_task_plan"
    label = "Advance Task Plan"
    description = (
        "Marks the current running step completed and activates the immediate next pending "
        "step, or completes a terminal plan. The Runtime requires the current step to have "
        "executed a VLA action. Rejected requests return the unchanged plan and valid "
        "correction candidates. Does not move the robot."
    )
    parameters = {
        "type": "object",
        "properties": {
            "completed_step_id": {
                "type": "string",
                "minLength": 1,
                "description": "The exact id of the single current running step.",
            },
            "next_step_id": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ],
                "description": (
                    "The explicitly selected pending step id, or null only if no pending "
                    "steps remain."
                ),
            },
            "decision_summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reason for choosing this tool now.",
            },
        },
        "required": ["completed_step_id", "next_step_id"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        advance: Callable[
            [str, str | None, asyncio.Event | None],
            Awaitable[dict[str, object]],
        ],
    ) -> None:
        self._advance = advance

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """把 Agent 指定的完成项和下一运行项提交给 runtime。"""
        del tool_call_id, on_update
        completed_step_id = str(params["completed_step_id"]).strip()
        next_value = params["next_step_id"]
        next_step_id = None if next_value is None else str(next_value).strip()
        payload = await self._advance(completed_step_id, next_step_id, cancel_event)
        return result_with_observation(payload)
