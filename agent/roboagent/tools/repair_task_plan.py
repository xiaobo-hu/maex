"""应用上层 Agent 显式选择的 TaskPlan 原子回退。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from .common import result_with_observation


class RepairTaskPlanTool:
    """暂停当前步骤并重开一个已完成的前置步骤。"""

    name = "repair_task_plan"
    label = "Repair Task Plan"
    description = (
        "Reopens one earlier completed step as running when new physical evidence shows "
        "that prerequisite was lost, and resets every later step to pending. "
        "The Runtime requires the current step to have executed a VLA action. Does not "
        "move the robot or alter step instructions or order. Rejected requests return the "
        "unchanged plan and valid correction candidates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "running_step_id": {
                "type": "string",
                "minLength": 1,
                "description": "The exact id of the single current running step.",
            },
            "reopen_step_id": {
                "type": "string",
                "minLength": 1,
                "description": "The exact id of an earlier completed prerequisite step.",
            },
            "decision_summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise physical evidence that the prerequisite was lost.",
            },
        },
        "required": ["running_step_id", "reopen_step_id"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        repair: Callable[
            [str, str, asyncio.Event | None],
            Awaitable[dict[str, object]],
        ],
    ) -> None:
        self._repair = repair

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """把 Agent 指定的当前步骤和待重开前置步骤提交给 runtime。"""
        del tool_call_id, on_update
        running_step_id = str(params["running_step_id"]).strip()
        reopen_step_id = str(params["reopen_step_id"]).strip()
        payload = await self._repair(
            running_step_id,
            reopen_step_id,
            cancel_event,
        )
        return result_with_observation(payload)
