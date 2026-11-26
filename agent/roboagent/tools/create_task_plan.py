"""接收上层 Agent 显式生成的 episode TaskPlan。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from .common import result_with_observation


class CreateTaskPlanTool:
    """让 Agent 在首次机器人动作前提交结构化子任务计划。"""

    name = "create_task_plan"
    label = "Create Task Plan"
    description = (
        "Creates the episode TaskPlan from the full ordered step list. Requires unique ids, "
        "exactly one running step, and all other unfinished steps pending. The Runtime "
        "accepts one plan creation per episode."
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "description": "Full ordered plan with exactly one running step.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Unique stable step id.",
                        },
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Language instruction passed to the VLA for this step.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "completed"],
                            "description": "Initial state of this plan step.",
                        },
                    },
                    "required": ["id", "instruction", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        create: Callable[
            [list[dict[str, object]], asyncio.Event | None],
            Awaitable[dict[str, object]],
        ],
    ) -> None:
        self._create = create

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """校验工具参数后把计划提交给 RoboAgent runtime。"""
        del tool_call_id, on_update
        steps = cast(list[dict[str, object]], params["steps"])
        payload = await self._create(steps, cancel_event)
        return result_with_observation(payload)
