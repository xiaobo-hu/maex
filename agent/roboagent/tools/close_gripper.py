"""RoboAgent Pi 层的失败恢复工具：保持末端位姿并闭合夹爪。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from ..types import Observation
from .common import result_with_observation


class CloseGripperTool:
    """把无参数的夹爪闭合请求交给 Runtime 执行。"""

    name = "close_gripper"
    label = "Close Gripper"
    description = (
        "Commands zero end-effector motion while closing the gripper, only after the "
        "intended target or handle appears aligned between the fingers. A successful "
        "execution only confirms that the close command ran; it does not prove alignment, "
        "contact, grasp, handle engagement, or target identity. Inspect the returned "
        "observation before pulling, advancing the plan, or claiming a task effect."
    )
    parameters = {
        "type": "object",
        "properties": {
            "decision_summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reason for choosing this tool now.",
            }
        },
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        close_gripper: Callable[
            [asyncio.Event | None],
            Awaitable[tuple[dict[str, object], Observation | None]],
        ],
    ) -> None:
        self._close_gripper = close_gripper

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """执行一次保持末端位姿的明确闭合命令。"""
        del tool_call_id, params, on_update
        payload, observation = await self._close_gripper(cancel_event)
        return result_with_observation(payload, observation)
