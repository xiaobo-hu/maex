"""RoboAgent Pi 层的失败恢复工具：保持末端位姿并打开夹爪。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from ..types import Observation
from .common import result_with_observation


class OpenGripperTool:
    """把无参数的夹爪打开请求交给 Runtime 执行。"""

    name = "open_gripper"
    label = "Open Gripper"
    description = (
        "Commands zero end-effector motion while opening the gripper for recovery, with "
        "a workspace check after every control step. A successful execution only confirms "
        "that the open command ran; it does not prove that an object was safely released "
        "or that nearby objects were undisturbed. Inspect the returned observation before "
        "claiming a task effect."
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
        open_gripper: Callable[
            [asyncio.Event | None],
            Awaitable[tuple[dict[str, object], Observation | None]],
        ],
    ) -> None:
        self._open_gripper = open_gripper

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, params, on_update
        payload, observation = await self._open_gripper(cancel_event)
        return result_with_observation(payload, observation)
