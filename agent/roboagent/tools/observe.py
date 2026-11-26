"""只读观察工具。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from ..types import Observation
from .common import result_with_observation


class ObserveTool:
    """获取最新观察，不推进机器人。"""

    name = "observe"
    label = "Observe"
    description = (
        "Returns the latest observation without changing the TaskPlan or robot state. "
        "Takes no arguments and does not advance the environment."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(self, observe: Callable[[asyncio.Event | None], Awaitable[Observation]]) -> None:
        self._observe = observe

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, params, on_update
        observation = await self._observe(cancel_event)
        return result_with_observation({"status": "observed"}, observation)
