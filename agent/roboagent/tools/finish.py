"""任务终止工具。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from ..types import Observation
from .common import result_with_observation


class FinishTool:
    """提交模型对任务终态的判断，并让 Agent Loop 停止。"""

    name = "finish"
    label = "Finish"
    description = (
        "Submits the requested episode outcome and terminates the Agent Loop. success=true "
        "may invoke an environment-specific terminal verifier. Returns the validated status "
        "and latest observation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "Requested episode outcome; true may be externally verified.",
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise final summary.",
            },
        },
        "required": ["success", "summary"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        finish: Callable[[bool, str, asyncio.Event | None], Awaitable[tuple[dict[str, object], Observation | None]]],
    ) -> None:
        self._finish = finish

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        payload, observation = await self._finish(
            bool(params["success"]), str(params["summary"]).strip(), cancel_event
        )
        result = result_with_observation(payload, observation)
        result.terminate = True
        return result
