"""把机械臂恢复到当前 episode 开始时记录的关节位姿。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback
from ai import ImageContent, TextContent

from ..types import Observation
from .common import result_with_observation


class ReturnToHomeTool:
    """把无参数全局恢复请求交给机器人端口的关节控制器。"""

    name = "return_to_home"
    label = "Return to Home"
    description = (
        "Attempts to return the arm to the seven joint positions recorded at the start of "
        "the current episode while preserving the environment and current gripper command. "
        "It accepts no pose parameters and reports the joint-space target, final joint error, "
        "whether home was reached, and before/after observations. Reaching home is an "
        "execution result, not proof that the task or current plan step succeeded. When "
        "the workspace safety lock is active, a failed home recovery ends the episode "
        "with safety_stop."
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
        return_home: Callable[
            [asyncio.Event | None],
            Awaitable[
                tuple[
                    dict[str, object],
                    Observation | None,
                    Observation | None,
                ]
            ],
        ],
    ) -> None:
        self._return_home = return_home

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, params, on_update
        payload, before, after = await self._return_home(cancel_event)
        before_content: list[TextContent | ImageContent] = []
        if before is not None and before.llm_content():
            before_content.append(
                TextContent(
                    text=(
                        "RETURN-TO-HOME START observation. Images follow in this order: "
                        f"{', '.join(_observation_views(before))}."
                    )
                )
            )
            before_content.extend(before.llm_content())
        before_content.append(
            TextContent(
                text=(
                    "RETURN-TO-HOME END/latest observation. Images follow in this order: "
                    f"{', '.join(_observation_views(after)) if after is not None else 'none'}."
                )
            )
        )
        return result_with_observation(
            payload,
            after,
            before_observation=before_content,
        )


def _observation_views(observation: Observation) -> list[str]:
    return [
        name
        for name, frame in (
            ("external", observation.rgb),
            ("wrist", observation.wrist_rgb),
            ("depth", observation.depth),
        )
        if frame is not None
    ]


__all__ = ["ReturnToHomeTool"]
