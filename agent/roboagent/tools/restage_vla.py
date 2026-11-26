"""在 frozen VLA 重试前执行固定的腕部相机反向局部回撤。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback
from ai import ImageContent, TextContent

from ..types import Observation
from .common import result_with_observation


class RestageVlaTool:
    """把无参数局部回撤请求交给 Runtime 的闭环 OSC 执行器。"""

    name = "restage_vla"
    label = "Restage VLA"
    description = (
        "Retracts the end effector 5 cm along the current wrist-camera -Z axis while "
        "preserving the current gripper command. Returns the applied displacement, "
        "before/after poses, and labeled before/after observations. Execution success only "
        "means the retraction reached its target; task effect remains unverified."
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
        restage: Callable[
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
        self._restage = restage

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, params, on_update
        payload, before, after = await self._restage(cancel_event)
        before_content: list[TextContent | ImageContent] = []
        if before is not None and before.llm_content():
            before_content.append(
                TextContent(
                    text=(
                        "RESTAGE START observation. Images follow in this order: "
                        f"{', '.join(_observation_views(before))}."
                    )
                )
            )
            before_content.extend(before.llm_content())
        before_content.append(
            TextContent(
                text=(
                    "RESTAGE END/latest observation. Images follow in this order: "
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


__all__ = ["RestageVlaTool"]
