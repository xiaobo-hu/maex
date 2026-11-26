"""把一次语言子任务交给 frozen VLA 并执行动作块。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback
from ai import ImageContent, TextContent

from ..memory import VlaWorkingMemory
from ..types import Observation


class ExecuteVlaTool:
    """执行受 runtime chunk 预算约束的 VLA 技能。"""

    name = "execute_vla"
    label = "Execute VLA"
    description = (
        "Runs VLA action chunks for the instruction associated with step_id. Returns "
        "execution status, the latest observation, and the multimodal working memory "
        "collected during this call. Execution progress alone does not prove the current "
        "plan step succeeded; only task_effect_status=verified_task_success represents "
        "official full-task success."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {
                "type": "string",
                "minLength": 1,
                "description": "The id of the single running TaskPlan step.",
            },
            "decision_summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reason for choosing this tool now.",
            },
        },
        "required": ["step_id"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        execute: Callable[
            [str, asyncio.Event | None],
            Awaitable[tuple[dict[str, object], Observation | None, VlaWorkingMemory]],
        ],
    ) -> None:
        self._execute = execute

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        step_id = str(params["step_id"]).strip()
        payload, observation, working_memory = await self._execute(step_id, cancel_event)
        details = (
            {**payload, "observation": observation.summary()}
            if observation is not None
            else payload
        )
        result = {
            key: value
            for key, value in payload.items()
            if key not in {"active_step", "plan_progress", "working_memory"}
        }
        visible_header = {
            "current_step": payload.get("active_step"),
            "execution_diagnostics": working_memory.execution.summary(),
            "result": result,
            "plan_progress": payload.get("plan_progress"),
            "latest_observation": (
                observation.summary() if observation is not None else None
            ),
        }
        content: list[TextContent | ImageContent] = [
            TextContent(
                text=json.dumps(visible_header, ensure_ascii=False, default=str)
            ),
            *working_memory.llm_content(),
        ]
        if observation is not None and observation.llm_content():
            views = [
                name
                for name, frame in (
                    ("external", observation.rgb),
                    ("wrist", observation.wrist_rgb),
                    ("depth", observation.depth),
                )
                if frame is not None
            ]
            sequence = (
                working_memory.execution.end.sequence
                if working_memory.execution.end is not None
                else None
            )
            content.append(
                TextContent(
                    text=(
                        f"VLA call end/latest observation at sequence {sequence}. "
                        f"Images follow in this order: {', '.join(views)}."
                    )
                )
            )
            content.extend(observation.llm_content())

        advisory: list[str] = []
        seen_advisory: set[str] = set()
        for keyframe in working_memory.visual.keyframes:
            text = keyframe.observation
            if text and text not in seen_advisory:
                advisory.append(f"Keyframe sequence {keyframe.sequence}: {text}")
                seen_advisory.add(text)
        for text in working_memory.visual.observations:
            if text not in seen_advisory:
                advisory.append(text)
                seen_advisory.add(text)
        if advisory:
            content.append(
                TextContent(
                    text=(
                        "Observer advisory — may contain visual recognition errors.\n"
                        "Use only as a temporal hint after examining the direct images.\n\n"
                        + "\n".join(f"- {item}" for item in advisory)
                    )
                )
            )
        return AgentToolResult(content=content, details=details)
