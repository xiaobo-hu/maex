"""维护 episode 级确定性工具事件，并裁剪发送给 Planner 的旧证据。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from agent_core import AgentMessage
from ai import ImageContent, TextContent, ToolResultMessage, UserMessage

_ARGUMENT_FIELDS = (
    "step_id",
    "decision_summary",
    "completed_step_id",
    "next_step_id",
    "running_step_id",
    "reopen_step_id",
    "action",
    "magnitude",
    "success",
)
_RESULT_FIELDS = (
    "status",
    "step_id",
    "stop_reason",
    "chunks_used",
    "action",
    "requested_magnitude",
    "requested_distance_m",
    "target_pose",
    "unit",
    "applied_translation_m",
    "applied_rotation_deg",
    "command",
    "executed_steps",
    "reason",
    "consecutive_vla_attempts",
    "recovery_notice",
    "home_pose_reached",
)
_REJECTION_RESULT_FIELDS = (
    "task_plan",
    "active_step",
    "plan_progress",
    "valid_next_step_ids",
    "valid_reopen_step_ids",
    "retry_instruction",
)


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    """一次已完成工具调用的确定性 episode 记录。"""

    call_id: str
    tool: str
    fields: dict[str, object]

    def summary(self) -> dict[str, object]:
        """返回不含图像和完整工作记忆的可序列化记录。"""
        return {"call_id": self.call_id, "tool": self.tool, **self.fields}


@dataclass(slots=True)
class EpisodeMemory:
    """在工具完成时追加事件，并为 Planner 渲染紧凑的长期记忆。"""

    _events: list[EpisodeEvent] = field(default_factory=list)
    _call_ids: set[str] = field(default_factory=set)

    def clear(self) -> None:
        """开始新 episode 前清空事件账本。"""
        self._events.clear()
        self._call_ids.clear()

    def append_tool_event(
        self,
        *,
        call_id: str,
        tool: str,
        arguments: dict[str, object],
        details: object,
        is_error: bool,
    ) -> None:
        """从一次工具完成事件机械抽取稳定字段；相同调用只记录一次。"""
        if call_id in self._call_ids:
            return
        fields = {
            name: arguments[name]
            for name in _ARGUMENT_FIELDS
            if name in arguments
        }
        if isinstance(details, dict):
            fields.update(
                (name, details[name])
                for name in _RESULT_FIELDS
                if name in details
            )
            if details.get("status") == "rejected":
                fields.update(
                    (name, details[name])
                    for name in _REJECTION_RESULT_FIELDS
                    if name in details
                )
        if is_error:
            fields["is_error"] = True
        self._events.append(EpisodeEvent(call_id=call_id, tool=tool, fields=fields))
        self._call_ids.add(call_id)

    def summary(self) -> list[dict[str, object]]:
        """返回完整确定性事件账本，供 trace 审计。"""
        return [event.summary() for event in self._events]

    def render_for_model(self, call_ids: set[str]) -> str:
        """只渲染已归档调用；最新物理证据继续由原 ToolResult 提供。"""
        events = [
            event.summary() for event in self._events if event.call_id in call_ids
        ]
        return json.dumps({"episode_memory": events}, ensure_ascii=False, default=str)

    def compact_transcript_for_model(
        self,
        messages: list[AgentMessage[object]],
        cancel_event: asyncio.Event | None = None,
    ) -> list[AgentMessage[object]]:
        """保留最新物理证据，以账本替换更早结果并删除历史图像。"""
        del cancel_event
        tool_results = [
            (index, message)
            for index, message in enumerate(messages)
            if isinstance(message, ToolResultMessage)
        ]
        latest_tool_index = tool_results[-1][0] if tool_results else None
        latest_vla_index = next(
            (
                index
                for index, message in reversed(tool_results)
                if message.tool_name == "execute_vla"
            ),
            None,
        )
        latest_image_index = next(
            (
                index
                for index, message in reversed(tool_results)
                if any(isinstance(item, ImageContent) for item in message.content)
            ),
            None,
        )
        full_result_indices = {
            index
            for index in (latest_tool_index, latest_vla_index, latest_image_index)
            if index is not None
        }
        archived = [
            (index, message)
            for index, message in tool_results
            if index not in full_result_indices
        ]
        archived_ids = {message.tool_call_id for _, message in archived}

        transformed = list(messages)
        for archived_index, (index, message) in enumerate(archived):
            payload = (
                self.render_for_model(archived_ids)
                if archived_index == len(archived) - 1
                else json.dumps(
                    {"episode_memory_ref": message.tool_call_id},
                    ensure_ascii=False,
                )
            )
            transformed[index] = message.model_copy(
                update={"content": [TextContent(text=payload)]}
            )

        for index, message in enumerate(messages):
            if not isinstance(message, UserMessage) or latest_image_index is None:
                continue
            content = message.content
            if isinstance(content, list) and any(
                isinstance(item, ImageContent) for item in content
            ):
                transformed[index] = message.model_copy(
                    update={
                        "content": [
                            item
                            for item in content
                            if not isinstance(item, ImageContent)
                        ]
                    }
                )
        return transformed


__all__ = ["EpisodeEvent", "EpisodeMemory"]
