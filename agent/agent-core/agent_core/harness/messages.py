"""Harness 扩展消息；摘要在 LLM 边界转换为普通用户消息。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai import AssistantMessage, Message, ToolResultMessage, UserMessage


@dataclass(frozen=True, slots=True)
class CompactionSummaryMessage:
    summary: str
    first_kept_entry_id: str | None = None
    tokens_before: int = 0
    compacted_message_count: int = 0
    role: Literal["compactionSummary"] = "compactionSummary"


def convert_to_llm(messages: list[Message | CompactionSummaryMessage]) -> list[Message]:
    """将 Harness 摘要转换为 Provider 可识别的 UserMessage。"""

    result: list[Message] = []
    for message in messages:
        if isinstance(message, CompactionSummaryMessage):
            result.append(
                UserMessage(
                    content=("<compaction_summary>\n" + message.summary + "\n</compaction_summary>")
                )
            )
        elif isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
            result.append(message)
    return result


__all__ = ["CompactionSummaryMessage", "convert_to_llm"]
