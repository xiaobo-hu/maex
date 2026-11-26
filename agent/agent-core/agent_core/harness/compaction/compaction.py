"""公开消息历史的 token 估算、保留尾部和摘要切分。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ai import AssistantMessage, Message, TextContent, ToolResultMessage, UserMessage


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


def _message_text(message: Message) -> str:
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return message.content
        return " ".join(getattr(item, "text", "") for item in message.content)
    if isinstance(message, AssistantMessage):
        return " ".join(
            item.text if isinstance(item, TextContent) else json.dumps(item.model_dump(mode="json"))
            for item in message.content
        )
    if isinstance(message, ToolResultMessage):
        return " ".join(getattr(item, "text", "") for item in message.content)
    return ""


def estimate_tokens(messages: list[Message]) -> int:
    """使用 Pi 的约四字符一 token 估算，避免依赖特定 tokenizer。"""

    return max(0, (sum(len(_message_text(message)) for message in messages) + 3) // 4)


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings | None = None,
) -> bool:
    settings = settings or CompactionSettings()
    return settings.enabled and context_tokens > max(0, context_window - settings.reserve_tokens)


def compact_messages(
    messages: list[Message],
    *,
    keep_recent_tokens: int = 20_000,
) -> tuple[list[Message], list[Message], int]:
    """按消息边界切分为需要摘要的前缀和保留的近期尾部。"""

    total = estimate_tokens(messages)
    if total <= keep_recent_tokens:
        return [], list(messages), total
    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        cost = max(1, estimate_tokens([message]))
        if kept and used + cost > keep_recent_tokens:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    cut = len(messages) - len(kept)
    roles = [message.role for message in messages]
    if (
        cut % 2 == 1
        and len(messages) % 2 == 0
        and roles[::2] == ["user"] * (len(messages) // 2)
        and roles[1::2] == ["assistant"] * (len(messages) // 2)
    ):
        cut += 1
        kept = messages[cut:]
    return messages[:cut], kept, total


__all__ = ["CompactionSettings", "compact_messages", "estimate_tokens", "should_compact"]
