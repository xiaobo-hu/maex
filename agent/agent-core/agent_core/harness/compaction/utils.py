"""压缩摘要的稳定文本格式。"""

from __future__ import annotations

from ai import AssistantMessage, Message, TextContent, ToolResultMessage, UserMessage


def messages_to_transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.role
        if isinstance(message, UserMessage):
            text = (
                message.content
                if isinstance(message.content, str)
                else " ".join(getattr(item, "text", "") for item in message.content)
            )
        elif isinstance(message, (AssistantMessage, ToolResultMessage)):
            text = " ".join(item.text for item in message.content if isinstance(item, TextContent))
        else:
            text = ""
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


__all__ = ["messages_to_transcript"]
