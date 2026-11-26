"""Context-window overflow detection matching Pi's three signal classes."""

from __future__ import annotations

import re

from ..types import AssistantMessage, Model

_OVERFLOW_RE = re.compile(
    r"context length|context window|maximum context|too many tokens|token limit|"
    r"prompt is too long|input is too long|reduce the length|exceeds.*context|"
    r"context_length_exceeded|max_tokens.*less than|request too large",
    re.IGNORECASE,
)
_THROTTLE_RE = re.compile(r"rate.?limit|too many requests|tokens per minute|\btpm\b", re.I)


def is_context_overflow(message: AssistantMessage, model: Model) -> bool:
    text = message.error_message or ""
    if (
        message.stop_reason == "error"
        and _OVERFLOW_RE.search(text)
        and not _THROTTLE_RE.search(text)
    ):
        return True
    used = message.usage.input + message.usage.cache_read + message.usage.cache_write
    if model.context_window <= 0:
        return False
    if used > model.context_window:
        return True
    return (
        message.stop_reason == "length"
        and message.usage.output == 0
        and used >= int(model.context_window * 0.99)
    )


__all__ = ["is_context_overflow"]
