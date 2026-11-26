"""Pi 风格 Session 的严格公共数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ai import Message

from ..messages import CompactionSummaryMessage

SessionEntryType = Literal["message", "compaction", "customer_state", "session_info"]
SessionData = Message | CompactionSummaryMessage | dict[str, object] | str | None


@dataclass(frozen=True, slots=True)
class SessionHeader:
    id: str
    timestamp: str
    cwd: str
    metadata: dict[str, object] = field(default_factory=dict)
    type: Literal["session"] = "session"
    version: Literal[3] = 3


@dataclass(frozen=True, slots=True)
class SessionEntry:
    id: str
    parent_id: str | None
    timestamp: str
    type: SessionEntryType
    message: Message | None = None
    data: SessionData = None


class SessionError(RuntimeError):
    """Session 文件、条目或存储操作错误。"""


class SessionStorage(Protocol):
    def get_entries(self) -> list[SessionEntry]: ...

    def get_leaf_id(self) -> str | None: ...

    def append_entry(self, entry: SessionEntry) -> None: ...

    def append_entries(self, entries: list[SessionEntry]) -> None: ...

    def clear(self) -> None: ...


__all__ = [
    "SessionData",
    "SessionEntry",
    "SessionEntryType",
    "SessionError",
    "SessionHeader",
    "SessionStorage",
]
