"""用于测试和临时运行的内存 Session Storage。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from .types import SessionEntry, SessionError, SessionHeader


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class InMemorySessionStorage:
    """与 JSONL Storage 共享同一条目契约的内存实现。"""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        cwd: str = "customer-agent",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.header = SessionHeader(
            id=session_id or uuid4().hex,
            timestamp=_timestamp(),
            cwd=cwd,
            metadata=dict(metadata or {}),
        )
        self._entries: list[SessionEntry] = []
        self._lock = Lock()

    def get_entries(self) -> list[SessionEntry]:
        with self._lock:
            return list(self._entries)

    def get_leaf_id(self) -> str | None:
        with self._lock:
            return self._entries[-1].id if self._entries else None

    def append_entries(self, entries: list[SessionEntry]) -> None:
        with self._lock:
            leaf = self._entries[-1].id if self._entries else None
            seen = {item.id for item in self._entries}
            prepared: list[SessionEntry] = []
            for entry in entries:
                if entry.id in seen:
                    raise SessionError(f"Duplicate session entry id: {entry.id}")
                if entry.parent_id != leaf:
                    raise SessionError(f"Invalid parentId for entry {entry.id}: expected {leaf!r}")
                prepared.append(replace(entry))
                seen.add(entry.id)
                leaf = entry.id
            self._entries.extend(prepared)

    def append_entry(self, entry: SessionEntry) -> None:
        self.append_entries([entry])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__ = ["InMemorySessionStorage"]
