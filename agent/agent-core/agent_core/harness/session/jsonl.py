"""Pi v3 风格 JSONL Session Storage。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from ai import AssistantMessage, Message, ToolResultMessage, UserMessage

from ..messages import CompactionSummaryMessage
from .types import SessionData, SessionEntry, SessionError, SessionHeader


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _message_to_json(message: Message) -> dict[str, object]:
    return message.model_dump(by_alias=True, mode="json")


def _message_from_json(value: object) -> Message:
    if not isinstance(value, dict):
        raise SessionError("Session message must be an object")
    role = value.get("role")
    try:
        if role == "user":
            return UserMessage.model_validate(value)
        if role == "assistant":
            return AssistantMessage.model_validate(value)
        if role == "toolResult":
            return ToolResultMessage.model_validate(value)
    except Exception as error:
        raise SessionError(f"Invalid {role!r} session message: {error}") from error
    raise SessionError(f"Unsupported session message role: {role!r}")


def _data_to_json(data: SessionData) -> object:
    if isinstance(data, CompactionSummaryMessage):
        return {
            "summary": data.summary,
            "tokensBefore": data.tokens_before,
            "compactedMessageCount": data.compacted_message_count,
        }
    if hasattr(data, "model_dump"):
        return data.model_dump(by_alias=True, mode="json")  # type: ignore[union-attr]
    return data


class JsonlSessionStorage:
    """按追加方式保存一个 Customer Agent Session。"""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str | None = None,
        cwd: str = "customer-agent",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.path = Path(path)
        self._lock = Lock()
        if self.path.exists():
            self.header, self._entries = self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.header = SessionHeader(
                id=session_id or uuid4().hex,
                timestamp=_timestamp(),
                cwd=cwd,
                metadata=dict(metadata or {}),
            )
            self._entries = []
            self.path.write_text(
                json.dumps(asdict(self.header), ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

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
            for entry in entries:
                if entry.id in seen:
                    raise SessionError(f"Duplicate session entry id: {entry.id}")
                if entry.parent_id != leaf:
                    raise SessionError(f"Invalid parentId for entry {entry.id}: expected {leaf!r}")
                seen.add(entry.id)
                leaf = entry.id
            encoded = [_entry_to_json(entry) for entry in entries]
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                for value in encoded:
                    handle.write(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                handle.flush()
            self._entries.extend(entries)

    def append_entry(self, entry: SessionEntry) -> None:
        self.append_entries([entry])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.path.write_text(
                json.dumps(asdict(self.header), ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

    def _load(self) -> tuple[SessionHeader, list[SessionEntry]]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise SessionError(f"Invalid JSONL session {self.path}: missing header")
        try:
            raw_header = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise SessionError(f"Invalid JSONL session header: {error}") from error
        if not isinstance(raw_header, dict):
            raise SessionError("Session header must be an object")
        if raw_header.get("type") != "session" or raw_header.get("version") != 3:
            raise SessionError("Unsupported JSONL session header; expected version 3")
        try:
            header = SessionHeader(
                id=str(raw_header["id"]),
                timestamp=str(raw_header["timestamp"]),
                cwd=str(raw_header["cwd"]),
                metadata=dict(raw_header.get("metadata") or {}),
            )
        except Exception as error:
            raise SessionError(f"Invalid JSONL session header: {error}") from error
        entries: list[SessionEntry] = []
        ids: set[str] = set()
        for line_number, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                entry = _entry_from_json(value)
            except (json.JSONDecodeError, SessionError) as error:
                raise SessionError(f"Invalid JSONL entry at line {line_number}: {error}") from error
            if entry.id in ids:
                raise SessionError(f"Duplicate session entry id at line {line_number}: {entry.id}")
            expected_parent = entries[-1].id if entries else None
            if entry.parent_id != expected_parent:
                raise SessionError(
                    f"Invalid parentId at line {line_number}: expected {expected_parent!r}"
                )
            ids.add(entry.id)
            entries.append(entry)
        return header, entries


def _entry_to_json(entry: SessionEntry) -> dict[str, object]:
    value: dict[str, object] = {
        "type": entry.type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": entry.timestamp,
    }
    if entry.message is not None:
        value["message"] = _message_to_json(entry.message)
    elif entry.data is not None:
        value["data"] = _data_to_json(entry.data)
    return value


def _entry_from_json(value: object) -> SessionEntry:
    if not isinstance(value, dict):
        raise SessionError("Session entry must be an object")
    entry_type = value.get("type")
    if entry_type not in {"message", "compaction", "customer_state", "session_info"}:
        raise SessionError(f"Unsupported session entry type: {entry_type!r}")
    entry_id = value.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise SessionError("Session entry id is required")
    parent_id = value.get("parentId")
    if parent_id is not None and not isinstance(parent_id, str):
        raise SessionError("Session entry parentId must be a string or null")
    timestamp = value.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise SessionError("Session entry timestamp is required")
    if entry_type == "message":
        return SessionEntry(
            id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            type="message",
            message=_message_from_json(value.get("message")),
        )
    return SessionEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp=timestamp,
        type=entry_type,
        data=value.get("data"),
    )


__all__ = ["JsonlSessionStorage"]
