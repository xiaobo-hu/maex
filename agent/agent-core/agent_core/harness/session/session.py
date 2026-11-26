"""Session 条目操作和公开上下文恢复。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ai import AssistantMessage, Message, ToolResultMessage, UserMessage

from ..messages import CompactionSummaryMessage
from .types import SessionData, SessionEntry, SessionStorage


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Session:
    """对内存或 JSONL Storage 提供统一的 append/replay 接口。"""

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage

    @property
    def storage(self) -> object:
        return self._storage

    def get_entries(self) -> list[SessionEntry]:
        return self._storage.get_entries()

    def get_leaf_id(self) -> str | None:
        return self._storage.get_leaf_id()

    def _append(
        self, entry_type: str, *, message: Message | None = None, data: SessionData = None
    ) -> SessionEntry:
        entry = SessionEntry(
            id=uuid4().hex[:16],
            parent_id=self.get_leaf_id(),
            timestamp=_timestamp(),
            type=entry_type,  # type: ignore[arg-type]
            message=message,
            data=data,
        )
        self._storage.append_entry(entry)
        return entry

    def append_message(self, message: Message) -> SessionEntry:
        return self._append("message", message=message)

    def append_compaction(
        self,
        summary: str,
        retained_tail: list[Message],
        tokens_before: int = 0,
        compacted_message_count: int = 0,
        first_kept_entry_id: str | None = None,
    ) -> SessionEntry:
        return self._append(
            "compaction",
            data={
                "summary": summary,
                "retainedTail": [
                    message.model_dump(by_alias=True, mode="json") for message in retained_tail
                ],
                "tokensBefore": tokens_before,
                "compactedMessageCount": compacted_message_count,
                "firstKeptEntryId": first_kept_entry_id,
            },
        )

    def append_customer_state(self, state: dict[str, object]) -> SessionEntry:
        return self._append("customer_state", data=dict(state))

    def clear(self) -> None:
        self._storage.clear()

    def build_context(self) -> list[Message | CompactionSummaryMessage]:
        """恢复逻辑消息；Provider 格式转换由 Agent 的转换器负责。"""

        entries = self.get_entries()
        compaction_index = max(
            (index for index, entry in enumerate(entries) if entry.type == "compaction"),
            default=-1,
        )
        messages: list[Message | CompactionSummaryMessage] = []
        if compaction_index >= 0:
            compaction = entries[compaction_index].data
            if isinstance(compaction, dict):
                summary = compaction.get("summary")
                if isinstance(summary, str) and summary:
                    tokens_raw = compaction.get("tokensBefore")
                    tokens_before = (
                        tokens_raw
                        if isinstance(tokens_raw, int)
                        else int(tokens_raw)
                        if isinstance(tokens_raw, str)
                        else 0
                    )
                    compacted_raw = compaction.get("compactedMessageCount")
                    compacted_count = (
                        compacted_raw
                        if isinstance(compacted_raw, int)
                        else int(compacted_raw)
                        if isinstance(compacted_raw, str)
                        else 0
                    )
                    messages.append(
                        CompactionSummaryMessage(
                            summary=summary,
                            first_kept_entry_id=(
                                str(compaction.get("firstKeptEntryId"))
                                if compaction.get("firstKeptEntryId") is not None
                                else None
                            ),
                            tokens_before=tokens_before,
                            compacted_message_count=compacted_count,
                        )
                    )
                first_kept = compaction.get("firstKeptEntryId")
                if isinstance(first_kept, str):
                    first_index = next(
                        (index for index, entry in enumerate(entries) if entry.id == first_kept),
                        -1,
                    )
                    if first_index >= 0:
                        for entry in entries[first_index:compaction_index]:
                            if entry.type == "message" and entry.message is not None:
                                messages.append(entry.message)
                    else:
                        first_kept = None
                if not isinstance(first_kept, str):
                    retained = compaction.get("retainedTail")
                    if isinstance(retained, list):
                        messages.extend(_decode_messages(retained))
            entries = entries[compaction_index + 1 :]
        for entry in entries:
            if entry.type == "message" and entry.message is not None:
                messages.append(entry.message)
        return messages

    def append_customer_turn(
        self,
        sales: Message,
        customer: Message,
        state: dict[str, object],
    ) -> list[SessionEntry]:
        """一次性追加公开 Sales、Customer 消息和运行态。"""

        entries: list[SessionEntry] = []
        parent = self.get_leaf_id()
        for entry_type, message, data in (
            ("message", sales, None),
            ("message", customer, None),
            ("customer_state", None, dict(state)),
        ):
            entry = SessionEntry(
                id=uuid4().hex[:16],
                parent_id=parent,
                timestamp=_timestamp(),
                type=entry_type,  # type: ignore[arg-type]
                message=message,
                data=data,
            )
            entries.append(entry)
            parent = entry.id
        self._storage.append_entries(entries)
        return entries

    def public_transcript(self) -> list[Message]:
        return [
            entry.message
            for entry in self.get_entries()
            if entry.type == "message" and entry.message
        ]

    def public_message_entry_ids(self) -> list[str]:
        """返回公开消息按 transcript 顺序对应的 Session entry ID。"""

        return [
            entry.id
            for entry in self.get_entries()
            if entry.type == "message" and entry.message is not None
        ]

    def latest_customer_state(self) -> dict[str, object] | None:
        for entry in reversed(self.get_entries()):
            if entry.type == "customer_state" and isinstance(entry.data, dict):
                return dict(entry.data)
        return None


def _decode_messages(values: list[object]) -> list[Message]:
    result: list[Message] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        role = value.get("role")
        if role == "user":
            result.append(UserMessage.model_validate(value))
        elif role == "assistant":
            result.append(AssistantMessage.model_validate(value))
        elif role == "toolResult":
            result.append(ToolResultMessage.model_validate(value))
    return result


__all__ = ["Session"]
