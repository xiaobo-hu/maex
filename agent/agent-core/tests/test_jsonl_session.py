from __future__ import annotations

import json

import pytest

from agent_core.harness import (
    CompactionSummaryMessage,
    JsonlSessionStorage,
    Session,
    SessionError,
    convert_to_llm,
)
from ai import AssistantMessage, TextContent, UserMessage


def test_jsonl_round_trip_messages_and_parent_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "customer.jsonl"
    storage = JsonlSessionStorage(path, session_id="customer-1", metadata={"scenarioId": "trial"})
    session = Session(storage)
    session.append_message(UserMessage(content="你好"))
    session.append_message(AssistantMessage(content=[TextContent(text="您好")], stop_reason="stop"))

    reopened = Session(JsonlSessionStorage(path))
    entries = reopened.get_entries()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["version"] == 3
    assert entries[0].parent_id is None
    assert entries[1].parent_id == entries[0].id
    assert [message.role for message in reopened.public_transcript()] == ["user", "assistant"]


def test_jsonl_restores_compaction_and_customer_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "customer.jsonl"
    session = Session(JsonlSessionStorage(path))
    first = UserMessage(content="早期销售消息")
    tail = [AssistantMessage(content=[TextContent(text="近期客户回复")], stop_reason="stop")]
    session.append_message(first)
    session.append_compaction("客户关注价格", tail, tokens_before=100)
    session.append_customer_state({"turn_index": 1, "status": "active"})

    reopened = Session(JsonlSessionStorage(path))
    context = reopened.build_context()
    assert isinstance(context[0], CompactionSummaryMessage)
    assert context[0].summary == "客户关注价格"
    assert context[1].content[0].text == "近期客户回复"
    assert reopened.latest_customer_state() == {"turn_index": 1, "status": "active"}
    assert [message.content for message in reopened.public_transcript()] == [
        "早期销售消息",
    ]


def test_jsonl_rejects_bad_header_and_parent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"type": "session", "version": 2}) + "\n", encoding="utf-8")
    with pytest.raises(SessionError, match="version 3"):
        JsonlSessionStorage(path)
    valid = tmp_path / "valid.jsonl"
    storage = JsonlSessionStorage(valid)
    Session(storage).append_message(UserMessage(content="x"))
    with valid.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "message",
                    "id": "orphan",
                    "parentId": "missing",
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "message": {"role": "user", "content": "bad"},
                }
            )
            + "\n"
        )
    with pytest.raises(SessionError, match="parentId"):
        JsonlSessionStorage(valid)


def test_compaction_summary_is_converted_only_at_llm_boundary() -> None:
    logical = CompactionSummaryMessage(summary="客户关注价格")
    converted = convert_to_llm([logical])
    assert converted[0].role == "user"
    assert converted[0].content == "<compaction_summary>\n客户关注价格\n</compaction_summary>"
