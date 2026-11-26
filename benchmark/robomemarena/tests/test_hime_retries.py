"""验证 HiMe 连接重试不重放已完成响应，并纳入实际请求预算。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from benchmark.robomemarena.baselines.hime_clients import HiMeChatClient, ModelBudgetExceededError
from openai import APIConnectionError


def make_client(tmp_path, create, max_calls=80):
    return HiMeChatClient(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="test", role="planner", log_path=tmp_path / "calls.jsonl",
        max_tokens=8192, max_calls=max_calls,
    )


def connection_error():
    return APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))


def test_retry_identical_request_then_return_one_complete_response(tmp_path, monkeypatch):
    attempts, delays = [], []
    monkeypatch.setattr("benchmark.robomemarena.baselines.hime_clients.time.sleep", delays.append)

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise connection_error()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="complete XML"), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 42}),
        )

    client = make_client(tmp_path, create)
    messages = [{"role": "user", "content": "current episode evidence"}]
    assert client.chat(messages) == "complete XML"
    assert len(attempts) == client.calls == 3
    assert attempts[0] == attempts[1] == attempts[2]
    assert delays == [2, 4]
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["attempt"] for r in records] == [1, 2, 3]
    assert [r.get("error_type") for r in records] == ["APIConnectionError", "APIConnectionError", None]
    assert sum("response" in r for r in records) == 1


@pytest.mark.parametrize("budget,expected_calls,error", [
    (80, 3, APIConnectionError), (2, 2, ModelBudgetExceededError),
])
def test_persistent_connection_failure_is_bounded(tmp_path, monkeypatch, budget, expected_calls, error):
    monkeypatch.setattr("benchmark.robomemarena.baselines.hime_clients.time.sleep", lambda _: None)

    def create(**kwargs):
        raise connection_error()

    client = make_client(tmp_path, create, max_calls=budget)
    with pytest.raises(error):
        client.chat([{"role": "user", "content": "evidence"}])
    assert client.calls == expected_calls


def test_non_connection_failure_is_not_retried(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr("benchmark.robomemarena.baselines.hime_clients.time.sleep", delays.append)

    def create(**kwargs):
        raise ValueError("invalid request")

    client = make_client(tmp_path, create)
    with pytest.raises(ValueError, match="invalid request"):
        client.chat([{"role": "user", "content": "evidence"}])
    assert client.calls == 1
    assert delays == []
