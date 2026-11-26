"""基线验证层：离线核对 VoLo 请求取证，不调用 API 或 GPU。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_audit_preserves_request_response_and_excludes_client_credentials(tmp_path):
    api = pytest.importorskip("vlm_orchestrator.vlm.api")
    calls = []
    response_data = {"model": "returned-model", "choices": [{"finish_reason": "stop"}],
                     "usage": {"completion_tokens": 12}}
    response = SimpleNamespace(model_dump=lambda **kw: response_data)

    def create(**kwargs):
        calls.append(kwargs)
        assert len(list((tmp_path / "vlm_requests").glob("*.json"))) == 1
        return response

    client = SimpleNamespace(api_key="test-only-client-secret", chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": [{"type": "text", "text": "current task"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}}]}]
    assert api.chat_create(client, model="deepseek-test", messages=messages,
                           max_tokens=512, audit_dir=str(tmp_path)) is response
    text = next((tmp_path / "vlm_requests").glob("*.json")).read_text()
    record = json.loads(text)
    assert record["request"] == calls[0]
    assert record["request"]["extra_body"]["thinking"]["type"] == "disabled"
    assert record["response"] == response_data
    assert "audit_dir" not in calls[0] and "test-only-client-secret" not in text


def test_audit_records_api_error_without_retry(tmp_path):
    api = pytest.importorskip("vlm_orchestrator.vlm.api")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("offline fixture")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ConnectionError):
        api.chat_create(client, model="other-model", messages=[], audit_dir=str(tmp_path))
    record = json.loads(next((tmp_path / "vlm_requests").glob("*.json")).read_text())
    assert record["error_type"] == "ConnectionError" and len(calls) == 1


def test_audit_disk_failure_does_not_repeat_model_call(monkeypatch, tmp_path):
    api = pytest.importorskip("vlm_orchestrator.vlm.api")
    calls = []
    response = SimpleNamespace(model_dump=lambda **kw: {})

    def fail_write(*args, **kwargs):
        raise OSError("disk full fixture")

    def create(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(Path, "write_text", fail_write)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert api.chat_create(client, model="other-model", messages=[], audit_dir=str(tmp_path)) is response
    assert len(calls) == 1
