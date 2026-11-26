"""验证 HiMe 基线的原调度、模型参数和 benchmark 边界，不伪造真实任务成绩。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmark.robomemarena.baselines.hime_bridge import HiMeSession
from benchmark.robomemarena.baselines.hime_clients import (
    HiMeChatClient,
    ModelBudgetExceededError,
    RecordedEncoder,
)
from benchmark.robomemarena.baselines.hime_eval import (
    PROJECT_ROOT,
    PROMPT_DIR,
    execute_episode,
)


def observation():
    image = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    return {
        "agentview_image": image, "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }


class ScriptedClient:
    encode_image_to_data_url = staticmethod(HiMeChatClient.encode_image_to_data_url)

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def chat(self, messages, max_tokens=512):
        self.messages.append(list(messages))
        return next(self.replies)


class TestEncoder:
    def encode_obj(self, text):
        return [1.0, 0.0]

    def encode_text(self, text):
        return [1.0, 0.0]


def xml(operations, plan=""):
    return f"<summary>test</summary><memory_operations>{operations}</memory_operations><plan_list>{plan}</plan_list>"


@pytest.mark.parametrize("last_status,fallback", [("not_done", 2), ("done", 50)])
def test_upstream_two_turn_memory_sentry_fallback_and_isolation(tmp_path, monkeypatch, last_status, fallback):
    root = PROJECT_ROOT / "HiMe"
    if not (root / "server/task_manager.py").exists():
        pytest.skip("Original HiMe checkout is required for this integration test")
    monkeypatch.chdir(tmp_path)
    query = "<operation><type>QUERY</type><query>cookies</query></operation>"
    create = "<operation><type>CREATE</type><tags>cookies</tags><text>cookies on table</text><image_path>1</image_path></operation>"
    planner = ScriptedClient([
        xml(query), xml(create, "[current] pick cookies"),
        xml(query), xml("", "[done] pick cookies"),
    ])
    sentry = ScriptedClient(["<status>not_done</status>"] * 2 + [f"<status>{last_status}</status>"])
    encoder = RecordedEncoder(TestEncoder(), tmp_path / "embeddings.jsonl")
    session = HiMeSession(root, PROMPT_DIR, planner, sentry, encoder, fallback_images=fallback)
    try:
        state = session.start("pick cookies", observation())
        assert len(planner.messages) == 2
        assert len(session.memory.all()) == 1
        for _ in range(2):
            session.update([session.frames(observation())])
            assert len(planner.messages) == 2
            assert not state.is_done
        session.update([session.frames(observation())])
        assert len(planner.messages) == 4
        assert state.is_done
        assert "cookies on table" in json.dumps(planner.messages[-1])
        assert any(
            block.get("meta", {}).get("memory_image_ref") == "IMG#1"
            for message in planner.messages[-1] if isinstance(message["content"], list)
            for block in message["content"]
        )
        assert len(sentry.messages) == 3
        fresh = HiMeSession(root, PROMPT_DIR, planner, sentry, encoder)
        try:
            assert fresh.memory.all() == []
        finally:
            fresh.close()
    finally:
        session.close()


def test_chat_model_parameters_usage_and_budget(tmp_path):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<status>done</status>"), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 42}),
        )

    client = HiMeChatClient(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="deepseek-test", role="planner", log_path=tmp_path / "calls.jsonl",
        max_tokens=8192, max_calls=1, extra_body={"thinking": {"type": "disabled"}},
    )
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret-image"}, "meta": {"path": "frame.png"}},
    ]}]
    assert client.chat(messages, max_tokens=4096) == "<status>done</status>"
    assert requests[0]["max_tokens"] == 8192
    assert requests[0]["extra_body"]["thinking"]["type"] == "disabled"
    assert "meta" not in requests[0]["messages"][0]["content"][0]
    assert messages[0]["content"][0]["meta"] == {"path": "frame.png"}
    logged = (tmp_path / "calls.jsonl").read_text()
    assert "secret-image" not in logged
    assert json.loads(logged)["usage"]["total_tokens"] == 42
    with pytest.raises(ModelBudgetExceededError):
        client.chat(messages)
    assert len(requests) == 1


class FakeEnv:
    task_description = "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket."

    def __init__(self, end_at=25):
        self.steps = 0
        self.end_at = end_at
        self.settles = 0

    def reset(self):
        return observation()

    def step(self, action):
        self.steps += 1
        return SimpleNamespace(observation=observation(), done=self.steps >= self.end_at)

    def metrics(self):
        raise AssertionError("Scoring must never enter the agent loop")

    def settle(self):
        self.settles += 1
        return SimpleNamespace(done=True, observation=observation())


class FakeSession:
    frames = staticmethod(HiMeSession.frames)

    def __init__(self, done=False):
        self.state = SimpleNamespace(
            plan_list="[done] pick cookies" if done else "[current] pick cookies",
            current_subtask_description=None if done else "pick cookies", is_done=done,
        )
        self.updates = []

    def start(self, instruction, raw):
        self.instruction = instruction
        return self.state

    def update(self, frames):
        self.updates.append(frames)
        return self.state


@pytest.mark.parametrize("end_at,max_steps", [(25, 100), (100, 25)])
@pytest.mark.parametrize("returned_actions", [10, 50])
def test_loop_stops_mid_chunk_without_evaluator_or_tail_model_call(tmp_path, end_at, max_steps, returned_actions):
    env, session = FakeEnv(end_at), FakeSession()
    payloads, videos = [], []

    def infer(payload):
        payloads.append(payload)
        return {"actions": np.zeros((returned_actions, 7))}

    outcome = execute_episode(
        env, SimpleNamespace(infer=infer), session, SimpleNamespace(publish=videos.append),
        max_steps=max_steps, action_chunk_size=20, sentry_interval=1,
        trace_path=tmp_path / "trace.jsonl",
    )
    assert env.steps == 25 and env.settles == 0
    assert len(payloads) == (3 if returned_actions == 10 else 2)
    assert len(videos) == 26
    assert len(session.updates) == len(payloads) - 1
    assert payloads[0]["prompt"] == "pick cookies"
    assert payloads[0]["observation/image"].shape == (256, 256, 3)
    assert "Ordered checkpoint-era VLA labels" in session.instruction
    assert outcome["stop_reason"] == ("environment_done" if end_at == 25 else "action_budget")


def test_hime_done_only_requests_benchmark_settle(tmp_path):
    env = FakeEnv()
    outcome = execute_episode(
        env, None, FakeSession(done=True), SimpleNamespace(publish=lambda raw: None),
        max_steps=100, action_chunk_size=20, sentry_interval=5,
        trace_path=tmp_path / "trace.jsonl",
    )
    assert env.settles == 1 and env.steps == 0
    assert outcome["stop_reason"] == "hime_done"
    assert "success" not in outcome
