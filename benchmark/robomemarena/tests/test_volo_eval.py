"""验证 VoLo 客户端的评分隔离、步数预算和终态停止，不启动模型或仿真。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmark.robomemarena.baselines.volo_eval import execute_episode


@pytest.mark.parametrize("reference", [None, "Ordered labels: pick cookies; place cookies"])
def test_native_vlm_roles_share_reference_without_cross_episode_leak(monkeypatch, reference):
    module = pytest.importorskip("vlm_orchestrator.strategies.subgoal")
    base = pytest.importorskip("vlm_orchestrator.strategies.base")
    vlm = pytest.importorskip("vlm_orchestrator.vlm")
    strategy = module.SubgoalStrategy(
        base.StrategyContext(vlm=None, image_key="observation/image",
                             extra_image_keys=["observation/wrist_image"]), module.SubgoalConfig(),
        failure_monitor="vlm", recovery_mode="replan_grasp",
    )
    strategy._client = object()
    messages = []

    def call(client, **kwargs):
        messages.append(kwargs["messages"])
        content = ('{"status":"in_progress","action":"continue","reason":"uncertain"}'
                   if len(messages) == 2 else '{"subgoals":["pick cookies"],"ordered":true}')
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=content),
        )])

    monkeypatch.setattr(vlm, "chat_create", call)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.full_like(image, 40)
    obs = {"prompt": "original task", "observation/image": image, "observation/wrist_image": wrist}
    if reference:
        obs["__task_primitive_reference"] = reference
    state = base.SessionState()
    strategy._on_new_episode(obs, state, "original task")
    state.episode_step = 40
    now = np.full_like(image, 180)
    now_wrist = np.full_like(image, 220)
    strategy._failure_handler.step({**obs, "observation/image": now,
                                   "observation/wrist_image": now_wrist}, state)
    strategy._recycle(obs, state, image)
    assert len(messages) == 3
    for request in messages:
        texts = [block["text"] for block in request[1]["content"] if block["type"] == "text"]
        assert sum(text == reference for text in texts) == int(bool(reference))
        assert ("strong reference" in request[0]["content"]) == bool(reference)
    monitor = messages[1][0]["content"]
    api = pytest.importorskip("vlm_orchestrator.vlm.api")
    image_urls = [block["image_url"]["url"] for block in messages[1][1]["content"]
                  if block["type"] == "image_url"]
    assert image_urls == ["data:image/jpeg;base64," + api.encode_image_b64(frame)
                          for frame in (image, wrist, now, now_wrist)]
    assert "CURRENT instruction's observable end state" not in monitor
    assert "at its destination" in monitor and "moved to its destination" in monitor
    assert "fire only for WRONG-OBJECT recovery" in monitor
    assert state.subgoals == ["pick cookies"]
    # 新任务参考必须替换旧参考；无参考任务不能继承上一任务的信息。
    strategy._on_new_episode({**obs, "__task_primitive_reference": "Task B reference"}, state, "Task B")
    assert messages[-1][1]["content"][-1]["text"] == "Task B reference"
    strategy._on_new_episode({"observation/image": image}, state, "Task C")
    assert "strong reference" not in messages[-1][0]["content"]
    assert all("Task B reference" not in block.get("text", "") for block in messages[-1][1]["content"])


def test_interrupted_episode_is_invalid(monkeypatch, tmp_path):
    from benchmark.robomemarena.baselines import volo_eval

    run_dir = tmp_path / "interrupted"
    monkeypatch.setattr("sys.argv", ["volo_eval", "--run-dir", str(run_dir)])
    policy = SimpleNamespace(get_server_metadata=lambda: {"orchestrator": True}, close=lambda: None)
    env = SimpleNamespace(metrics=lambda: {}, task_description="task", close=lambda: None)
    monkeypatch.setattr(volo_eval, "WebsocketPolicyClient", lambda *a: policy)
    monkeypatch.setattr(volo_eval, "RemoteRoboMemArenaEnv", lambda *a, **kw: env)
    monkeypatch.setattr(volo_eval, "RoboMemArenaVideoRecorder",
                        lambda *a, **kw: SimpleNamespace(close=lambda **kw: {}))
    monkeypatch.setattr(volo_eval, "write_official_artifacts", lambda *a, **kw: None)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(volo_eval, "execute_episode", interrupt)
    assert volo_eval.main() == 130
    result = json.loads((run_dir / "result.json").read_text())
    assert result["valid_rollout"] is False
    assert result["stop_reason"] == "interrupted"


@pytest.mark.parametrize("enable_reference", [False, True])
def test_episode_boundary_budget_and_no_privileged_observation(tmp_path, enable_reference):
    raw = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2), "CSR": 100, "stage": "secret",
    }
    calls, executed = [], []

    def infer(payload):
        calls.append(payload)
        return {"actions": np.zeros((10, 7)), "orchestrator_flush_actions": True}

    def step(action):
        executed.append(action)
        return SimpleNamespace(observation=raw, done=False)

    instruction = "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket."
    env = SimpleNamespace(reset=lambda: raw, task_description=instruction, step=step)
    policy = SimpleNamespace(infer=infer)
    recorder = SimpleNamespace(publish=lambda obs: None)
    result = execute_episode(
        env, policy, recorder, max_steps=7, action_chunk_size=5, trace_path=tmp_path / "a.jsonl",
        enable_task_reference=enable_reference,
    )
    assert result["stop_reason"] == "action_budget"
    assert len(executed) == 7
    assert [call["__step"] for call in calls] == [0, 5]
    assert len({call["__episode_id"] for call in calls}) == 1
    assert all("CSR" not in call and "stage" not in call for call in calls)
    assert all(call["prompt"] == env.task_description for call in calls)
    if enable_reference:
        reference = calls[0]["__task_primitive_reference"]
        assert "task 1, advisory" in reference
        assert "cookies" in reference and "tomato sauce" in reference
        assert (tmp_path / "task_reference.txt").read_text() == reference + "\n"
    else:
        assert all("__task_primitive_reference" not in call for call in calls)

    first_id = calls[0]["__episode_id"]
    calls.clear()
    executed.clear()

    def terminal_step(action):
        executed.append(action)
        return SimpleNamespace(observation=raw, done=True)

    env.step = terminal_step
    result = execute_episode(
        env, policy, recorder, max_steps=7, action_chunk_size=5, trace_path=tmp_path / "b.jsonl",
    )
    assert result["stop_reason"] == "environment_done"
    assert len(executed) == 1
    assert calls[0]["__episode_id"] != first_id

    policy.infer = lambda payload: {"actions": np.full((5, 7), np.nan)}
    with pytest.raises(ValueError, match="finite"):
        execute_episode(
            env, policy, recorder, max_steps=7, action_chunk_size=5, trace_path=tmp_path / "c.jsonl",
        )
    assert len(executed) == 1
