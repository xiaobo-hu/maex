"""验证外部基线桥接的终止、观测与动作边界，不复制 HarnessVLA 策略。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from benchmark.robomemarena.baselines.harness_bridge import HarnessEnvBridge, HarnessPolicyBridge
from benchmark.robomemarena.baselines.harness_server import HarnessEnvironmentService
from benchmark.robomemarena.environment import RoboMemArenaStep
from benchmark.robomemarena.preprocessing import build_policy_input


def observation():
    return {
        "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }


class FakeRemote:
    task_description = "pour twice"

    def __init__(self, *, success=False):
        self.success = success
        self.actions = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        return observation()

    def step(self, action):
        self.actions.append(action)
        return RoboMemArenaStep(
            observation=observation(), done=len(self.actions) == 2,
            success=self.success,
        )


@pytest.mark.parametrize("success", [True, False])
def test_chunk_stops_at_official_done_and_distinguishes_failure(success):
    remote = FakeRemote(success=success)
    env = HarnessEnvBridge(remote)
    env.reset()
    frames, _rewards, term, trunc, info = env.chunk_step(
        np.zeros((20, 7)), return_all_frames=True,
    )
    assert len(remote.actions) == len(frames) == 2
    assert term.tolist() == [False, success]
    assert trunc.tolist() == [False, not success]
    assert info == {}
    with pytest.raises(RuntimeError, match="ended"):
        env.step(np.zeros(7))


def test_evaluation_cannot_reset_or_expose_reward_shaping():
    remote = FakeRemote()
    env = HarnessEnvBridge(remote)
    env.reset()
    with pytest.raises(RuntimeError, match="one episode"):
        env.reset()
    assert remote.resets == 1
    assert env.step(np.zeros(7))[1] == 0.0


@pytest.mark.parametrize("success", [True, False])
def test_exploration_reset_clears_terminal_state(success):
    remote = FakeRemote(success=success)
    env = HarnessEnvBridge(remote, allow_reset=True)
    env.reset()
    env.chunk_step(np.zeros((3, 7)))
    assert env.terminated == success
    assert env.truncated == (not success)
    env.reset()
    assert remote.resets == 2
    assert not env.terminated and not env.truncated
    env.step(np.zeros(7))


def test_new_policy_entry_keeps_short_checkpoint_output():
    model = HarnessPolicyBridge(SimpleNamespace(infer=lambda _obs: {"actions": np.ones((10, 7))}))
    obs, _ = HarnessEnvBridge(FakeRemote()).reset()
    np.testing.assert_array_equal(model.predict(obs, options={"mode": "eval"}), np.ones((10, 7)))


def test_reset_archives_actual_episodes_and_fixed_seed(tmp_path):
    from benchmark.robomemarena.baselines.harness_official import RecordedHarnessEnv
    from benchmark.robomemarena.config import RoboMemArenaConfig
    import json

    remote = FakeRemote()
    remote.metrics = lambda: {"TSR": 0, "CSR": 0, "step": len(remote.actions)}
    env = RecordedHarnessEnv(
        remote, config=RoboMemArenaConfig(task_id=1, seed=0),
        episode_dir=tmp_path, allow_reset=True,
    )
    env.reset()
    env.step(np.zeros(7))
    env.reset()
    env.archive_episode()
    assert env.episode_count == 2
    assert json.loads((tmp_path / "episode_001/evaluation.json").read_text())["step"] == 1
    assert json.loads((tmp_path / "episode_002/summary.json").read_text())[0]["episodes"][0]["seed"] == 0


def test_official_task_context_matches_other_baselines(tmp_path):
    from benchmark.robomemarena.baselines.harness_official import RecordedHarnessEnv
    from benchmark.robomemarena.config import RoboMemArenaConfig
    from roboagent.prompts.robomemarena import task_primitive_reference

    remote = FakeRemote()
    remote.task_description = "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket."
    env = RecordedHarnessEnv(
        remote, config=RoboMemArenaConfig(task_id=1), episode_dir=tmp_path, allow_reset=True,
    )
    reference = task_primitive_reference(remote.task_description)
    assert "pick cookies" in reference
    assert env.get_task_language() == remote.task_description + "\n\n" + reference
    obs, _ = env.reset()
    # 原模型接口保留任务本身；每次动作的子指令仍由官方 Agent 选择。
    assert obs["task_descriptions"] == remote.task_description


@pytest.mark.parametrize("horizon", [10, 40])
def test_policy_preserves_benchmark_preprocessing_and_subtask(horizon):
    captured = {}

    def infer(payload):
        captured.update(payload)
        return {"actions": np.arange(horizon * 7).reshape(horizon, 7)}

    env = HarnessEnvBridge(FakeRemote())
    obs, _ = env.reset()
    obs["task_descriptions"] = "grasp the sauce bottle"
    model = HarnessPolicyBridge(SimpleNamespace(infer=infer))
    actions, _ = model.predict_action_batch(obs)
    expected = build_policy_input(observation(), "grasp the sauce bottle")
    assert captured.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_equal(captured[key], value)
    executed = min(horizon, 20)
    np.testing.assert_array_equal(actions, np.arange(executed * 7).reshape(executed, 7))


@pytest.mark.parametrize("actions", [
    np.zeros((20, 8)), np.full((20, 7), np.nan), np.zeros((0, 7)),
])
def test_policy_rejects_invalid_actions(actions):
    model = HarnessPolicyBridge(SimpleNamespace(infer=lambda _obs: {"actions": actions}))
    obs, _ = HarnessEnvBridge(FakeRemote()).reset()
    with pytest.raises(ValueError, match="finite actions"):
        model.predict_action_batch(obs)


def test_rgbd_read_is_passive_and_does_not_expose_object_coordinates():
    rendered = []

    def render(**params):
        rendered.append(params)
        return np.zeros((256, 256, 3), dtype=np.uint8), np.full((256, 256), 0.5)

    service = HarnessEnvironmentService()
    service._env = SimpleNamespace(sim=SimpleNamespace(render=render))
    service._observation = {
        **observation(), "sauce_pos": np.array([9.0, 8.0, 7.0]),
        "sauce_quat": np.ones(4), "hidden_stage": "pour_2",
    }
    result = service.dispatch("harness_raw_obs", {})["raw"]
    assert result["sauce_pos"] is None
    assert "sauce_quat" not in result
    assert "hidden_stage" not in result
    assert len(rendered) == 2
    assert result["agentview_depth"].shape == (256, 256)
    assert result["robot0_eye_in_hand_image"].shape == (256, 256, 3)


def test_camera_api_rejects_non_benchmark_views():
    service = HarnessEnvironmentService()
    with pytest.raises(ValueError, match="external and wrist"):
        service.dispatch("harness_render_camera", {"camera_name": "birdview"})


def test_camera_calibration_uses_resolution_and_opencv_axes():
    sim = SimpleNamespace(
        model=SimpleNamespace(
            camera_name2id=lambda _name: 0, cam_fovy=[90],
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10)),
            stat=SimpleNamespace(extent=2),
        ),
        data=SimpleNamespace(cam_xmat=np.eye(3).reshape(1, 9), cam_xpos=[[1, 2, 3]]),
    )
    service = HarnessEnvironmentService()
    service._env = SimpleNamespace(sim=sim)
    meta = service.dispatch("harness_camera_meta", {"height": 100, "width": 200})["meta"]
    np.testing.assert_allclose(meta["intrinsic_K"], [[50, 0, 100], [0, 50, 50], [0, 0, 1]])
    # 中心像素前方 1 米，位于 MuJoCo 相机局部 -Z 方向。
    world = np.asarray(meta["extrinsic_cam2world"]) @ [0, 0, 1, 1]
    np.testing.assert_allclose(world, [1, 2, 2, 1])
    assert meta["depth_near"] == 0.02
    assert meta["depth_far"] == 20


def test_official_policy_observation_preserves_libero_orientation_and_state():
    from benchmark.robomemarena.baselines.harness_bridge import OfficialHarnessEnvBridge
    raw = observation()
    raw['robot0_eef_quat'] = np.array([0.0, 0.0, np.sqrt(0.5), -np.sqrt(0.5)])
    before = raw['robot0_eef_quat'].copy()
    obs = OfficialHarnessEnvBridge(FakeRemote())._observation(raw)
    np.testing.assert_array_equal(obs['main_images'], raw['agentview_image'][::-1, ::-1])
    assert obs['main_images'].flags.c_contiguous
    assert obs['main_images'].shape == (2, 2, 3)
    np.testing.assert_allclose(obs['states'][3:6], [0, 0, 1.5*np.pi])
    np.testing.assert_array_equal(raw['robot0_eef_quat'], before)


def test_official_client_passes_five_actions_without_padding(monkeypatch):
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / '.external/RPent-official'
    monkeypatch.syspath_prepend(str(source))
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    calls = []
    class Rpc:
        def call(self, method, *, args, timeout_s):
            calls.append((method, args))
            return np.arange(35, dtype=np.float32).reshape(1, 5, 7)
    env_obs = dict(main_images=np.zeros((256,256,3),np.uint8),
                   wrist_images=np.zeros((256,256,3),np.uint8),
                   states=np.zeros(8), task_descriptions='pick up the bowl')
    actions = Pi05VLAClient(Rpc(), embodiment='libero').predict(env_obs, options={'mode':'eval'})
    assert actions.shape == (5,7)
    method, args = calls[0]
    assert method == 'vla.predict'
    assert args[0]['main_images'].shape == (1,256,256,3)
    assert args[0]['task_descriptions'] == ['pick up the bowl']
