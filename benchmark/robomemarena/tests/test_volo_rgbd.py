"""验证相机投影、深度单位和 VoLo 抓取图像方向，全程不启动仿真或 GPU。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmark.robomemarena.baselines.volo_observation import build_volo_observation
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.preprocessing import build_policy_input
from benchmark.robomemarena.server.camera_observation import camera_observation
from benchmark.robomemarena.server.service import RoboMemArenaEnvironmentService


def _scene():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    raw = {
        "agentview_image": image, "robot0_eye_in_hand_image": image.copy(),
        "agentview_depth": np.full((4, 6, 1), 0.5),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]), "stage": "private",
    }
    sim = SimpleNamespace(
        model=SimpleNamespace(
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.1, zfar=10.0)),
            stat=SimpleNamespace(extent=2.0), cam_fovy=[90.0], camera_name2id=lambda _: 0,
        ),
        data=SimpleNamespace(cam_xmat=[np.eye(3).flatten()], cam_xpos=[[1.0, 2.0, 3.0]]),
    )
    return raw, sim


def test_depth_projection_and_policy_input_unchanged():
    raw, sim = _scene()
    depth = camera_observation(sim, raw)
    assert depth["agentview_depth_m"].shape == (4, 6)
    np.testing.assert_allclose(depth["agentview_depth_m"], 0.2 / (1 - 0.5 * 0.99))
    np.testing.assert_allclose(depth["agentview_camera_K"], [[2, 0, 3], [0, 2, 2], [0, 0, 1]])
    # 中心像素沿相机 OpenGL -Z 轴投影，验证外参无需再求逆。
    center_ray = np.linalg.inv(depth["agentview_camera_K"]) @ [3, 2, 1]
    point = depth["agentview_camera_to_world"] @ [*center_ray, 1]
    np.testing.assert_allclose(point, [1, 2, 2, 1])

    raw.update(depth)
    payload = build_volo_observation(raw, "pick up the cube", enable_depth=True)
    expected = build_policy_input(raw, "pick up the cube")
    for key, value in expected.items():
        np.testing.assert_equal(payload[key], value)
    np.testing.assert_equal(payload["observation/image_for_grasp"], raw["agentview_image"][::-1])
    np.testing.assert_equal(payload["observation/depth_for_grasp"], depth["agentview_depth_m"][::-1])
    np.testing.assert_equal(payload["observation/ee_quat"], [1, 0, 0, 0])
    np.testing.assert_allclose(payload["observation/gripper_position"], [0])
    assert "stage" not in payload
    raw["agentview_depth_m"][0, 0] = np.nan
    with pytest.raises(ValueError, match="depth"):
        build_volo_observation(raw, "test", enable_depth=True)


def test_service_depth_is_opt_in_and_excludes_evaluator_state():
    raw, sim = _scene()
    service = RoboMemArenaEnvironmentService()
    service._observation = raw
    service._env = SimpleNamespace(sim=sim)
    service._evaluator = SimpleNamespace(done=False, success=False)
    service._config = RoboMemArenaConfig()
    assert "enable_depth" not in service._config.to_reset_params()
    original = service.dispatch("observe", {})["observation"]
    assert "agentview_depth_m" not in original
    service._config = RoboMemArenaConfig(enable_depth=True)
    assert service._config.to_reset_params()["enable_depth"] is True
    observation = service.dispatch("observe", {})["observation"]
    assert "agentview_depth_m" in observation
    assert "agentview_camera_to_world" in observation
    assert "stage" not in observation
    assert "agentview_depth" not in observation


def test_native_volo_uses_calibrated_grasp_images_without_extra_flip():
    tool_module = pytest.importorskip("vlm_orchestrator.grasp.tool")
    raw, sim = _scene()
    raw.update(camera_observation(sim, raw))
    payload = build_volo_observation(raw, "pick up the cube", enable_depth=True)
    tool = tool_module.GraspToolExecutor(env_mode="libero", seg_mode="sam3")
    np.testing.assert_equal(tool._extract_image(payload), raw["agentview_image"][::-1])
    np.testing.assert_equal(tool._extract_depth(payload), raw["agentview_depth_m"][::-1])
    np.testing.assert_equal(tool._get_camera_to_world(payload), raw["agentview_camera_to_world"])
    assert tool._action_dim == 7


@pytest.mark.parametrize("enabled", [False, True])
def test_reset_requests_depth_only_when_enabled(monkeypatch, enabled):
    from benchmark.robomemarena.server import service as service_module

    raw, sim = _scene()
    raw["robot0_joint_pos"] = np.zeros(7)
    created = {}

    def make_env(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(sim=sim, reset=lambda: raw, close=lambda: None)

    common = SimpleNamespace(
        _resolve_bddl_path=lambda task: Path("task4.bddl"),
        _get_env_class=lambda: make_env, get_prompt=lambda *args: "pick up the cube",
    )
    monkeypatch.setattr(service_module, "load_official_modules", lambda: (common, None))
    monkeypatch.setattr(service_module, "OfficialStageEvaluator",
                        lambda *args, **kwargs: SimpleNamespace(done=False, success=False))
    service = RoboMemArenaEnvironmentService()
    params = RoboMemArenaConfig(task_id=4, warmup_steps=0, enable_depth=enabled).to_reset_params()
    response = service.dispatch("reset", params)
    assert (created.get("camera_depths") is True) == enabled
    assert ("agentview_depth_m" in response["observation"]) == enabled
    if not enabled:
        assert "camera_depths" not in created
