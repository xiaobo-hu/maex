from __future__ import annotations

import numpy as np
from benchmark.libero.observation import build_openpi_input, build_roboagent_observation
from benchmark.libero.preprocessing import build_policy_input


def _raw_observation() -> dict[str, object]:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
    }


def test_policy_input_matches_libero_preprocessing() -> None:
    raw = _raw_observation()
    payload = build_policy_input(raw, "pick", resize_size=2)

    np.testing.assert_array_equal(
        payload["observation/image"],
        np.asarray(raw["agentview_image"])[::-1, ::-1],
    )
    np.testing.assert_allclose(
        payload["observation/state"],
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 0.2],
    )
    assert payload["prompt"] == "pick"


def test_roboagent_observation_round_trips_openpi_payload() -> None:
    raw = _raw_observation()
    observation = build_roboagent_observation(raw, success=False)
    payload = build_openpi_input("place", observation)
    expected = build_policy_input(raw, "place")

    np.testing.assert_array_equal(payload["observation/image"], expected["observation/image"])
    np.testing.assert_array_equal(
        payload["observation/wrist_image"], expected["observation/wrist_image"]
    )
    np.testing.assert_allclose(payload["observation/state"], expected["observation/state"])
    assert observation.robot_state["gripper_qpos"] == [0.1, 0.2]
    assert observation.task_success is False
