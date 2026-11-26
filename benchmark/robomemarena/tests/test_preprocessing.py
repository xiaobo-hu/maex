from __future__ import annotations

import cv2
import numpy as np
import pytest
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.preprocessing import (
    build_policy_input,
    process_image,
    validate_action,
)


def _raw_observation() -> dict[str, object]:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, -1.0]),
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
    }


def test_official_image_flip_resize_and_state() -> None:
    raw = _raw_observation()
    payload = build_policy_input(raw, "pick", resize_size=2)
    expected = cv2.resize(
        np.flipud(np.asarray(raw["agentview_image"])),
        (2, 2),
        interpolation=cv2.INTER_AREA,
    )

    np.testing.assert_array_equal(payload["observation/image"], expected)
    np.testing.assert_allclose(
        payload["observation/state"],
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 0.2],
    )
    assert payload["prompt"] == "pick"


def test_image_is_not_horizontally_flipped() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    np.testing.assert_array_equal(process_image(image, 2), np.flipud(image))


def test_roboagent_observation_round_trips_openpi_payload_without_metrics() -> None:
    from benchmark.robomemarena.client.observation import (
        build_openpi_input,
        build_roboagent_observation,
    )

    raw = _raw_observation()
    observation = build_roboagent_observation(raw, success=False)
    payload = build_openpi_input("place", observation)
    expected = build_policy_input(raw, "place")

    np.testing.assert_array_equal(payload["observation/image"], expected["observation/image"])
    np.testing.assert_array_equal(
        payload["observation/wrist_image"], expected["observation/wrist_image"]
    )
    np.testing.assert_allclose(payload["observation/state"], expected["observation/state"])
    np.testing.assert_allclose(observation.robot_state["gripper_qpos"], [0.1, 0.2])
    np.testing.assert_allclose(observation.robot_state["eef_position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(observation.robot_state["eef_axis_angle"], [0.0, 0.0, 0.0])
    assert observation.metadata == {}
    assert observation.task_success is False


@pytest.mark.parametrize("task_id", [0, 27])
def test_task_id_validation(task_id: int) -> None:
    with pytest.raises(ValueError, match="task_id"):
        RoboMemArenaConfig(task_id=task_id)


def test_seed_max_steps_and_action_validation() -> None:
    with pytest.raises(ValueError, match="seed"):
        RoboMemArenaConfig(seed=-1)
    with pytest.raises(ValueError, match="max_steps"):
        RoboMemArenaConfig(max_steps=0)
    with pytest.raises(ValueError, match="7 finite"):
        validate_action([0.0] * 6)
    with pytest.raises(ValueError, match="7 finite"):
        validate_action([0.0] * 6 + [float("nan")])
