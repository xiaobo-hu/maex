"""Official RoboMemArena image orientation, state, and OpenPI conversion."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ROBOMEMARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ROBOMEMARENA_POLICY_RESOLUTION = 256
_OBSERVATION_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def validate_action(action: object) -> list[float]:
    """Return a finite 7-D action or raise a stable boundary error."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("RoboMemArena action must contain exactly 7 finite values")
    return values.tolist()


def select_policy_observation(observation: Mapping[str, object]) -> dict[str, object]:
    """Keep only images and robot state consumed outside the simulator."""
    return {key: observation[key] for key in _OBSERVATION_KEYS}


def process_image(raw: object, size: int = ROBOMEMARENA_POLICY_RESOLUTION) -> np.ndarray:
    """Match the official recorder: vertical flip then forced INTER_AREA resize."""
    import cv2

    image = np.asarray(raw)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.size and float(image.max()) <= 1.0:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.flipud(image)
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def quat2axisangle(raw: object) -> np.ndarray:
    """Convert robosuite ``[x, y, z, w]`` quaternion to canonical axis-angle."""
    quat = np.asarray(raw, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError("robot0_eef_quat must contain 4 values")
    norm = float(np.linalg.norm(quat))
    if norm > 1e-12:
        quat /= norm
    if quat[3] < 0:
        quat = -quat
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3])) / denominator).astype(np.float32)


def build_policy_input(
    observation: Mapping[str, object],
    instruction: str,
    *,
    resize_size: int = ROBOMEMARENA_POLICY_RESOLUTION,
) -> dict[str, object]:
    """Build the official single-observation OpenPI payload without inference."""
    main = process_image(observation["agentview_image"], resize_size)
    wrist = process_image(observation["robot0_eye_in_hand_image"], resize_size)
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return {
        "observation/image": main,
        "observation/wrist_image": wrist,
        "base_0_rgb": main,
        "left_wrist_0_rgb": np.zeros_like(wrist),
        "right_wrist_0_rgb": wrist,
        "observation/state": state,
        "prompt": str(instruction),
    }
