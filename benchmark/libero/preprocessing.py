"""Official LIBERO observation preprocessing for the pi05_libero policy."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ._vendor.openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_POLICY_RESOLUTION = 224
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def build_policy_input(
    observation: Mapping[str, object],
    instruction: str,
    *,
    resize_size: int = LIBERO_POLICY_RESOLUTION,
) -> dict[str, object]:
    """Convert a raw LIBERO observation into the official OpenPI payload."""
    image = _prepare_image(observation["agentview_image"], resize_size)
    wrist_image = _prepare_image(observation["robot0_eye_in_hand_image"], resize_size)
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"])),
            np.asarray(observation["robot0_gripper_qpos"]),
        )
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": instruction,
    }


def select_policy_observation(observation: Mapping[str, object]) -> dict[str, object]:
    """Keep only the five LIBERO fields consumed by pi05_libero."""
    keys = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    return {key: observation[key] for key in keys}


def quat2axisangle(quaternion: np.ndarray) -> np.ndarray:
    """Convert LIBERO's quaternion to the axis-angle form used during training."""
    quaternion = np.asarray(quaternion).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def _prepare_image(raw: object, resize_size: int) -> np.ndarray:
    image = np.ascontiguousarray(np.asarray(raw)[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, resize_size, resize_size)
    )
