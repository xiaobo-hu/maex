"""Bridge LIBERO policy observations to RoboAgent's provider-neutral types."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO

import numpy as np
from PIL import Image

from roboagent import ImageFrame, Observation

from .preprocessing import build_policy_input

_OPENPI_STATE_KEY = "openpi_state"


def build_roboagent_observation(
    raw: object,
    *,
    success: bool | None = None,
) -> Observation:
    """Create one LLM/VLA-compatible observation from raw LIBERO values."""
    if not isinstance(raw, Mapping):
        raise TypeError("LIBERO observation must be a mapping")
    payload = build_policy_input(raw, "")
    state = np.asarray(payload["observation/state"])
    gripper_qpos = np.asarray(raw["robot0_gripper_qpos"]).tolist()
    return Observation(
        rgb=_encode_png(np.asarray(payload["observation/image"])),
        wrist_rgb=_encode_png(np.asarray(payload["observation/wrist_image"])),
        robot_state={
            _OPENPI_STATE_KEY: state.tolist(),
            "eef_position": state[:3].tolist(),
            "eef_axis_angle": state[3:6].tolist(),
            "gripper_qpos": gripper_qpos,
        },
        task_success=success,
    )


def build_openpi_input(instruction: str, observation: Observation) -> dict[str, object]:
    """Recover the exact pi05_libero payload stored in a RoboAgent observation."""
    if observation.rgb is None or observation.wrist_rgb is None:
        raise ValueError("LIBERO observation requires main and wrist RGB images")
    state = observation.robot_state.get(_OPENPI_STATE_KEY)
    if not isinstance(state, list):
        raise ValueError("LIBERO observation is missing openpi_state")
    return {
        "observation/image": _decode_png(observation.rgb),
        "observation/wrist_image": _decode_png(observation.wrist_rgb),
        "observation/state": np.asarray(state),
        "prompt": instruction,
    }


def _encode_png(image: np.ndarray) -> ImageFrame:
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return ImageFrame(data=base64.b64encode(output.getvalue()).decode("ascii"))


def _decode_png(frame: ImageFrame) -> np.ndarray:
    raw = base64.b64decode(frame.data)
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image).copy()
