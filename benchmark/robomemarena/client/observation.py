"""Convert RoboMemArena values to provider-neutral RoboAgent observations."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO

import numpy as np
from PIL import Image

from roboagent import ImageFrame, Observation

from ..preprocessing import build_policy_input

_OPENPI_STATE_KEY = "openpi_state"
_GRIPPER_QPOS_KEY = "robot0_gripper_qpos"


def build_roboagent_observation(raw: object, *, success: bool | None = None) -> Observation:
    """Expose only dual RGB, robot state, and final task success to RoboAgent."""
    if not isinstance(raw, Mapping):
        raise TypeError("RoboMemArena observation must be a mapping")
    payload = build_policy_input(raw, "")
    state = np.asarray(payload["observation/state"])
    return Observation(
        rgb=_encode_png(np.asarray(payload["observation/image"])),
        wrist_rgb=_encode_png(np.asarray(payload["observation/wrist_image"])),
        robot_state={
            _OPENPI_STATE_KEY: state.tolist(),
            "eef_position": state[:3].tolist(),
            "eef_axis_angle": state[3:6].tolist(),
            "gripper_qpos": np.asarray(raw[_GRIPPER_QPOS_KEY]).tolist(),
        },
        task_success=success,
        metadata={},
    )


def build_openpi_input(instruction: str, observation: Observation) -> dict[str, object]:
    """Recover an OpenPI input payload; this function never invokes a policy."""
    if observation.rgb is None or observation.wrist_rgb is None:
        raise ValueError("RoboMemArena observation requires main and wrist RGB images")
    state = observation.robot_state.get(_OPENPI_STATE_KEY)
    if not isinstance(state, list):
        raise ValueError("RoboMemArena observation is missing openpi_state")
    main = _decode_png(observation.rgb)
    wrist = _decode_png(observation.wrist_rgb)
    return {
        "observation/image": main,
        "observation/wrist_image": wrist,
        "base_0_rgb": main,
        "left_wrist_0_rgb": np.zeros_like(wrist),
        "right_wrist_0_rgb": wrist,
        "observation/state": np.asarray(state),
        "prompt": instruction,
    }


def _encode_png(image: np.ndarray) -> ImageFrame:
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return ImageFrame(data=base64.b64encode(output.getvalue()).decode("ascii"))


def _decode_png(frame: ImageFrame) -> np.ndarray:
    with Image.open(BytesIO(base64.b64decode(frame.data))) as image:
        return np.asarray(image).copy()
