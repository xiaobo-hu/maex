"""基线适配层：为 VoLo 抓取工具补充标定 RGB-D，保留官方 VLA 预处理。"""

from __future__ import annotations

import numpy as np

from benchmark.robomemarena.preprocessing import build_policy_input


def build_volo_observation(raw, instruction: str, *, enable_depth: bool = False) -> dict:
    """构造 VoLo 输入；抓取专用字段避免其 LIBERO 分支再次水平翻转图像。"""
    wire = build_policy_input(raw, instruction)
    if not enable_depth:
        return wire
    rgb = np.ascontiguousarray(np.flipud(np.asarray(raw["agentview_image"])))
    depth = np.ascontiguousarray(np.flipud(np.asarray(raw["agentview_depth_m"], dtype=np.float32)))
    intrinsics = np.asarray(raw["agentview_camera_K"], dtype=np.float64)
    extrinsics = np.asarray(raw["agentview_camera_to_world"], dtype=np.float64)
    if (depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or np.any(depth <= 0)
            or intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4)
            or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all()):
        raise ValueError("VoLo requires aligned positive metric depth and finite camera matrices")
    quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
    width = float(np.mean(np.abs(raw["robot0_gripper_qpos"])))
    wire.update({
        "observation/image_for_grasp": rgb,
        "observation/depth_for_grasp": depth,
        "observation/camera_K": intrinsics.flatten(),
        "observation/camera_extrinsic": extrinsics.flatten(),
        "observation/ee_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
        "observation/ee_quat": quat[[3, 0, 1, 2]],
        "observation/gripper_position": np.array([1.0 - np.clip(width / 0.04, 0.0, 1.0)]),
    })
    return wire
