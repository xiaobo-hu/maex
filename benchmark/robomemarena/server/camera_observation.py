"""环境服务层：导出工具所需的相机深度与标定，不读取对象或官方评分状态。"""

from __future__ import annotations

import numpy as np


def camera_observation(sim, observation) -> dict:
    """返回与原始 agentview RGB 同帧、同方向的米制深度及 OpenCV 相机位姿。"""
    image = np.asarray(observation["agentview_image"])
    depth = np.asarray(observation["agentview_depth"], dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = image.shape[:2]
    if depth.shape != (height, width) or not np.isfinite(depth).all():
        raise ValueError("agentview depth must be finite and match RGB dimensions")
    if np.any(depth < 0) or np.any(depth > 1):
        raise ValueError("MuJoCo depth must be normalized to [0, 1]")
    near = float(sim.model.vis.map.znear * sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
    if not 0 < near < far:
        raise ValueError("Invalid camera clipping planes")
    depth_m = near / (1.0 - depth * (1.0 - near / far))
    camera_id = sim.model.camera_name2id("agentview")
    focal = height / (2.0 * np.tan(np.deg2rad(sim.model.cam_fovy[camera_id]) / 2.0))
    intrinsics = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]])
    camera_to_world = np.eye(4)
    # MuJoCo 相机看向 -Z、Y 向上；OpenCV 看向 +Z、Y 向下。
    camera_to_world[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3) @ np.diag([1, -1, -1])
    camera_to_world[:3, 3] = sim.data.cam_xpos[camera_id]
    return {
        "agentview_depth_m": depth_m.astype(np.float32),
        "agentview_camera_K": intrinsics,
        "agentview_camera_to_world": camera_to_world,
    }
