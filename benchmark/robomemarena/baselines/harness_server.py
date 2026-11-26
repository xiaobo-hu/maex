"""Python 3.8 环境侧适配：为原版 HarnessVLA 提供只读 RGB-D 与标定。"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from benchmark.robomemarena.server.service import (
    RoboMemArenaEnvironmentServer,
    RoboMemArenaEnvironmentService,
)


class HarnessEnvironmentService(RoboMemArenaEnvironmentService):
    """复用现有物理与评分流程，仅增加基线专用的感知 RPC。"""

    def dispatch(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """返回相机观测或委托原有服务；不把隐藏阶段送入感知结果。"""
        if method == "harness_raw_obs":
            return {"raw": self._raw_obs()}
        if method == "harness_render_camera":
            return self._render_camera(**params)
        if method == "harness_camera_meta":
            return {"meta": self._camera_meta(**params)}
        return super().dispatch(method, params)

    def _sim(self):
        if self._env is None:
            raise RuntimeError("Reset the environment before reading cameras")
        return self._env.sim

    @staticmethod
    def _validate_camera(camera_name: str, height: int, width: int) -> None:
        if camera_name not in ("agentview", "robot0_eye_in_hand"):
            raise ValueError("Only the external and wrist cameras are available")
        if not 1 <= height <= 1024 or not 1 <= width <= 1024:
            raise ValueError("Camera dimensions must be within [1, 1024]")

    def _render_camera(
        self, camera_name: str = "agentview", height: int = 1024,
        width: int = 1024, depth: bool = False,
    ) -> dict[str, object]:
        self._validate_camera(camera_name, height, width)
        rendered = self._sim().render(
            camera_name=camera_name, height=height, width=width, depth=depth,
        )
        # 原工具自行翻转 RGB/depth；此处保留 MuJoCo 原始帧方向。
        if depth:
            rgb, depth_image = rendered
            return {"rgb": rgb, "depth": depth_image}
        return {"rgb": rendered}

    def _camera_meta(
        self, camera_name: str = "agentview", height: int = 256, width: int = 256,
    ) -> dict[str, object]:
        self._validate_camera(camera_name, height, width)
        sim = self._sim()
        # 与 robosuite camera_utils 的针孔模型及 OpenCV 轴向修正一致；
        # 直接读取标定，避免该工具模块额外导入离线数据集依赖 h5py。
        camera_id = sim.model.camera_name2id(camera_name)
        focal = 0.5 * height / np.tan(sim.model.cam_fovy[camera_id] * np.pi / 360)
        intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = sim.data.cam_xmat[camera_id].reshape(3, 3)
        extrinsic[:3, 3] = sim.data.cam_xpos[camera_id]
        extrinsic = extrinsic @ np.diag([1.0, -1.0, -1.0, 1.0])
        return {
            "camera_name": camera_name, "height": height, "width": width,
            "intrinsic_K": intrinsic.tolist(),
            "extrinsic_cam2world": extrinsic.tolist(),
            "depth_near": float(sim.model.vis.map.znear * sim.model.stat.extent),
            "depth_far": float(sim.model.vis.map.zfar * sim.model.stat.extent),
        }

    def _raw_obs(self) -> dict[str, object]:
        self._sim()
        raw = {
            key: np.asarray(self._observation[key]).copy()
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        }
        # 原工具只通过 *_pos 的键名构造物体清单；不传输物体真实坐标。
        raw.update({
            key: None for key in self._observation
            if key.endswith("_pos") and "robot0" not in key and "to_robot" not in key
        })
        for camera in ("agentview", "robot0_eye_in_hand"):
            frame = self._render_camera(camera, height=256, width=256, depth=True)
            raw[f"{camera}_image"] = frame["rgb"]
            raw[f"{camera}_depth"] = frame["depth"]
        return raw


def main() -> None:
    """启动独立基线环境服务，不改变普通 RoboMemArena 服务入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = RoboMemArenaEnvironmentServer(args.host, args.port)
    server._service = HarnessEnvironmentService()
    server.serve_forever()


if __name__ == "__main__":
    main()
