"""客户端适配层：将 RoboMemArena/OpenPI 接到未修改的 HarnessVLA 原语。"""

from __future__ import annotations

import math
import numpy as np

from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.preprocessing import build_policy_input


class HarnessEnvBridge:
    """提供 HarnessVLA 的单环境契约，物理步与成功条件均由 benchmark 管理。"""

    def __init__(self, remote: RemoteRoboMemArenaEnv, *, allow_reset: bool = False) -> None:
        self.remote = remote
        self.return_all_frames = False
        self.episode_terminated = False
        self.episode_truncated = False
        self._reset_done = False
        self._allow_reset = allow_reset

    @property
    def terminated(self) -> bool:
        """对应官方新版环境契约的成功标志。"""
        return self.episode_terminated

    @property
    def truncated(self) -> bool:
        """对应官方新版环境契约的失败或预算终止标志。"""
        return self.episode_truncated

    def reset(self):
        """评测只初始化一次；探索由官方 Toolkit 管理重试预算。"""
        if self._reset_done and not self._allow_reset:
            raise RuntimeError("Harness baseline evaluation allows only one episode")
        raw = self.remote.reset()
        self._reset_done = True
        self.episode_terminated = False
        self.episode_truncated = False
        return self._observation(raw), {}

    def step(self, action):
        """推进一个真实环境步，区分官方成功与失败终止。"""
        if self.episode_terminated or self.episode_truncated:
            raise RuntimeError("The evaluation episode has ended")
        result = self.remote.step(action)
        self._record_done(result)
        return (
            self._observation(result.observation), 0.0,
            self.episode_terminated, self.episode_truncated, {},
        )

    def chunk_step(self, actions, *, return_all_frames=None):
        """逐步执行动作块，在 benchmark 终态处丢弃剩余动作。"""
        observations, rewards, terminated, truncated = [], [], [], []
        for action in actions:
            obs, reward, term, trunc, _info = self.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                break
        if not observations:
            raise ValueError("An action chunk cannot be empty")
        all_frames = self.return_all_frames if return_all_frames is None else return_all_frames
        return (
            observations if all_frames else observations[-1], np.asarray(rewards),
            np.asarray(terminated), np.asarray(truncated), {},
        )

    def settle(self) -> None:
        """在规划器报告成功后，执行 benchmark 原有终态确认窗口。"""
        if not (self.episode_terminated or self.episode_truncated):
            self._record_done(self.remote.settle())

    def raw_obs(self):
        """读取双视角 RGB-D、本体状态与物体名称，不包含真实物体坐标。"""
        return self.remote._call("harness_raw_obs")["raw"]

    def render_camera(
        self, camera_name="agentview", height=1024, width=1024, depth=False,
    ):
        """读取相机帧；渲染不会推进物理时间。"""
        result = self.remote._call("harness_render_camera", {
            "camera_name": camera_name, "height": height, "width": width, "depth": depth,
        })
        return (result["rgb"], result["depth"]) if depth else result["rgb"]

    def get_camera_meta(self, camera_name="agentview", height=256, width=256):
        """提供与相机分辨率匹配的内外参。"""
        return self.remote._call("harness_camera_meta", {
            "camera_name": camera_name, "height": height, "width": width,
        })["meta"]

    def get_task_language(self):
        """返回官方任务指令，不提供隐藏阶段或 reference plan。"""
        return self.remote.task_description

    def _record_done(self, result) -> None:
        self.episode_terminated = bool(result.done and result.success)
        self.episode_truncated = bool(result.done and not result.success)

    def _observation(self, raw):
        payload = build_policy_input(raw, self.remote.task_description)
        return {
            "main_images": payload["observation/image"],
            "wrist_images": payload["observation/wrist_image"],
            "states": payload["observation/state"],
            "task_descriptions": self.remote.task_description,
        }


class OfficialHarnessEnvBridge(HarnessEnvBridge):
    """按官方 LIBERO 权重的观测约定适配；不改变旧基线的预处理。"""

    def _observation(self, raw):
        # 与 RLinf libero/utils.py 一致：双轴翻转，不做额外缩放或四元数符号归一化。
        quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64).copy()
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = math.sqrt(1.0 - quat[3] ** 2)
        rotation = np.zeros(3) if math.isclose(den, 0.0) else quat[:3] * 2.0 * math.acos(quat[3]) / den
        return {
            "main_images": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
            "wrist_images": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
            "states": np.concatenate((raw["robot0_eef_pos"], rotation, raw["robot0_gripper_qpos"])),
            "task_descriptions": self.remote.task_description,
        }


class HarnessPolicyBridge:
    """将原语的 predict_action_batch 调用转为现有 OpenPI 推理请求。"""

    def __init__(self, policy, *, action_chunk_size: int = 20) -> None:
        self.policy = policy
        self.action_chunk_size = action_chunk_size

    def predict_action_batch(self, env_obs, mode="eval"):
        """使用 benchmark 预处理后的双视角和当前子指令执行冻结 VLA。"""
        main, wrist = env_obs["main_images"], env_obs["wrist_images"]
        result = self.policy.infer({
            "observation/image": main, "observation/wrist_image": wrist,
            "base_0_rgb": main, "left_wrist_0_rgb": np.zeros_like(wrist),
            "right_wrist_0_rgb": wrist, "observation/state": env_obs["states"],
            "prompt": env_obs["task_descriptions"],
        })
        actions = np.asarray(result["actions"])
        if (
            actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0
            or not np.isfinite(actions).all()
        ):
            raise ValueError("OpenPI must return non-empty finite actions with shape [N, 7]")
        # 与现有 benchmark 一致：chunk size 是上限，不补齐 checkpoint 的短输出。
        return actions[:self.action_chunk_size], {}

    def predict(self, env_obs, *, options=None):
        """适配新版 RPent 的 VLA 入口，保持同一预处理与动作截断规则。"""
        actions, _ = self.predict_action_batch(env_obs, mode=(options or {}).get("mode", "eval"))
        return actions
