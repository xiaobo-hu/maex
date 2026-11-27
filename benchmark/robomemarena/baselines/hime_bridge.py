"""基线组合层：复用原版 HiMe 调度与记忆，仅适配双相机和任务生命周期。"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from benchmark.robomemarena.preprocessing import build_policy_input


class HiMePlanError(RuntimeError):
    """HiMe 未生成可执行的计划。"""


class HiMeSession:
    """一个 episode 独享的 HiMe；不注入 RoboAgent 的计划或观察逻辑。"""

    def __init__(
        self, root: Path, prompt_dir: Path, planner_client, sentry_client, encoder,
        *, window_size: int = 8, fallback_images: int = 50,
    ) -> None:
        sys.path.insert(0, str(root))
        from agent.multitag_planner import PlannerAgent
        from agent.observer import ObserverAgent
        from client.observer_vlm import ObserverVLM
        from client.planner_vlm import PlannerVLM
        from memory.multitag_recorder import MultiTagMemory
        from server.image_utils import RobotImageInput
        from server.schema import TaskConfig
        from server.task_manager import ServerTaskManager

        self.encoder = encoder
        self.memory = MultiTagMemory(encoder)
        self.manager = ServerTaskManager()
        self.image_input = RobotImageInput
        self.config = TaskConfig(
            observer_window_size=window_size,
            planner_round_fallback_max_images=fallback_images,
            planner_execution_mode="sync",
        )
        self.manager.set_agents(
            PlannerAgent(
                PlannerVLM(planner_client), self.memory,
                prompt_name=str(prompt_dir / "robomemarena"),
            ),
            ObserverAgent(
                ObserverVLM(sentry_client), prompt_name=str(prompt_dir / "robomemarena_obs"),
            ),
        )
        self.state = None

    def start(self, instruction: str, raw: dict):
        """从初始相机帧启动原有初始规划，记忆在构造时已隔离。"""
        main, wrist = self.frames(raw)
        self.state = self.manager.create_task(
            global_instruction=instruction, config=self.config,
            initial_robot_input=self.image_input(image=main, waist_image=wrist),
        )
        self._check_state()
        return self.state

    def update(self, frames: list[tuple[Image.Image, Image.Image]]):
        """提交执行后的帧，由原 Sentry 与 fallback 决定是否重规划。"""
        self.state = self.manager.add_step_and_maybe_refine_robot(
            self.state.task_id,
            self.image_input(
                image=[main for main, _ in frames],
                waist_image=[wrist for _, wrist in frames],
            ),
        )
        self._check_state()
        return self.state

    def close(self) -> None:
        """释放原任务管理器的后台线程资源。"""
        self.manager._planner_executor.shutdown(wait=True)

    @staticmethod
    def frames(raw: dict) -> tuple[Image.Image, Image.Image]:
        """统一使用 benchmark 预处理，并保持左腕部、右外部的拼图约定。"""
        payload = build_policy_input(raw, "")
        return (
            Image.fromarray(payload["observation/image"]),
            Image.fromarray(payload["observation/wrist_image"]),
        )

    def _check_state(self) -> None:
        if self.encoder.failed:
            raise RuntimeError("Embedding failed inside HiMe; see embedding_calls.jsonl")
        # 原版把空计划解释为 done；记录为接口失败，不能算成功或额外修正计划。
        if not self.state.plan_list.strip():
            raise HiMePlanError("HiMe returned an empty plan")
        if self.state.is_done:
            from extractor import is_plan_done

            if not is_plan_done(self.state.plan_list):
                raise HiMePlanError("HiMe stopped without a completed plan")
