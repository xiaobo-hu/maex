"""RoboAgent 的观察、动作和 episode 结果类型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ai import ImageContent

EpisodeStatus = Literal[
    "success",
    "failure",
    "uncertain",
    "incomplete",
    "max_turns",
    "timeout",
    "cancelled",
    "safety_stop",
    "error",
]
ExecutionStatus = Literal["running", "success", "failure", "timeout", "cancelled"]
TaskStepStatus = Literal["pending", "running", "completed"]


def summarize_robot_state(robot_state: dict[str, object]) -> dict[str, object]:
    """保留原始状态，并为 Agent 补充确定性的夹爪命令语义和实际开度。"""
    summary = dict(robot_state)
    command = robot_state.get("gripper_command")
    if isinstance(command, int | float) and not isinstance(command, bool):
        value = float(command)
        if math.isfinite(value):
            summary["gripper_command_state"] = (
                "close" if value > 0 else "open" if value < 0 else "neutral"
            )

    qpos = robot_state.get("gripper_qpos")
    if isinstance(qpos, list | tuple) and len(qpos) == 2:
        try:
            left, right = (float(item) for item in qpos)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(left) and math.isfinite(right):
                summary["gripper_width_m"] = abs(left - right)
    return summary


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """已经编码为 base64 的图像，避免领域层依赖具体图像库。"""

    data: str
    mime_type: str = "image/png"

    def to_content(self) -> ImageContent:
        """转换成 ``ai`` Provider 可消费的图像内容。"""
        return ImageContent(data=self.data, mimeType=self.mime_type)


@dataclass(slots=True)
class Observation:
    """一次机器人观察；深度和任务成功信号均可缺省。"""

    rgb: ImageFrame | None = None
    wrist_rgb: ImageFrame | None = None
    depth: ImageFrame | None = None
    robot_state: dict[str, object] = field(default_factory=dict)
    task_success: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def llm_content(self) -> list[ImageContent]:
        """返回当前观察中的图像内容，顺序固定为主视角、腕部、深度。"""
        return [
            frame.to_content()
            for frame in (self.rgb, self.wrist_rgb, self.depth)
            if frame is not None
        ]

    def summary(self) -> dict[str, object]:
        """返回不包含图像 payload 的 JSON-safe 摘要。"""
        return {
            "robot_state": summarize_robot_state(self.robot_state),
            "task_success": self.task_success,
            "metadata": self.metadata,
            "has_rgb": self.rgb is not None,
            "has_wrist_rgb": self.wrist_rgb is not None,
            "has_depth": self.depth is not None,
        }


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """供机器人执行的一段动作，第一版使用二维数值列表保持序列化简单。"""

    actions: list[list[float]]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionFeedback:
    """一次动作块执行后的状态和最新观察。"""

    status: ExecutionStatus
    observation: Observation | None = None
    executed_steps: int = 0
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskStep:
    """Agent 规划的一项可由 VLA 执行且可验证的语义子任务。"""

    id: str
    instruction: str
    status: TaskStepStatus

    def summary(self) -> dict[str, object]:
        """返回供工具结果和 episode trace 使用的 JSON-safe 表示。"""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """当前 episode 中由上层 Agent 显式创建和推进的有序任务计划。"""

    steps: list[TaskStep]

    def summary(self) -> dict[str, object]:
        """返回不包含执行日志或视觉证据的计划快照。"""
        return {"steps": [step.summary() for step in self.steps]}


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """RoboAgent 一次任务运行的最终结果。"""

    status: EpisodeStatus
    summary: str
    turns: int
    tool_calls: int
    details: dict[str, object] = field(default_factory=dict)
