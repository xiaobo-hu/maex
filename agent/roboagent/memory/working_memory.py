"""组合视觉哨兵与执行哨兵产出的单次 VLA 多模态工作记忆。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ai import ImageContent, TextContent

from ..types import ImageFrame, Observation, summarize_robot_state

GripperEvent = Literal["gripper_open", "gripper_close"]
_ROBOT_STATE_KEYS = (
    "eef_position",
    "eef_axis_angle",
    "gripper_qpos",
    "gripper_command",
    "gripper_command_state",
    "gripper_width_m",
)


def _robot_state(observation: Observation) -> dict[str, object]:
    state = summarize_robot_state(observation.robot_state)
    return {
        key: state[key]
        for key in _ROBOT_STATE_KEYS
        if key in state
    }


@dataclass(frozen=True, slots=True)
class ExecutionState:
    """一个 observation sequence 上可回传 Agent 的机器人状态。"""

    sequence: int | None
    robot_state: dict[str, object]

    def summary(self) -> dict[str, object]:
        """返回不含图像 payload 的执行状态。"""
        return {"sequence": self.sequence, "robot_state": self.robot_state}


def build_execution_state(
    sequence: int | None,
    observation: Observation | None,
) -> ExecutionState | None:
    """从 observation 提取工作记忆需要的 EEF 与夹爪状态。"""
    if observation is None:
        return None
    return ExecutionState(sequence=sequence, robot_state=_robot_state(observation))


@dataclass(frozen=True, slots=True)
class GripperEventRecord:
    """执行哨兵确定的夹爪事件与对应机器人状态。"""

    sequence: int
    command_sequence: int
    event: GripperEvent
    command: float
    robot_state: dict[str, object]

    @property
    def id(self) -> str:
        """返回单个 episode 内稳定的关键帧标识。"""
        return f"gripper-keyframe-{self.sequence}"

    def summary(self) -> dict[str, object]:
        """返回关键帧的确定性动作与机器人状态。"""
        return {
            "id": self.id,
            "sequence": self.sequence,
            "command_sequence": self.command_sequence,
            "event": self.event,
            "command": self.command,
            "robot_state": self.robot_state,
        }


@dataclass(frozen=True, slots=True)
class VisualKeyframe:
    """视觉哨兵保存的关键帧文本与双视角图像。"""

    sequence: int
    event: GripperEvent
    rgb: ImageFrame | None = None
    wrist_rgb: ImageFrame | None = None
    observation: str | None = None

    @property
    def id(self) -> str:
        """返回与对应执行事件一致的关键帧标识。"""
        return f"gripper-keyframe-{self.sequence}"

    def summary(self) -> dict[str, object]:
        """返回关键帧的视觉事实和可用视角。"""
        return {
            "id": self.id,
            "sequence": self.sequence,
            "event": self.event,
            "observation": self.observation,
            "views": self._views(),
        }

    def llm_content(self) -> list[TextContent | ImageContent]:
        """按主视角、腕部视角顺序生成带标签的关键帧内容。"""
        views = self._views()
        content: list[TextContent | ImageContent] = [
            TextContent(
                text=(
                    f"Keyframe {self.id}: {self.event} at sequence {self.sequence}. "
                    f"Images follow in this order: {', '.join(views) or 'none'}."
                )
            )
        ]
        content.extend(
            frame.to_content() for frame in (self.rgb, self.wrist_rgb) if frame is not None
        )
        return content

    def _views(self) -> list[str]:
        return [
            name
            for name, frame in (
                ("external", self.rgb),
                ("wrist", self.wrist_rgb),
            )
            if frame is not None
        ]


@dataclass(frozen=True, slots=True)
class VisualMemory:
    """视觉哨兵在单次 VLA 调用内记录的文本与关键帧。"""

    observations: tuple[str, ...] = ()
    keyframes: tuple[VisualKeyframe, ...] = ()

    def summary(self) -> dict[str, object]:
        """返回视觉文本与关键帧元数据。"""
        return {
            "observations": list(self.observations),
            "keyframes": [keyframe.summary() for keyframe in self.keyframes],
        }

    def llm_content(self) -> list[TextContent | ImageContent]:
        """返回按 sequence 排列且带标签的关键帧图像内容。"""
        return [content for keyframe in self.keyframes for content in keyframe.llm_content()]


def build_visual_memory(
    observations: Iterable[str | None],
    keyframes: Iterable[VisualKeyframe] = (),
) -> VisualMemory:
    """精确去重文本并按 sequence 保留全部物理关键帧。"""
    memory: list[str] = []
    for observation in observations:
        if observation is None:
            continue
        text = observation.strip()
        if not text:
            continue
        if text in memory:
            memory.remove(text)
        memory.append(text)
    return VisualMemory(
        observations=tuple(memory),
        keyframes=tuple(sorted(keyframes, key=lambda item: item.sequence)),
    )


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    """执行哨兵记录的起点、夹爪事件、终点和调用边界。"""

    start: ExecutionState | None = None
    gripper_events: tuple[GripperEventRecord, ...] = ()
    end: ExecutionState | None = None
    chunks_used: int = 0
    stop_reason: str | None = None

    def summary(self) -> dict[str, object]:
        """返回不含图像和失败推断的确定性执行事实。"""
        return {
            "start": self.start.summary() if self.start is not None else None,
            "gripper_events": [event.summary() for event in self.gripper_events],
            "end": self.end.summary() if self.end is not None else None,
            "chunks_used": self.chunks_used,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class VlaWorkingMemory:
    """交给 Agent 的单次 VLA 视觉与机器人执行工作记忆。"""

    visual: VisualMemory = VisualMemory()
    execution: ExecutionDiagnostics = ExecutionDiagnostics()
    start_observation: Observation | None = None

    def summary(self) -> dict[str, object]:
        """返回可直接写入工具文本和 trace 的工作记忆摘要。"""
        return {
            "visual": self.visual.summary(),
            "execution": self.execution.summary(),
        }

    def llm_content(self) -> list[TextContent | ImageContent]:
        """返回带标签的调用起点；关键帧图像暂只保留在记录中。"""
        content: list[TextContent | ImageContent] = []
        observation = self.start_observation
        if observation is not None and observation.llm_content():
            views = [
                name
                for name, frame in (
                    ("external", observation.rgb),
                    ("wrist", observation.wrist_rgb),
                    ("depth", observation.depth),
                )
                if frame is not None
            ]
            sequence = self.execution.start.sequence if self.execution.start else None
            content.append(
                TextContent(
                    text=(
                        f"VLA call start observation at sequence {sequence}. "
                        f"Images follow in this order: {', '.join(views)}."
                    )
                )
            )
            content.extend(observation.llm_content())
        return content
