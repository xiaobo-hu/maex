"""Observer 的流式观察事件、采样窗口和结构化判断类型。"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Observation


@dataclass(frozen=True, slots=True)
class ObserverAssessment:
    """记录视觉观察与请求诊断；正常空观察的 error_stage/error_type 均为空。"""

    keyframe: int | None
    observation: str | None
    finish_reason: str | None = None
    error_stage: str | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """视觉流中的只读 observation；sequence 在单个 episode 内单调递增。"""

    sequence: int
    captured_at: float
    observation: Observation


@dataclass(frozen=True, slots=True)
class ObserverWindow:
    """一次 Observer 请求使用的锚点和未消费 observation 区间。"""

    events: tuple[ObservationEvent, ...]
    range_start: int
    range_end: int
    anchor_sequence: int | None = None
    gripper_transition_sequence: int | None = None

    @property
    def anchor_index(self) -> int | None:
        """返回请求窗口内 1-based 锚点位置。"""
        if self.anchor_sequence is None:
            return None
        return next(
            (
                index
                for index, event in enumerate(self.events, start=1)
                if event.sequence == self.anchor_sequence
            ),
            None,
        )
