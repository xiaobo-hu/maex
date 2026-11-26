"""只读记录单次 VLA 调用中的机器人状态与确定性夹爪事件。"""

from __future__ import annotations

from ..memory import ExecutionDiagnostics, build_execution_state
from ..types import Observation
from .ports import ObservationSource
from .sampler import detect_gripper_keyframes
from .types import ObservationEvent


class ExecutionDiagnosticsRunner:
    """按 VLA 调用边界从 observation 流整理确定性执行事实。"""

    def __init__(
        self,
        source: ObservationSource | None,
        *,
        gripper_transition_threshold: float,
    ) -> None:
        self._source = source
        self._threshold = gripper_transition_threshold
        self._anchor: ObservationEvent | None = None
        self._fallback_start: Observation | None = None
        self._active = False

    def begin_vla_call(self, observation: Observation | None) -> None:
        """从当前 observation sequence 开始记录一次 VLA 调用。"""
        if self._active:
            raise RuntimeError("ExecutionDiagnosticsRunner already has an active VLA call")
        self._anchor = self._source.latest() if self._source is not None else None
        self._fallback_start = observation
        self._active = True

    def end_vla_call(
        self,
        observation: Observation | None,
        *,
        chunks_used: int,
        stop_reason: str | None,
    ) -> ExecutionDiagnostics:
        """固定调用终点并返回不含任务语义推断的执行事实。"""
        if not self._active:
            raise RuntimeError("ExecutionDiagnosticsRunner has no active VLA call")
        anchor = self._anchor
        latest = self._source.latest() if self._source is not None else None
        start_sequence = anchor.sequence if anchor is not None else 0
        end_sequence = latest.sequence if latest is not None else start_sequence
        events = (
            tuple(
                event
                for event in self._source.snapshot_after(start_sequence)
                if event.sequence <= end_sequence
            )
            if self._source is not None
            else ()
        )
        keyframes = detect_gripper_keyframes(
            anchor,
            events,
            threshold=self._threshold,
        )
        start_observation = (
            anchor.observation if anchor is not None else self._fallback_start
        )
        end_observation = latest.observation if latest is not None else observation
        diagnostics = ExecutionDiagnostics(
            start=build_execution_state(
                anchor.sequence if anchor is not None else None,
                start_observation,
            ),
            gripper_events=keyframes,
            end=build_execution_state(
                latest.sequence if latest is not None else None,
                end_observation,
            ),
            chunks_used=chunks_used,
            stop_reason=stop_reason,
        )
        self._anchor = None
        self._fallback_start = None
        self._active = False
        return diagnostics
