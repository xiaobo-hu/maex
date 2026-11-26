"""本地视觉 Observer 的调用生命周期、窗口游标和单并发推理。"""

from __future__ import annotations

import asyncio
import time

from ..memory import (
    GripperEvent,
    VisualKeyframe,
    VisualMemory,
    build_visual_memory,
)
from .ports import ObservationSource, ObserverPort
from .sampler import (
    ObserverSamplingOptions,
    sample_observer_window,
)
from .types import ObservationEvent, ObserverAssessment, ObserverWindow


class VisualObserver:
    """只读消费 observation 流，并按当前 Agent 子任务异步记录视觉事实。

    单并发推理期间持续收集新帧，返回后用未消费区间构造下一窗口，不按 chunk 排队。
    end_vla_call 等待进行中的评估并补齐尾部，使过程观察进入同轮 feedback。
    """

    def __init__(
        self,
        observer: ObserverPort,
        source: ObservationSource,
        *,
        sampling: ObserverSamplingOptions | None = None,
    ) -> None:
        self._observer = observer
        self._source = source
        self._sampling = sampling or ObserverSamplingOptions()
        self._task = ""
        self._step_id = ""
        self._instruction = ""
        self._previous_observation = ""
        self._previous_instruction = ""
        self._instruction_id = 0
        self._cursor = 0
        self._anchor: ObservationEvent | None = None
        self._trace: list[dict[str, object]] = []
        self._request_sequence = 0
        self._control_signal = asyncio.Event()
        self._call_active = False
        self._collected_trace = 0
        self._call_start_sequence: int | None = None
        self._window_assessed = False
        self._assessment_lock = asyncio.Lock()
        self._stopping = False
        self._run_task: asyncio.Task[None] | None = None

    @property
    def trace(self) -> list[dict[str, object]]:
        """返回不包含图像 payload 的 Observer trace 快照。"""
        return [dict(event) for event in self._trace]

    @property
    def source(self) -> ObservationSource:
        """返回供并列执行哨兵复用的只读 observation 流。"""
        return self._source

    async def start(self, task: str) -> None:
        """开始一次 episode 的只读视觉消费。"""
        if self._run_task is not None:
            raise RuntimeError("VisualObserver is already running")
        self._task = task.strip()
        self._step_id = ""
        self._instruction = ""
        self._previous_observation = ""
        self._previous_instruction = ""
        self._instruction_id = 0
        self._trace = []
        self._request_sequence = 0
        self._call_active = False
        self._collected_trace = 0
        self._call_start_sequence = None
        self._window_assessed = False
        self._stopping = False
        self._control_signal = asyncio.Event()
        latest = self._source.latest()
        self._cursor = latest.sequence if latest is not None else 0
        self._anchor = latest
        self._run_task = asyncio.create_task(self._run())

    def set_instruction(
        self,
        instruction: str,
        *,
        step_id: str | None = None,
    ) -> int:
        """设置当前步骤；相同步骤保留短期上下文，切换时把晚到结果标为 stale。"""
        if self._run_task is None:
            raise RuntimeError("VisualObserver is not running")
        instruction = instruction.strip()
        step_id = step_id.strip() if step_id is not None else instruction
        if (
            self._instruction_id > 0
            and step_id == self._step_id
            and instruction == self._instruction
        ):
            return self._instruction_id
        if self._instruction_id > 0:
            self._previous_instruction = self._instruction
        self._step_id = step_id
        self._instruction = instruction
        self._instruction_id += 1
        self._window_assessed = False
        self._call_active = True
        latest = self._source.latest()
        self._cursor = latest.sequence if latest is not None else 0
        self._anchor = latest
        self._notify_control()
        return self._instruction_id

    def begin_vla_call(self, instruction: str, *, step_id: str) -> None:
        """从当前视觉流位置开始收集一次 VLA 调用的 Observer 输出。"""
        if self._run_task is None:
            raise RuntimeError("VisualObserver is not running")
        if self._call_start_sequence is not None:
            raise RuntimeError("VisualObserver already has an active VLA call")
        self.set_instruction(instruction, step_id=step_id)
        latest = self._source.latest()
        self._cursor = latest.sequence if latest is not None else 0
        self._anchor = latest
        self._call_start_sequence = latest.sequence if latest is not None else 0
        self._window_assessed = False
        self._call_active = True
        self._notify_control()

    async def end_vla_call(
        self,
        keyframe_refs: tuple[tuple[int, GripperEvent], ...] = (),
    ) -> VisualMemory:
        """等待进行中的推理，再评估截至调用结束的剩余窗口。"""
        if self._call_start_sequence is None:
            raise RuntimeError("VisualObserver has no active VLA call")
        latest = self._source.latest()
        end_sequence = latest.sequence if latest is not None else self._cursor
        self._call_active = False
        self._notify_control()
        # 锁内读取最新已消费游标，覆盖推理期间积累的帧，避免尾部重复或遗漏。
        async with self._assessment_lock:
            tail_events = self._visual_events_after(
                self._cursor,
                through_sequence=end_sequence,
            )
            if tail_events:
                window = self._sample_window(tail_events)
                event = await self._assess(
                    window,
                    self._instruction,
                    self._instruction_id,
                    self._step_id,
                    self._previous_observation,
                    self._previous_instruction,
                    is_tail=True,
                )
                event["stale"] = False
                self._trace.append(event)
                self._cursor = window.range_end
                self._anchor = next(
                    item
                    for item in reversed(window.events)
                    if item.sequence == window.range_end
                )

        trace_events = self._trace[self._collected_trace :]
        self._collected_trace = len(self._trace)
        source_events = {
            event.sequence: event
            for event in self._source.snapshot_after(self._call_start_sequence)
            if event.sequence <= end_sequence
        }
        keyframes: list[VisualKeyframe] = []
        for sequence, gripper_event in keyframe_refs:
            source_event = source_events.get(sequence)
            if source_event is None:
                continue
            # 关键帧描述不再单独推理；由窗口评估的 keyframe 字段产出。
            keyframes.append(
                VisualKeyframe(
                    sequence=sequence,
                    event=gripper_event,
                    rgb=source_event.observation.rgb,
                    wrist_rgb=source_event.observation.wrist_rgb,
                    observation=None,
                )
            )
        self._call_start_sequence = None
        observations: list[str | None] = []
        for event in trace_events:
            if event.get("stale") is not False and event.get("is_tail") is not True:
                continue
            observation = event.get("observation")
            observations.append(observation if isinstance(observation, str) else None)
        return build_visual_memory(observations, keyframes)

    async def stop(self) -> None:
        """停止视觉哨兵；已开始的模型请求会在退出前完成。"""
        task = self._run_task
        if task is None:
            return
        self._stopping = True
        self._notify_control()
        try:
            await task
        finally:
            self._run_task = None

    async def _run(self) -> None:
        while True:
            if (
                not self._instruction
                or not self._call_active
            ):
                if self._stopping:
                    return
                await self._control_signal.wait()
                continue

            new_events = self._visual_events_after(self._cursor)
            # 首窗攒够最小帧数；后续推理完成即可消费积累的帧，不再等一个 chunk。
            mid_call = bool(new_events) and (
                self._window_assessed
                or len(new_events) >= self._sampling.min_new_observations
            )
            stopping_drain = self._stopping and bool(new_events)
            if not (mid_call or stopping_drain):
                if self._stopping:
                    return
                await self._wait_for_input(self._cursor)
                continue

            instruction = self._instruction
            step_id = self._step_id
            previous_observation = self._previous_observation
            previous_instruction = self._previous_instruction
            instruction_id = self._instruction_id
            window = self._sample_window(new_events)
            async with self._assessment_lock:
                # end_vla_call 可能已经接管本次尾部；此时不再发起过期的中途请求。
                if not self._call_active:
                    continue
                event = await self._assess(
                    window,
                    instruction,
                    instruction_id,
                    step_id,
                    previous_observation,
                    previous_instruction,
                )
                event["stale"] = instruction_id != self._instruction_id
                self._trace.append(event)
                if instruction_id == self._instruction_id:
                    self._window_assessed = True
                    self._cursor = window.range_end
                    self._anchor = next(
                        item
                        for item in reversed(window.events)
                        if item.sequence == window.range_end
                    )
                if self._stopping and instruction_id == self._instruction_id:
                    remaining = self._visual_events_after(self._cursor)
                    if not remaining:
                        return

    def _sample_window(self, events: tuple[ObservationEvent, ...]) -> ObserverWindow:
        # 调用内相邻窗口共享边界图；游标仅推进到输入末帧，不推进到推理完成时刻。
        anchor = self._anchor
        if self._call_start_sequence is not None and anchor is not None and (
            anchor.observation.rgb is not None or anchor.observation.wrist_rgb is not None
        ):
            events = (anchor, *events)
        return sample_observer_window(anchor, events, self._sampling)

    def _visual_events_after(
        self,
        sequence: int,
        *,
        through_sequence: int | None = None,
    ) -> tuple[ObservationEvent, ...]:
        return tuple(
            event
            for event in self._source.snapshot_after(sequence)
            if through_sequence is None or event.sequence <= through_sequence
            if event.observation.rgb is not None or event.observation.wrist_rgb is not None
        )

    async def _assess(
        self,
        window: ObserverWindow,
        instruction: str,
        instruction_id: int,
        step_id: str,
        previous_observation: str,
        previous_instruction: str,
        *,
        is_tail: bool = False,
    ) -> dict[str, object]:
        self._request_sequence += 1
        started = time.perf_counter()
        gripper_transition_index = next(
            (
                index
                for index, event in enumerate(window.events, start=1)
                if event.sequence == window.gripper_transition_sequence
            ),
            None,
        )
        try:
            assessment = await self._observer.assess(
                self._task,
                instruction,
                [event.observation for event in window.events],
                anchor_index=window.anchor_index,
                gripper_transition_index=gripper_transition_index,
                previous_observation=previous_observation,
                previous_instruction=previous_instruction,
            )
        except Exception as error:
            assessment = ObserverAssessment(
                keyframe=None, observation=None,
                error_stage="request", error_type=type(error).__name__,
            )
        if instruction_id == self._instruction_id:
            if assessment.observation:
                self._previous_observation = assessment.observation
            if self._previous_instruction == previous_instruction:
                self._previous_instruction = ""
        keyframe_sequence = None
        if (
            assessment.keyframe is not None
            and assessment.keyframe == gripper_transition_index
            and 1 <= assessment.keyframe <= len(window.events)
        ):
            candidate = window.events[assessment.keyframe - 1]
            if candidate.sequence != window.anchor_sequence:
                keyframe_sequence = candidate.sequence
        return {
            "sequence": self._request_sequence,
            "instruction_id": instruction_id,
            "step_id": step_id,
            "instruction": instruction,
            "is_tail": is_tail,
            "range_start": window.range_start,
            "range_end": window.range_end,
            "input_sequences": [event.sequence for event in window.events],
            "anchor_sequence": window.anchor_sequence,
            "gripper_transition_sequence": window.gripper_transition_sequence,
            "keyframe_sequence": keyframe_sequence,
            "previous_observation": previous_observation,
            "observation": assessment.observation,
            "previous_instruction": previous_instruction,
            "finish_reason": assessment.finish_reason,
            "error_stage": assessment.error_stage,
            "error_type": assessment.error_type,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    async def _wait_for_input(self, sequence: int) -> None:
        source_wait = asyncio.create_task(self._source.wait_for_update(sequence))
        control_wait = asyncio.create_task(self._control_signal.wait())
        done, pending = await asyncio.wait(
            (source_wait, control_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()

    def _notify_control(self) -> None:
        previous_signal = self._control_signal
        self._control_signal = asyncio.Event()
        previous_signal.set()
