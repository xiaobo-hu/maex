"""从 Observer 尚未消费的 observation 区间选择有限视觉 timestep。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..memory import GripperEvent, GripperEventRecord, build_execution_state
from .types import ObservationEvent, ObserverWindow


@dataclass(frozen=True, slots=True)
class ObserverSamplingOptions:
    """Observer 窗口的输入预算和触发阈值。"""

    max_timesteps: int = 5
    min_new_observations: int = 10
    gripper_transition_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.max_timesteps < 2 or self.min_new_observations < 1:
            raise ValueError("Observer max_timesteps must be at least 2")
        if (
            self.gripper_transition_threshold is not None
            and self.gripper_transition_threshold < 0
        ):
            raise ValueError("gripper_transition_threshold must be non-negative")


def sample_observer_window(
    anchor: ObservationEvent | None,
    new_events: tuple[ObservationEvent, ...],
    options: ObserverSamplingOptions,
) -> ObserverWindow:
    """构造首尾窗口:START 帧 + END 帧 + 检测到的夹爪转变帧,不含 anchor 图。

    历史连续性只通过上一条文本 observation 传递,因此窗口内不再携带上一窗口的
    anchor 帧;窗口始终从区间首帧开始、以末帧结束,夹爪转变帧只在检测到且预算
    允许时插入。
    """
    visual_events = tuple(
        event
        for event in new_events
        if event.observation.rgb is not None or event.observation.wrist_rgb is not None
    )
    if not visual_events:
        raise ValueError("Observer window requires at least one visual observation")

    transition_index = _gripper_transition_index(
        anchor,
        visual_events,
        options.gripper_transition_threshold,
    )
    selected_indices = {0, len(visual_events) - 1}
    if transition_index is not None and len(selected_indices) < options.max_timesteps:
        selected_indices.add(transition_index)
    # 预算不足时按 END > START > 夹爪 的优先级裁剪。
    while len(selected_indices) > options.max_timesteps:
        if transition_index in selected_indices:
            selected_indices.discard(transition_index)
            continue
        selected_indices.discard(0)

    selected = tuple(visual_events[index] for index in sorted(selected_indices))
    selected_transition_index = (
        transition_index if transition_index in selected_indices else None
    )
    return ObserverWindow(
        events=selected,
        range_start=visual_events[0].sequence,
        range_end=visual_events[-1].sequence,
        anchor_sequence=None,
        gripper_transition_sequence=(
            visual_events[selected_transition_index].sequence
            if selected_transition_index is not None
            else None
        ),
    )


def _gripper_transition_index(
    anchor: ObservationEvent | None,
    events: tuple[ObservationEvent, ...],
    threshold: float | None,
) -> int | None:
    if threshold is None:
        return None
    keyframes = detect_gripper_keyframes(anchor, events, threshold=threshold)
    if keyframes:
        sequence = keyframes[-1].sequence
        return next(
            index for index, event in enumerate(events) if event.sequence == sequence
        )
    if _has_gripper_command(anchor, events):
        return None

    return _qpos_transition_index(anchor, events, threshold)


def detect_gripper_keyframes(
    anchor: ObservationEvent | None,
    events: tuple[ObservationEvent, ...],
    *,
    threshold: float | None,
) -> tuple[GripperEventRecord, ...]:
    """按夹爪命令区间选择累计 qpos 变化最大的实际 observation。"""
    if threshold is None:
        return ()

    keyframes: list[GripperEventRecord] = []
    active_sign = _gripper_command_sign(anchor)
    active_command = _gripper_command(anchor)
    command_sequence = anchor.sequence if anchor is not None and active_sign else None
    baseline = _gripper_qpos(anchor)
    previous_qpos = baseline
    candidate: ObservationEvent | None = None
    best_delta = threshold

    def finish_run() -> None:
        if (
            active_sign is None
            or active_command is None
            or command_sequence is None
            or candidate is None
        ):
            return
        event: GripperEvent = "gripper_close" if active_sign > 0 else "gripper_open"
        state = build_execution_state(candidate.sequence, candidate.observation)
        if state is None:
            return
        keyframes.append(
            GripperEventRecord(
                sequence=candidate.sequence,
                command_sequence=command_sequence,
                event=event,
                command=active_command,
                robot_state=state.robot_state,
            )
        )

    for event in events:
        command = _gripper_command(event)
        sign = _command_sign(command)
        current_qpos = _gripper_qpos(event)
        if sign is not None and sign != active_sign:
            finish_run()
            active_sign = sign
            active_command = command
            command_sequence = event.sequence
            baseline = previous_qpos if previous_qpos is not None else current_qpos
            candidate = None
            best_delta = threshold

        if (
            active_sign is not None
            and baseline is not None
            and current_qpos is not None
            and len(baseline) == len(current_qpos)
        ):
            delta = _qpos_distance(baseline, current_qpos)
            if delta > best_delta:
                best_delta = delta
                candidate = event
        if current_qpos is not None:
            previous_qpos = current_qpos

    finish_run()
    return tuple(keyframes)


def _qpos_transition_index(
    anchor: ObservationEvent | None,
    events: tuple[ObservationEvent, ...],
    threshold: float,
) -> int | None:
    previous = _gripper_qpos(anchor) if anchor is not None else None
    best_index: int | None = None
    best_delta = threshold
    for index, event in enumerate(events):
        current = _gripper_qpos(event)
        if previous is not None and current is not None and len(previous) == len(current):
            delta = _qpos_distance(previous, current)
            if delta > best_delta:
                best_delta = delta
                best_index = index
        if current is not None:
            previous = current
    return best_index


def _qpos_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(
        sum((right_item - left_item) ** 2 for left_item, right_item in zip(left, right, strict=True))
    )


def _gripper_command_sign(event: ObservationEvent | None) -> int | None:
    return _command_sign(_gripper_command(event))


def _has_gripper_command(
    anchor: ObservationEvent | None,
    events: tuple[ObservationEvent, ...],
) -> bool:
    return any(_gripper_command(event) is not None for event in (anchor, *events))


def _command_sign(command: float | None) -> int | None:
    if command is None or command == 0:
        return None
    return 1 if command > 0 else -1


def _gripper_command(event: ObservationEvent | None) -> float | None:
    if event is None:
        return None
    value = event.observation.robot_state.get("gripper_command")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _gripper_qpos(event: ObservationEvent | None) -> tuple[float, ...] | None:
    if event is None:
        return None
    values = event.observation.robot_state.get("gripper_qpos")
    if not isinstance(values, list):
        return None
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
