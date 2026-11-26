"""进程内只读 observation 流：执行端发布，Observer 端订阅。"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from ..types import Observation
from .ports import ObservationSink, ObservationSource
from .types import ObservationEvent


class _ObservationChannel:
    """保存有界事件历史，并用一次性信号唤醒异步订阅者。"""

    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self.events: deque[ObservationEvent] = deque(maxlen=max_events)
        self.sequence = 0
        self.signal = asyncio.Event()

    def publish(self, observation: Observation) -> ObservationEvent:
        self.sequence += 1
        event = ObservationEvent(self.sequence, time.monotonic(), observation)
        self.events.append(event)
        previous_signal = self.signal
        self.signal = asyncio.Event()
        previous_signal.set()
        return event

    def snapshot_after(self, sequence: int) -> tuple[ObservationEvent, ...]:
        return tuple(event for event in self.events if event.sequence > sequence)


class _ObservationSink:
    def __init__(self, channel: _ObservationChannel) -> None:
        self._channel = channel

    def publish(self, observation: Observation) -> ObservationEvent:
        return self._channel.publish(observation)


class _ObservationSource:
    def __init__(self, channel: _ObservationChannel) -> None:
        self._channel = channel

    def latest(self) -> ObservationEvent | None:
        return self._channel.events[-1] if self._channel.events else None

    def snapshot_after(self, sequence: int) -> tuple[ObservationEvent, ...]:
        return self._channel.snapshot_after(sequence)

    async def wait_for_update(self, sequence: int) -> tuple[ObservationEvent, ...]:
        while True:
            events = self.snapshot_after(sequence)
            if events:
                return events
            signal = self._channel.signal
            events = self.snapshot_after(sequence)
            if events:
                return events
            await signal.wait()


def create_observation_stream(
    *,
    max_events: int = 2048,
) -> tuple[ObservationSink, ObservationSource]:
    """创建能力分离的发布端和只读订阅端。"""
    channel = _ObservationChannel(max_events)
    return _ObservationSink(channel), _ObservationSource(channel)
