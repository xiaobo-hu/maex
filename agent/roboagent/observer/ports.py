"""Observer 模型以及只读 observation 流所需的最小端口。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from ..types import Observation
from .types import ObservationEvent, ObserverAssessment


class ObserverPort(Protocol):
    """结合当前指令和上一条视觉记录描述窗口，不执行动作或修改环境。"""

    async def assess(
        self,
        task: str,
        instruction: str,
        observations: Sequence[Observation],
        cancel_event: asyncio.Event | None = None,
        *,
        anchor_index: int | None = None,
        gripper_transition_index: int | None = None,
        previous_observation: str = "",
        previous_instruction: str = "",
    ) -> ObserverAssessment: ...


class ObservationSink(Protocol):
    """由机器人适配器持有的非阻塞 observation 发布端。"""

    def publish(self, observation: Observation) -> ObservationEvent: ...


class ObservationSource(Protocol):
    """视觉和执行哨兵共享的只读 observation 订阅端。"""

    def latest(self) -> ObservationEvent | None: ...

    def snapshot_after(self, sequence: int) -> tuple[ObservationEvent, ...]: ...

    async def wait_for_update(self, sequence: int) -> tuple[ObservationEvent, ...]: ...
