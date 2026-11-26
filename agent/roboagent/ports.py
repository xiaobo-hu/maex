"""机器人和 VLA 适配器的最小异步端口。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from .types import ActionChunk, ExecutionFeedback, Observation


class RobotPort(Protocol):
    """提供观察和动作块执行，不包含高层规划。"""

    async def observe(self, cancel_event: asyncio.Event | None = None) -> Observation: ...

    async def execute_action_chunk(
        self,
        actions: ActionChunk,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionFeedback: ...

    async def return_to_home(
        self,
        gripper_command: float,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionFeedback: ...


class EpisodeFinalizerPort(Protocol):
    """在 Agent 请求成功后推进环境自身的终态验证流程。"""

    async def finalize_episode(
        self,
        cancel_event: asyncio.Event | None = None,
    ) -> Observation: ...


VlaInputBuilder = Callable[[str, Observation], dict[str, object]]


class VlaPolicyPort(Protocol):
    """根据语言子任务和观察生成一个动作块。"""

    async def infer(
        self,
        instruction: str,
        observation: Observation,
        cancel_event: asyncio.Event | None = None,
    ) -> ActionChunk: ...
