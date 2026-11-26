"""OpenPI ``BasePolicy`` 的轻量异步适配，不绑定具体环境输入格式。"""

from __future__ import annotations

import asyncio
from typing import Protocol

from ..ports import VlaInputBuilder
from ..types import ActionChunk, Observation


class _Policy(Protocol):
    def infer(self, observation: dict[str, object]) -> dict[str, object]: ...


class OpenPiPolicyAdapter:
    """把同步 OpenPI websocket/client policy 转为 RoboAgent VLA 端口。"""

    def __init__(
        self,
        policy: _Policy,
        input_builder: VlaInputBuilder,
        *,
        max_actions: int | None = None,
    ) -> None:
        if max_actions is not None and max_actions < 1:
            raise ValueError("max_actions must be at least 1")
        self._policy = policy
        self._input_builder = input_builder
        self._max_actions = max_actions

    async def infer(
        self,
        instruction: str,
        observation: Observation,
        cancel_event: asyncio.Event | None = None,
    ) -> ActionChunk:
        """编码观察、执行一次 policy inference 并校验动作块。"""
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        payload = self._input_builder(instruction, observation)
        # OpenPI 的 websocket client 是同步协议；第一阶段不引入额外线程池
        # 生命周期，避免在 Agent 进程退出时遗留非 daemon worker。
        result = self._policy.infer(payload)
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        raw_actions = result.get("actions")
        if hasattr(raw_actions, "tolist"):
            raw_actions = raw_actions.tolist()
        if not isinstance(raw_actions, list) or any(not isinstance(row, list) for row in raw_actions):
            raise ValueError("OpenPI policy result must contain a 2-D 'actions' list")
        try:
            actions = [[float(value) for value in row] for row in raw_actions]
        except (TypeError, ValueError) as error:
            raise ValueError("OpenPI actions must contain numeric values") from error
        if self._max_actions is not None:
            actions = actions[: self._max_actions]
        metadata = {key: value for key, value in result.items() if key != "actions"}
        return ActionChunk(actions=actions, metadata=metadata)
