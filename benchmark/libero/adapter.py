"""Adapt a LIBERO environment to RoboAgent's serialized RobotPort."""

from __future__ import annotations

import asyncio
from typing import Protocol

from roboagent import ActionChunk, ExecutionFeedback, Observation
from roboagent.observer import ObservationSink
from roboagent.types import ExecutionStatus

from .environment import LiberoEnv, LiberoStep


class ObservationBuilder(Protocol):
    """将 LIBERO 原始 observation 转成 RoboAgent observation。"""

    def __call__(self, raw: object, *, success: bool | None = None) -> Observation: ...


class LiberoRobotPort:
    """把 LIBERO 的同步 ``reset/step`` 映射为串行 RobotPort。"""

    def __init__(
        self,
        env: LiberoEnv,
        build_observation: ObservationBuilder,
        observation_sink: ObservationSink | None = None,
    ) -> None:
        self._env = env
        self._build_observation = build_observation
        self._observation_sink = observation_sink
        self._raw_observation: object | None = None
        self._success = False
        self._done = False

    async def reset(self, seed: int | None = None) -> Observation:
        """重置 LIBERO episode。"""
        self._raw_observation = self._env.reset(seed=seed)
        self._success = False
        self._done = False
        observation = self._build_observation(self._raw_observation, success=False)
        self._publish(observation)
        return observation

    async def observe(self, cancel_event: asyncio.Event | None = None) -> Observation:
        """返回最近一次同步环境状态，不额外推进仿真。"""
        self._check_cancelled(cancel_event)
        if self._raw_observation is None:
            self._raw_observation = self._env.reset()
        return self._build_observation(self._raw_observation, success=self._success)

    async def execute_action_chunk(
        self,
        actions: ActionChunk,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionFeedback:
        """逐条执行动作，禁止对物理环境使用并发或 gather。"""
        if self._done:
            return ExecutionFeedback("success" if self._success else "failure", await self.observe(), 0)

        executed = 0
        last_step: LiberoStep | None = None
        observation: Observation | None = None
        for action in actions.actions:
            self._check_cancelled(cancel_event)
            last_step = self._env.step(action)
            self._raw_observation = last_step.observation
            self._success = bool(last_step.success)
            self._done = bool(last_step.done)
            executed += 1
            observation = self._build_observation(
                self._raw_observation,
                success=self._success,
            )
            if len(action) > 6:
                observation.robot_state["gripper_command"] = float(action[6])
            self._publish(observation)
            # 发布端不等待 Observer，但让已经在运行的网络推理继续推进。
            await asyncio.sleep(0)
            if self._done:
                break

        if observation is None:
            observation = self._build_observation(self._raw_observation, success=self._success)
        status: ExecutionStatus = "success" if self._success else "failure" if self._done else "running"
        details = dict(last_step.info or {}) if last_step is not None else {}
        return ExecutionFeedback(status, observation, executed, details)

    def _publish(self, observation: Observation) -> None:
        if self._observation_sink is not None:
            self._observation_sink.publish(observation)

    @staticmethod
    def _check_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
