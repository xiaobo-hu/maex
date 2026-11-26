"""Adapt RoboMemArena to RoboAgent without exposing evaluator internals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from roboagent import ActionChunk, EePoseAdjustmentConfig, ExecutionFeedback, Observation
from roboagent.observer import ObservationSink
from roboagent.types import ExecutionStatus
from roboagent.utils.ee_pose import pose_from_observation, workspace_violation_distance

from ..environment import RoboMemArenaEnv


class ObservationBuilder(Protocol):
    """Build an Agent-facing observation from raw simulator values."""

    def __call__(self, raw: object, *, success: bool | None = None) -> Observation: ...


class RoboMemArenaRobotPort:
    """Serialize action chunks while keeping CSR and stage metrics out of Agent I/O."""

    def __init__(
        self,
        env: RoboMemArenaEnv,
        build_observation: ObservationBuilder,
        observation_sink: ObservationSink | None = None,
        observation_recorder: Callable[[Observation], None] | None = None,
        home_workspace: EePoseAdjustmentConfig | None = None,
    ) -> None:
        self._env = env
        self._build_observation = build_observation
        self._observation_sink = observation_sink
        self._observation_recorder = observation_recorder
        self._home_workspace = home_workspace
        self._raw_observation: object | None = None
        self._success = False
        self._done = False

    async def reset(self, seed: int | None = None) -> Observation:
        """Reset an episode and return its post-warmup observation."""
        self._raw_observation = self._env.reset(seed=seed)
        self._success = False
        self._done = False
        observation = self._build_observation(self._raw_observation, success=False)
        self._publish(observation)
        return observation

    async def observe(self, cancel_event: asyncio.Event | None = None) -> Observation:
        """Read the current observation without advancing the simulator."""
        self._check_cancelled(cancel_event)
        if self._raw_observation is None:
            self._raw_observation = self._env.observe()
        return self._build_observation(self._raw_observation, success=self._success)

    async def execute_action_chunk(
        self,
        actions: ActionChunk,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionFeedback:
        """Execute finite 7-D actions strictly one at a time."""
        if self._done:
            status: ExecutionStatus = "success" if self._success else "failure"
            return ExecutionFeedback(status, await self.observe(), 0, {})

        executed = 0
        observation: Observation | None = None
        for action in actions.actions:
            self._check_cancelled(cancel_event)
            step = self._env.step(action)
            self._raw_observation = step.observation
            self._success = bool(step.success)
            self._done = bool(step.done)
            executed += 1
            observation = self._build_observation(
                self._raw_observation,
                success=self._success,
            )
            observation.robot_state["gripper_command"] = float(action[6])
            self._publish(observation)
            # Observer 只读并发消费；物理 step 本身仍严格串行。
            await asyncio.sleep(0)
            if self._done:
                break

        if observation is None:
            observation = self._build_observation(self._raw_observation, success=self._success)
        status = "success" if self._success else "failure" if self._done else "running"
        # 评分是独立 evaluation 通道；details 不携带 CSR、阶段名或中间信号。
        return ExecutionFeedback(status, observation, executed, {})

    async def return_to_home(
        self,
        gripper_command: float,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionFeedback:
        """Execute the environment-owned joint-space home trajectory."""
        self._check_cancelled(cancel_event)
        observation = self._build_observation(self._raw_observation, success=self._success)
        previous_workspace_distance = self._home_workspace_distance(observation)
        steps = getattr(self._env, "iter_return_to_home", None)
        trajectory = (
            steps(gripper_command)
            if callable(steps)
            else iter((self._env.return_to_home(gripper_command),))
        )
        details: dict[str, object] = {}
        executed = 0
        try:
            for step in trajectory:
                self._check_cancelled(cancel_event)
                self._raw_observation = step.observation
                self._success = bool(step.success)
                self._done = bool(step.done)
                observation = self._build_observation(
                    self._raw_observation,
                    success=self._success,
                )
                observation.robot_state["gripper_command"] = float(gripper_command)
                self._publish(observation)
                details = dict(step.details)
                executed = int(details.get("executed_steps", executed + 1))
                current_workspace_distance = self._home_workspace_distance(observation)
                if (
                    current_workspace_distance > 0.0
                    and (
                        previous_workspace_distance == 0.0
                        or current_workspace_distance >= previous_workspace_distance
                    )
                ):
                    details.update(
                        {
                            "status": "out_of_workspace",
                            "home_pose_reached": False,
                            "reason": (
                                "The observed EEF crossed or failed to move toward the "
                                "configured workspace during return_to_home."
                            ),
                        }
                    )
                    break
                previous_workspace_distance = current_workspace_distance
                await asyncio.sleep(0)
                if self._done or details.get("status") != "running":
                    break
        finally:
            close = getattr(trajectory, "close", None)
            if callable(close):
                close()
        status: ExecutionStatus = (
            "success" if self._success else "failure" if self._done else "running"
        )
        return ExecutionFeedback(
            status,
            observation,
            executed,
            details,
        )

    def _home_workspace_distance(self, observation: Observation) -> float:
        """Return the configured EEF AABB distance for one home observation."""
        if self._home_workspace is None:
            return 0.0
        return workspace_violation_distance(
            pose_from_observation(observation),
            self._home_workspace,
        )

    async def finalize_episode(
        self,
        cancel_event: asyncio.Event | None = None,
    ) -> Observation:
        """让官方 evaluator 用安全空动作完成终态验证窗口。"""
        self._check_cancelled(cancel_event)
        step = self._env.settle()
        self._raw_observation = step.observation
        self._success = bool(step.success)
        self._done = bool(step.done)
        observation = self._build_observation(self._raw_observation, success=self._success)
        self._publish(observation)
        return observation

    def _publish(self, observation: Observation) -> None:
        if self._observation_recorder is not None:
            self._observation_recorder(observation)
        if self._observation_sink is not None:
            self._observation_sink.publish(observation)

    @staticmethod
    def _check_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
