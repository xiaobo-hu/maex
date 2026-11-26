from __future__ import annotations

import asyncio
from collections.abc import Sequence

from benchmark.libero.adapter import LiberoRobotPort
from benchmark.libero.environment import LiberoStep

from roboagent import ActionChunk, Observation
from roboagent.observer import create_observation_stream


class FakeEnv:
    def __init__(self) -> None:
        self.steps: list[list[float]] = []

    def reset(self, *, seed: int | None = None) -> dict[str, object]:
        return {"seed": seed, "step": 0}

    def step(self, action: Sequence[float]) -> LiberoStep:
        action_list = list(action)
        self.steps.append(action_list)
        done = len(self.steps) >= 2
        return LiberoStep({"step": len(self.steps)}, done=done, success=done)

    def close(self) -> None:
        pass


def build(raw: object, *, success: bool | None = None) -> Observation:
    return Observation(robot_state={"raw": raw}, task_success=success)


def test_libero_port_serializes_action_chunk() -> None:
    async def run() -> None:
        env = FakeEnv()
        sink, source = create_observation_stream()
        port = LiberoRobotPort(env, build, observation_sink=sink)
        await port.reset(seed=3)
        feedback = await port.execute_action_chunk(ActionChunk([[1.0], [2.0], [3.0]]))
        assert env.steps == [[1.0], [2.0]]
        assert feedback.status == "success"
        assert feedback.executed_steps == 2
        assert [event.observation.robot_state["raw"] for event in source.snapshot_after(0)] == [
            {"seed": 3, "step": 0},
            {"step": 1},
            {"step": 2},
        ]
        assert feedback.observation is source.latest().observation

    asyncio.run(run())


def test_libero_port_publishes_gripper_command_from_action() -> None:
    async def run() -> None:
        env = FakeEnv()
        sink, source = create_observation_stream()
        port = LiberoRobotPort(env, build, observation_sink=sink)
        await port.reset(seed=3)
        await port.execute_action_chunk(
            ActionChunk(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ]
            )
        )
        assert [
            event.observation.robot_state.get("gripper_command")
            for event in source.snapshot_after(0)
        ] == [None, 1.0, -1.0]

    asyncio.run(run())
