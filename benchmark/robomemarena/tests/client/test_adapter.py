from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from benchmark.robomemarena.client.adapter import RoboMemArenaRobotPort
from benchmark.robomemarena.environment import RoboMemArenaStep

from roboagent import ActionChunk, EePoseAdjustmentConfig, Observation
from roboagent.observer import create_observation_stream


class FakeEnv:
    def __init__(self) -> None:
        self.steps: list[list[float]] = []
        self.observe_calls = 0

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        return {"step": 0, "seed": seed}

    def observe(self) -> Mapping[str, object]:
        self.observe_calls += 1
        return {"step": len(self.steps)}

    def step(self, action: Sequence[float]) -> RoboMemArenaStep:
        values = list(action)
        self.steps.append(values)
        done = len(self.steps) == 2
        return RoboMemArenaStep({"step": len(self.steps)}, done=done, success=done)

    def return_to_home(self, gripper_command: float) -> RoboMemArenaStep:
        return RoboMemArenaStep(
            {"step": len(self.steps), "home": True},
            details={
                "status": "success",
                "controller": "JOINT_POSITION",
                "executed_steps": 4,
                "home_pose_reached": True,
                "gripper_command": gripper_command,
            },
        )

    def settle(self) -> RoboMemArenaStep:
        self.steps.append([0.0] * 7)
        return RoboMemArenaStep({"step": len(self.steps)}, done=True, success=True)

    def metrics(self) -> Mapping[str, object]:
        return {"CSR": 100.0, "stage_done": {"secret": True}}

    def close(self) -> None:
        pass


def build(raw: object, *, success: bool | None = None) -> Observation:
    return Observation(robot_state={"raw": raw}, task_success=success, metadata={})


def test_port_observe_does_not_step_and_chunk_is_strictly_serial() -> None:
    async def run() -> None:
        env = FakeEnv()
        sink, source = create_observation_stream()
        recorded: list[Observation] = []
        port = RoboMemArenaRobotPort(
            env,
            build,
            observation_sink=sink,
            observation_recorder=recorded.append,
        )
        await port.reset(seed=100)
        await port.observe()
        feedback = await port.execute_action_chunk(
            ActionChunk([[1.0] * 7, [2.0] * 7, [3.0] * 7])
        )

        assert env.observe_calls == 0
        assert env.steps == [[1.0] * 7, [2.0] * 7]
        assert feedback.status == "success"
        assert feedback.executed_steps == 2
        assert feedback.observation is not None
        assert feedback.observation.metadata == {}
        assert feedback.details == {}
        assert feedback.observation.robot_state["gripper_command"] == 2.0
        assert "CSR" not in feedback.observation.summary()
        events = source.snapshot_after(0)
        assert [event.observation.robot_state["raw"] for event in events] == [
            {"step": 0, "seed": 100},
            {"step": 1},
            {"step": 2},
        ]
        assert [event.observation.robot_state.get("gripper_command") for event in events] == [
            None,
            1.0,
            2.0,
        ]
        assert feedback.observation is source.latest().observation
        assert recorded == [event.observation for event in events]

    asyncio.run(run())


def test_finalize_episode_uses_environment_settle() -> None:
    async def run() -> None:
        env = FakeEnv()
        port = RoboMemArenaRobotPort(env, build)
        await port.reset(seed=100)

        observation = await port.finalize_episode()

        assert env.steps == [[0.0] * 7]
        assert observation.task_success is True

    asyncio.run(run())


def test_return_to_home_uses_environment_joint_controller() -> None:
    async def run() -> None:
        env = FakeEnv()
        port = RoboMemArenaRobotPort(env, build)
        await port.reset(seed=50)

        feedback = await port.return_to_home(-1.0)

        assert feedback.status == "running"
        assert feedback.executed_steps == 4
        assert feedback.details["controller"] == "JOINT_POSITION"
        assert feedback.details["home_pose_reached"] is True
        assert feedback.observation is not None
        assert feedback.observation.robot_state["gripper_command"] == -1.0

    asyncio.run(run())


def test_return_to_home_publishes_each_observable_home_step() -> None:
    class StreamingHomeEnv(FakeEnv):
        def iter_return_to_home(self, gripper_command: float):
            for index, status in enumerate(("running", "running", "success"), start=1):
                yield RoboMemArenaStep(
                    {"step": index, "home": True},
                    details={
                        "status": status,
                        "controller": "JOINT_POSITION",
                        "executed_steps": index,
                        "home_pose_reached": status == "success",
                        "gripper_command": gripper_command,
                    },
                )

    async def run() -> None:
        recorded: list[Observation] = []
        port = RoboMemArenaRobotPort(
            StreamingHomeEnv(),
            build,
            observation_recorder=recorded.append,
        )
        await port.reset(seed=50)

        feedback = await port.return_to_home(-1.0)

        assert feedback.details["status"] == "success"
        assert feedback.executed_steps == 3
        assert [item.robot_state["raw"]["step"] for item in recorded] == [0, 1, 2, 3]

    asyncio.run(run())


def test_return_to_home_aborts_when_path_leaves_workspace() -> None:
    class UnsafeHomeEnv(FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.aborted = False

        def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
            return {"eef_position": [0.0, 0.0, 0.5]}

        def iter_return_to_home(self, gripper_command: float):
            del gripper_command
            try:
                yield RoboMemArenaStep(
                    {"eef_position": [0.6, 0.0, 0.5]},
                    details={"status": "running", "executed_steps": 1},
                )
            finally:
                self.aborted = True

    def build_pose(raw: object, *, success: bool | None = None) -> Observation:
        assert isinstance(raw, Mapping)
        return Observation(
            robot_state={
                "eef_position": raw["eef_position"],
                "eef_axis_angle": [0.0, 0.0, 0.0],
            },
            task_success=success,
            metadata={},
        )

    async def run() -> None:
        env = UnsafeHomeEnv()
        workspace = EePoseAdjustmentConfig(
            workspace_min_m=(-0.45, -0.4, 0.3),
            workspace_max_m=(0.35, 0.65, 1.2),
        )
        port = RoboMemArenaRobotPort(env, build_pose, home_workspace=workspace)
        await port.reset(seed=50)

        feedback = await port.return_to_home(-1.0)

        assert feedback.details["status"] == "out_of_workspace"
        assert feedback.executed_steps == 1
        assert env.aborted is True

    asyncio.run(run())
