"""RoboMemArena 环境服务终态收敛的回归测试。"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.server.service import RoboMemArenaEnvironmentService


def _observation() -> dict[str, object]:
    return {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


class FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[list[float]] = []

    def step(self, action):
        self.actions.append(list(action))
        return _observation(), 0.0, False, {}


class FakeEvaluator:
    def __init__(self) -> None:
        self.updates = 0

    @property
    def done(self) -> bool:
        return self.updates >= 3

    @property
    def success(self) -> bool:
        return self.done

    def update(self, observation, *, simulator_done=False) -> None:
        del observation, simulator_done
        self.updates += 1


class DelayedStageEvaluator(FakeEvaluator):
    @property
    def done(self) -> bool:
        return self.updates >= 4

    @property
    def remaining_stage_checks(self) -> int:
        return 2


def test_settle_advances_neutral_actions_until_official_done() -> None:
    service = RoboMemArenaEnvironmentService()
    environment = FakeEnvironment()
    evaluator = FakeEvaluator()
    service._env = environment
    service._observation = _observation()
    service._evaluator = evaluator
    service._config = RoboMemArenaConfig(post_goal_steps=2, extra_pour_monitor_steps=1)

    result = service.dispatch("settle", {})

    assert result["done"] is True
    assert result["success"] is True
    assert result["settled_steps"] == 3
    assert environment.actions == [[0.0] * 6 + [-1.0]] * 3


def test_settle_allows_each_remaining_ordered_stage_one_update() -> None:
    service = RoboMemArenaEnvironmentService()
    environment = FakeEnvironment()
    evaluator = DelayedStageEvaluator()
    service._env = environment
    service._observation = _observation()
    service._evaluator = evaluator
    service._config = RoboMemArenaConfig(post_goal_steps=2, extra_pour_monitor_steps=1)

    result = service.dispatch("settle", {})

    assert result["done"] is True
    assert result["settled_steps"] == 4


def test_return_to_home_allows_long_converging_joint_trajectory(monkeypatch) -> None:
    class Controller:
        def reset_goal(self) -> None:
            pass

    class Robot:
        def __init__(self) -> None:
            self.controller = Controller()
            self.controller_config = {}
            self.recent_actions = object()
            self.action_dim = 8

    class HomeEnvironment(FakeEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.robots = [Robot()]
            self._action_dim = 7

        def step(self, action):
            self.actions.append(list(action))
            joints = np.ones(7) if len(self.actions) < 217 else np.zeros(7)
            observation = {**_observation(), "robot0_joint_pos": joints}
            return observation, 0.0, False, {}

    controllers = ModuleType("robosuite.controllers")
    controllers.controller_factory = lambda *_args, **_kwargs: Controller()
    controllers.load_controller_config = lambda **_kwargs: {}
    buffers = ModuleType("robosuite.utils.buffers")
    buffers.DeltaBuffer = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "robosuite", ModuleType("robosuite"))
    monkeypatch.setitem(sys.modules, "robosuite.controllers", controllers)
    monkeypatch.setitem(sys.modules, "robosuite.utils", ModuleType("robosuite.utils"))
    monkeypatch.setitem(sys.modules, "robosuite.utils.buffers", buffers)

    service = RoboMemArenaEnvironmentService()
    environment = HomeEnvironment()
    service._env = environment
    service._observation = {**_observation(), "robot0_joint_pos": np.ones(7)}
    service._evaluator = SimpleNamespace(done=False, success=False, update=lambda *_a, **_k: None)
    service._home_joint_positions = np.zeros(7)

    original_controller = environment.robots[0].controller
    begin = service.dispatch("begin_return_to_home", {"gripper_command": -1.0})
    first = service.dispatch("step_return_to_home", {})
    result = first
    while result["status"] == "running":
        result = service.dispatch("step_return_to_home", {})

    assert begin["status"] == "running"
    assert begin["executed_steps"] == 0
    assert first["status"] == "running"
    assert first["executed_steps"] == 1
    assert result["status"] == "success"
    assert result["home_pose_reached"] is True
    assert result["executed_steps"] == 217
    assert environment.robots[0].controller is original_controller
