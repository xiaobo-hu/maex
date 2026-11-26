"""Python 3.8 owner for RoboMemArena physics and official evaluation state."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from websockets.sync.server import serve

from benchmark.libero.rpc import pack_message, unpack_message

from ..config import RoboMemArenaConfig
from ..preprocessing import (
    ROBOMEMARENA_DUMMY_ACTION,
    select_policy_observation,
    validate_action,
)
from .camera_observation import camera_observation
from .evaluation import OfficialStageEvaluator
from .upstream import load_official_modules

_RETURN_HOME_MAX_CONTROL_STEPS = 480


@dataclass
class _ReturnHomeSession:
    """Keep the temporary joint controller alive across observable RPC steps."""

    robot: Any
    base_env: Any
    old_controller: Any
    old_action_dim: int
    old_recent_actions: Any
    before: np.ndarray
    target: np.ndarray
    gripper_command: float
    executed_steps: int = 0


class RoboMemArenaEnvironmentService:
    """Own one upstream environment and dispatch serial lifecycle calls."""

    def __init__(self) -> None:
        self._env = None
        self._observation = None
        self._evaluator = None
        self._config: RoboMemArenaConfig | None = None
        self._task_description = ""
        self._home_joint_positions = None
        self._return_home_session: _ReturnHomeSession | None = None

    def dispatch(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Execute one reset/observe/step/metrics/close operation."""
        if method == "reset":
            return self._reset(params)
        if method == "observe":
            return self._result()
        if method == "step":
            return self._step(params)
        if method == "return_to_home":
            return self._return_to_home(params)
        if method == "begin_return_to_home":
            return self._begin_return_to_home(params)
        if method == "step_return_to_home":
            return self._step_return_to_home()
        if method == "abort_return_to_home":
            return self._abort_return_to_home()
        if method == "settle":
            return self._settle()
        if method == "metrics":
            return {"evaluation": self._metrics()}
        if method == "close":
            self.close()
            return {"closed": True}
        raise ValueError(f"Unknown RoboMemArena environment method: {method}")

    def close(self) -> None:
        """Close MuJoCo and discard episode state."""
        self._finish_return_to_home()
        if self._env is not None:
            self._env.close()
        self._env = None
        self._observation = None
        self._evaluator = None
        self._config = None
        self._task_description = ""
        self._home_joint_positions = None
        self._return_home_session = None

    def _reset(self, params: dict[str, object]) -> dict[str, object]:
        config = RoboMemArenaConfig(
            task_id=int(params.get("task_id", 1)),
            seed=int(params.get("seed", 100)),
            warmup_steps=int(params.get("warmup_steps", 10)),
            max_steps=int(params.get("max_steps", 2500)),
            post_goal_steps=int(params.get("post_goal_steps", 200)),
            extra_pour_monitor_steps=int(params.get("extra_pour_monitor_steps", 30)),
            camera_height=int(params.get("camera_height", 480)),
            camera_width=int(params.get("camera_width", 640)),
            enable_depth=bool(params.get("enable_depth", False)),
        )
        eval_common, stages = load_official_modules()
        bddl_path = eval_common._resolve_bddl_path(config.task_id)
        task_key = f"task{config.task_id}"

        self.close()
        env_class = eval_common._get_env_class()
        # 构造函数也会采样物体位置，必须先固定随机数；正式 reset 仍独立重新设种子。
        np.random.seed(config.seed)
        self._env = env_class(
            bddl_file_name=str(bddl_path),
            camera_heights=config.camera_height,
            camera_widths=config.camera_width,
            ignore_done=True,
            reward_shaping=True,
            control_freq=20,
            initialization_noise=None,
            **({"camera_depths": True} if config.enable_depth else {}),
        )
        np.random.seed(config.seed)
        if hasattr(self._env, "seed"):
            self._env.seed(config.seed)
        self._observation = self._env.reset()
        for _ in range(config.warmup_steps):
            self._observation, _reward, simulator_done, _info = self._env.step(
                ROBOMEMARENA_DUMMY_ACTION
            )
            if simulator_done:
                break
        self._home_joint_positions = self._joint_positions(self._observation).copy()
        self._task_description = str(eval_common.get_prompt(task_key, bddl_path.stem))
        self._evaluator = OfficialStageEvaluator(
            config.task_id,
            self._env,
            max_steps=config.max_steps,
            post_goal_steps=config.post_goal_steps,
            extra_pour_monitor_steps=config.extra_pour_monitor_steps,
            official_module=stages,
        )
        self._config = config
        return self._result()

    def _step(self, params: dict[str, object]) -> dict[str, object]:
        if self._env is None or self._evaluator is None:
            raise RuntimeError("RoboMemArena environment must be reset before step")
        if self._evaluator.done:
            raise RuntimeError("RoboMemArena episode is done; call reset before step")
        action = validate_action(params.get("action"))
        self._observation, reward, simulator_done, _info = self._env.step(action)
        self._evaluator.update(self._observation, simulator_done=bool(simulator_done))
        result = self._result()
        result["reward"] = float(reward)
        return result

    def _settle(self) -> dict[str, object]:
        """用安全空动作推进官方 post-goal / extra-pour 终态窗口。"""
        if self._evaluator is None or self._config is None:
            raise RuntimeError("RoboMemArena environment must be reset before settle")
        settled_steps = 0
        budget = max(
            self._config.post_goal_steps,
            self._config.extra_pour_monitor_steps,
        ) + max(1, int(getattr(self._evaluator, "remaining_stage_checks", 1)))
        while not self._evaluator.done and settled_steps < budget:
            self._step({"action": ROBOMEMARENA_DUMMY_ACTION})
            settled_steps += 1
        result = self._result()
        result["settled_steps"] = settled_steps
        return result

    def _return_to_home(self, params: dict[str, object]) -> dict[str, object]:
        """Run the observable home-step API to completion for legacy clients."""
        self._begin_return_to_home(params)
        while True:
            result = self._step_return_to_home()
            if result.get("status") != "running":
                return result

    def _begin_return_to_home(self, params: dict[str, object]) -> dict[str, object]:
        """Install the temporary joint controller without advancing physics."""
        if self._env is None or self._evaluator is None:
            raise RuntimeError("RoboMemArena environment must be reset before return_to_home")
        if self._evaluator.done:
            raise RuntimeError("RoboMemArena episode is done; call reset before return_to_home")
        if self._home_joint_positions is None:
            raise RuntimeError("Episode-start joint positions are unavailable")
        if self._return_home_session is not None:
            raise RuntimeError("A return_to_home trajectory is already active")
        gripper_command = float(params.get("gripper_command", 0.0))
        if not np.isfinite(gripper_command) or not -1.0 <= gripper_command <= 1.0:
            raise ValueError("gripper_command must be finite and within [-1, 1]")

        from robosuite.controllers import controller_factory, load_controller_config
        from robosuite.utils.buffers import DeltaBuffer

        robot = self._env.robots[0]
        base_env = self._env
        while not hasattr(base_env, "action_dim") and hasattr(base_env, "env"):
            base_env = base_env.env
        old_controller = robot.controller
        old_action_dim = base_env._action_dim
        old_recent_actions = robot.recent_actions
        before = self._joint_positions(self._observation).copy()
        target = self._home_joint_positions.copy()

        controller_config = dict(robot.controller_config)
        controller_config.update(
            load_controller_config(default_controller="JOINT_POSITION")
        )
        try:
            robot.controller = controller_factory("JOINT_POSITION", controller_config)
            robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
            base_env._action_dim = robot.action_dim
            self._return_home_session = _ReturnHomeSession(
                robot=robot,
                base_env=base_env,
                old_controller=old_controller,
                old_action_dim=old_action_dim,
                old_recent_actions=old_recent_actions,
                before=before,
                target=target,
                gripper_command=gripper_command,
            )
        except Exception:
            robot.controller = old_controller
            robot.recent_actions = old_recent_actions
            base_env._action_dim = old_action_dim
            raise
        return self._return_home_result("running")

    def _step_return_to_home(self) -> dict[str, object]:
        """Execute exactly one observable joint-controller physics step."""
        session = self._return_home_session
        if session is None:
            raise RuntimeError("No return_to_home trajectory is active")
        try:
            current = self._joint_positions(self._observation)
            error = session.target - current
            if float(np.max(np.abs(error))) <= 0.02:
                result = self._return_home_result("success")
                self._finish_return_to_home()
                return result
            action = np.concatenate(
                (
                    np.clip(error / 0.05, -1.0, 1.0),
                    [session.gripper_command],
                )
            )
            self._observation, _reward, simulator_done, _info = self._env.step(action)
            self._evaluator.update(
                self._observation,
                simulator_done=bool(simulator_done),
            )
            session.executed_steps += 1

            after = self._joint_positions(self._observation)
            home_reached = float(np.max(np.abs(session.target - after))) <= 0.02
            if home_reached:
                status = "success"
                reason = None
            elif self._evaluator.done:
                status = "controller_failed"
                reason = "The environment ended before the joint home was reached."
            elif session.executed_steps >= _RETURN_HOME_MAX_CONTROL_STEPS:
                status = "controller_failed"
                reason = (
                    "The joint-position controller did not converge within "
                    f"{_RETURN_HOME_MAX_CONTROL_STEPS} steps."
                )
            else:
                return self._return_home_result("running")
            result = self._return_home_result(status, reason)
            self._finish_return_to_home()
            return result
        except Exception as error:
            result = self._return_home_result("controller_failed", str(error))
            self._finish_return_to_home()
            return result

    def _abort_return_to_home(self) -> dict[str, object]:
        """Restore the normal controller after a client-side safety stop."""
        if self._return_home_session is None:
            return self._result()
        result = self._return_home_result(
            "controller_failed",
            "The return_to_home trajectory was aborted by the client.",
        )
        self._finish_return_to_home()
        return result

    def _return_home_result(
        self,
        status: str,
        reason: str | None = None,
    ) -> dict[str, object]:
        """Build one cumulative result for the active home trajectory."""
        session = self._return_home_session
        if session is None:
            raise RuntimeError("No return_to_home trajectory is active")
        after = self._joint_positions(self._observation).copy()
        max_error = float(np.max(np.abs(session.target - after)))
        home_reached = max_error <= 0.02
        if home_reached:
            status = "success"
            reason = None
        result = self._result()
        result.update(
            {
                "status": status,
                "controller": "JOINT_POSITION",
                "joint_positions_before": session.before.tolist(),
                "joint_positions_target": session.target.tolist(),
                "joint_positions_after": after.tolist(),
                "max_joint_error_rad": max_error,
                "executed_steps": session.executed_steps,
                "home_pose_reached": home_reached,
            }
        )
        if reason is not None:
            result["reason"] = reason
        return result

    def _finish_return_to_home(self) -> None:
        """Restore the episode's normal controller after a home trajectory."""
        session = self._return_home_session
        if session is None:
            return
        session.robot.controller = session.old_controller
        session.robot.recent_actions = session.old_recent_actions
        session.base_env._action_dim = session.old_action_dim
        session.old_controller.reset_goal()
        self._return_home_session = None

    def _metrics(self) -> dict[str, object]:
        if self._evaluator is None:
            raise RuntimeError("RoboMemArena environment must be reset before metrics")
        return self._evaluator.metrics()

    @staticmethod
    def _joint_positions(observation: object) -> np.ndarray:
        if not isinstance(observation, dict) or "robot0_joint_pos" not in observation:
            raise RuntimeError("RoboMemArena observation is missing robot0_joint_pos")
        positions = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            raise RuntimeError("robot0_joint_pos must contain seven finite values")
        return positions

    def _result(self) -> dict[str, object]:
        if self._observation is None or self._evaluator is None:
            raise RuntimeError("RoboMemArena environment has no observation; call reset first")
        observation = select_policy_observation(self._observation)
        if self._config is not None and self._config.enable_depth:
            observation.update(camera_observation(self._env.sim, self._observation))
        return {
            "observation": observation,
            "done": bool(self._evaluator.done),
            "success": bool(self._evaluator.success),
            "task_description": self._task_description,
        }


class RoboMemArenaEnvironmentServer:
    """Expose one serialized RoboMemArena service over WebSocket RPC."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        self._host = host
        self._port = port
        self._service = RoboMemArenaEnvironmentService()
        self._physical_lock = threading.Lock()

    def serve_forever(self) -> None:
        """Serve requests until interrupted."""
        logging.info("RoboMemArena environment server listening on %s:%s", self._host, self._port)
        with serve(
            self._handle_connection,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            try:
                server.serve_forever()
            finally:
                self._service.close()

    def _handle_connection(self, websocket: Any) -> None:
        for encoded in websocket:
            request_id = None
            try:
                request = unpack_message(encoded)
                request_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})
                if not isinstance(method, str) or not isinstance(params, dict):
                    raise ValueError("RPC request requires string method and mapping params")
                with self._physical_lock:
                    result = self._service.dispatch(method, params)
                response = {"id": request_id, "ok": True, "result": result}
            except Exception as error:
                response = {"id": request_id, "ok": False, "error": str(error)}
            websocket.send(pack_message(response))
