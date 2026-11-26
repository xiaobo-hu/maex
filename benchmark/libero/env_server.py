"""Python 3.8 LIBERO/MuJoCo owner with strictly serialized environment calls."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from websockets.sync.server import serve

from .preprocessing import LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, select_policy_observation
from .rpc import pack_message, unpack_message


class LiberoEnvironmentService:
    """Own one LIBERO environment and dispatch reset/observe/step/close calls."""

    def __init__(self) -> None:
        self._env = None
        self._observation = None
        self._done = False
        self._task_description = ""

    def dispatch(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """Execute one environment method synchronously."""
        if method == "reset":
            return self._reset(params)
        if method == "observe":
            return self._result()
        if method == "step":
            return self._step(params)
        if method == "close":
            self.close()
            return {"closed": True}
        raise ValueError(f"Unknown LIBERO environment method: {method}")

    def close(self) -> None:
        """Close the current MuJoCo environment, if one exists."""
        if self._env is not None:
            self._env.close()
        self._env = None
        self._observation = None
        self._done = False

    def _reset(self, params: dict[str, object]) -> dict[str, object]:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite_name = str(params.get("suite", "libero_spatial"))
        task_id = int(params.get("task_id", 0))
        seed = int(params.get("seed", 0))
        init_state_id = int(params.get("init_state_id", 0))
        warmup_steps = int(params.get("warmup_steps", 10))
        resolution = int(params.get("resolution", LIBERO_ENV_RESOLUTION))

        suites = benchmark.get_benchmark_dict()
        if suite_name not in suites:
            raise ValueError(f"Unknown LIBERO suite: {suite_name}")
        task_suite = suites[suite_name]()
        if not 0 <= task_id < task_suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {task_suite.n_tasks})")
        initial_states = task_suite.get_task_init_states(task_id)
        if not 0 <= init_state_id < len(initial_states):
            raise ValueError(f"init_state_id must be in [0, {len(initial_states)})")

        self.close()
        task = task_suite.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self._env.seed(seed)
        self._env.reset()
        self._observation = self._env.set_init_state(initial_states[init_state_id])
        self._done = False
        self._task_description = str(task.language)
        for _ in range(warmup_steps):
            self._observation, _reward, self._done, _info = self._env.step(
                LIBERO_DUMMY_ACTION
            )
            if self._done:
                break
        return self._result()

    def _step(self, params: dict[str, object]) -> dict[str, object]:
        if self._env is None:
            raise RuntimeError("LIBERO environment must be reset before step")
        action = params.get("action")
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError("LIBERO action must be a 7-D list")
        self._observation, reward, self._done, info = self._env.step(action)
        result = self._result()
        result["reward"] = float(reward)
        result["info"] = info or {}
        return result

    def _result(self) -> dict[str, object]:
        if self._observation is None:
            raise RuntimeError("LIBERO environment has no observation; call reset first")
        return {
            "observation": select_policy_observation(self._observation),
            "done": bool(self._done),
            "success": bool(self._done),
            "task_description": self._task_description,
        }


class LiberoEnvironmentServer:
    """Expose :class:`LiberoEnvironmentService` over one serial WebSocket RPC endpoint."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        self._host = host
        self._port = port
        self._service = LiberoEnvironmentService()
        self._physical_lock = threading.Lock()

    def serve_forever(self) -> None:
        """Serve requests until the process receives an interrupt."""
        logging.info("LIBERO environment server listening on %s:%s", self._host, self._port)
        with serve(self._handle_connection, self._host, self._port, compression=None, max_size=None) as server:
            try:
                server.serve_forever()
            finally:
                self._service.close()

    def _handle_connection(self, websocket) -> None:
        for encoded in websocket:
            request_id = None
            try:
                request = unpack_message(encoded)
                request_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})
                if not isinstance(method, str) or not isinstance(params, dict):
                    raise ValueError("RPC request requires string method and mapping params")
                # MuJoCo 环境由单一进程持有；即使误连多个客户端也不能并发 step。
                with self._physical_lock:
                    result = self._service.dispatch(method, params)
                response = {"id": request_id, "ok": True, "result": result}
            except Exception as error:
                response = {"id": request_id, "ok": False, "error": str(error)}
            websocket.send(pack_message(response))
