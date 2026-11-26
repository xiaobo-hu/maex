"""Python 3.11 synchronous client implementing the existing LiberoEnv boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

from websockets.sync.client import connect

from .config import LiberoEvalConfig
from .environment import LiberoStep
from .rpc import RpcError, pack_message, unpack_message


class RemoteLiberoEnv:
    """Drive the isolated Python 3.8 LIBERO environment over serial RPC."""

    def __init__(
        self,
        config: LiberoEvalConfig,
        *,
        host: str = "127.0.0.1",
        port: int = 8001,
        resolution: int = 256,
    ) -> None:
        self._config = config
        self._resolution = resolution
        self._socket = connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self._task_description = ""

    @property
    def task_description(self) -> str:
        """Return the language description reported by the last reset."""
        return self._task_description

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        """Reset the configured suite/task and return its stabilized observation."""
        task = self._config.task
        result = self._call(
            "reset",
            {
                "suite": task.suite,
                "task_id": task.task_id,
                "seed": task.seed if seed is None else seed,
                "init_state_id": task.init_state_id,
                "warmup_steps": self._config.warmup_steps,
                "resolution": self._resolution,
            },
        )
        self._task_description = str(result.get("task_description", ""))
        return self._observation_from(result)

    def observe(self) -> Mapping[str, object]:
        """Return the last observation without advancing MuJoCo."""
        return self._observation_from(self._call("observe"))

    def step(self, action: Sequence[float]) -> LiberoStep:
        """Execute exactly one 7-D action in the remote environment."""
        result = self._call("step", {"action": [float(value) for value in action]})
        return LiberoStep(
            observation=self._observation_from(result),
            reward=float(result.get("reward", 0.0)),
            done=bool(result.get("done", False)),
            success=bool(result.get("success", False)),
            info=result.get("info") if isinstance(result.get("info"), Mapping) else None,
        )

    def close(self) -> None:
        """Close the remote MuJoCo environment and the WebSocket connection."""
        try:
            self._call("close")
        finally:
            self._socket.close()

    def _call(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request_id = str(uuid4())
        self._socket.send(pack_message({"id": request_id, "method": method, "params": params or {}}))
        response = unpack_message(self._socket.recv())
        if response.get("id") != request_id:
            raise RpcError("LIBERO RPC response id does not match request")
        if response.get("ok") is not True:
            raise RpcError(str(response.get("error", "Unknown LIBERO RPC error")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RpcError("LIBERO RPC result must be a mapping")
        return result

    @staticmethod
    def _observation_from(result: dict[str, object]) -> Mapping[str, object]:
        observation = result.get("observation")
        if not isinstance(observation, Mapping):
            raise RpcError("LIBERO RPC result is missing observation")
        return observation
