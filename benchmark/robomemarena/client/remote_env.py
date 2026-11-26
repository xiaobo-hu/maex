"""Python 3.11 client for the isolated RoboMemArena simulator service."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from uuid import uuid4

from websockets.sync.client import connect

from benchmark.libero.rpc import RpcError, pack_message, unpack_message

from ..config import RoboMemArenaConfig
from ..environment import RoboMemArenaStep
from ..preprocessing import validate_action


class RemoteRoboMemArenaEnv:
    """Drive one serialized Python 3.8 RoboMemArena environment over RPC."""

    def __init__(
        self,
        config: RoboMemArenaConfig | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8001,
    ) -> None:
        self._config = config or RoboMemArenaConfig()
        self._socket = connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self._task_description = ""

    @property
    def task_description(self) -> str:
        """Return the official prompt reported by reset."""
        return self._task_description

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        """Reset and return the stabilized post-warmup observation."""
        params = self._config.to_reset_params(seed)
        if int(params["seed"]) < 0:
            raise ValueError("seed must be non-negative")
        result = self._call("reset", params)
        self._task_description = str(result.get("task_description", ""))
        return self._observation_from(result)

    def observe(self) -> Mapping[str, object]:
        """Read the latest observation without stepping MuJoCo."""
        return self._observation_from(self._call("observe"))

    def step(self, action: Sequence[float]) -> RoboMemArenaStep:
        """Execute exactly one finite 7-D action."""
        result = self._call("step", {"action": validate_action(action)})
        return RoboMemArenaStep(
            observation=self._observation_from(result),
            reward=float(result.get("reward", 0.0)),
            done=bool(result.get("done", False)),
            success=bool(result.get("success", False)),
        )

    def return_to_home(self, gripper_command: float) -> RoboMemArenaStep:
        """Drive the arm to the joint positions captured at episode start."""
        result = self._call("return_to_home", {"gripper_command": float(gripper_command)})
        return self._step_from_result(result)

    def iter_return_to_home(self, gripper_command: float) -> Iterator[RoboMemArenaStep]:
        """Yield every physics observation produced by the joint-space home path."""
        self._call("begin_return_to_home", {"gripper_command": float(gripper_command)})
        completed = False
        try:
            while True:
                result = self._call("step_return_to_home")
                step = self._step_from_result(result)
                completed = step.details.get("status") != "running"
                yield step
                if completed:
                    return
        finally:
            if not completed:
                self._call("abort_return_to_home")

    @classmethod
    def _step_from_result(cls, result: dict[str, object]) -> RoboMemArenaStep:
        """Build a normalized step without exposing the raw RPC envelope."""
        return RoboMemArenaStep(
            observation=cls._observation_from(result),
            done=bool(result.get("done", False)),
            success=bool(result.get("success", False)),
            details={
                key: value
                for key, value in result.items()
                if key not in {"observation", "done", "success", "task_description"}
            },
        )

    def settle(self) -> RoboMemArenaStep:
        """Advance the benchmark-owned terminal verification with neutral actions."""
        result = self._call("settle")
        return RoboMemArenaStep(
            observation=self._observation_from(result),
            done=bool(result.get("done", False)),
            success=bool(result.get("success", False)),
        )

    def metrics(self) -> Mapping[str, object]:
        """Read official CSR/TSR diagnostics through the evaluation-only channel."""
        result = self._call("metrics")
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise RpcError("RoboMemArena metrics result is missing evaluation")
        return evaluation

    def close(self) -> None:
        """Close both the remote environment and WebSocket connection."""
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
            raise RpcError("RoboMemArena RPC response id does not match request")
        if response.get("ok") is not True:
            raise RpcError(str(response.get("error", "Unknown RoboMemArena RPC error")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RpcError("RoboMemArena RPC result must be a mapping")
        return result

    @staticmethod
    def _observation_from(result: dict[str, object]) -> Mapping[str, object]:
        observation = result.get("observation")
        if not isinstance(observation, Mapping):
            raise RpcError("RoboMemArena RPC result is missing observation")
        return observation
