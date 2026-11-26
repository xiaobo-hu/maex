from __future__ import annotations

import numpy as np
from benchmark.libero.rpc import pack_message, unpack_message
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.config import RoboMemArenaConfig


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    def send(self, message: bytes) -> None:
        self.sent.append(message)

    def recv(self) -> bytes:
        request = unpack_message(self.sent[-1])
        method = request["method"]
        if method in {"reset", "observe", "step", "settle", "return_to_home"}:
            result: dict[str, object] = {
                "observation": _observation(),
                "task_description": "official task",
                "done": method in {"step", "settle"},
                "success": method in {"step", "settle"},
                "reward": 1.0,
            }
            if method == "return_to_home":
                result.update(
                    {
                        "status": "success",
                        "controller": "JOINT_POSITION",
                        "executed_steps": 8,
                        "home_pose_reached": True,
                    }
                )
        elif method == "metrics":
            result = {"evaluation": {"CSR": 50.0, "TSR": 0}}
        else:
            result = {"closed": True}
        return pack_message({"id": request["id"], "ok": True, "result": result})

    def close(self) -> None:
        self.closed = True


def _observation() -> dict[str, object]:
    return {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


def test_rpc_reset_observe_step_metrics_close(monkeypatch) -> None:
    socket = FakeSocket()
    monkeypatch.setattr("benchmark.robomemarena.client.remote_env.connect", lambda *a, **k: socket)
    env = RemoteRoboMemArenaEnv(RoboMemArenaConfig(task_id=2))

    env.reset()
    env.observe()
    step = env.step([0.0] * 7)
    home = env.return_to_home(-1.0)
    settled = env.settle()
    metrics = env.metrics()
    env.close()

    requests = [unpack_message(message) for message in socket.sent]
    assert [request["method"] for request in requests] == [
        "reset",
        "observe",
        "step",
        "return_to_home",
        "settle",
        "metrics",
        "close",
    ]
    assert requests[0]["params"]["task_id"] == 2
    assert requests[0]["params"]["seed"] == 100
    assert requests[0]["params"]["max_steps"] == 2500
    assert step.success is True
    assert requests[3]["params"]["gripper_command"] == -1.0
    assert home.details["controller"] == "JOINT_POSITION"
    assert home.details["executed_steps"] == 8
    assert settled.success is True
    assert metrics == {"CSR": 50.0, "TSR": 0}
    assert socket.closed is True


def test_return_to_home_iterator_yields_each_server_step(monkeypatch) -> None:
    class StreamingSocket(FakeSocket):
        def __init__(self) -> None:
            super().__init__()
            self.home_steps = 0

        def recv(self) -> bytes:
            request = unpack_message(self.sent[-1])
            method = request["method"]
            if method == "begin_return_to_home":
                result = {"observation": _observation(), "status": "running"}
            elif method == "step_return_to_home":
                self.home_steps += 1
                status = "success" if self.home_steps == 3 else "running"
                result = {
                    "observation": _observation(),
                    "status": status,
                    "executed_steps": self.home_steps,
                    "home_pose_reached": status == "success",
                }
            else:
                return super().recv()
            return pack_message({"id": request["id"], "ok": True, "result": result})

    socket = StreamingSocket()
    monkeypatch.setattr("benchmark.robomemarena.client.remote_env.connect", lambda *a, **k: socket)
    env = RemoteRoboMemArenaEnv()

    steps = list(env.iter_return_to_home(-1.0))

    requests = [unpack_message(message) for message in socket.sent]
    assert [request["method"] for request in requests] == [
        "begin_return_to_home",
        "step_return_to_home",
        "step_return_to_home",
        "step_return_to_home",
    ]
    assert [step.details["executed_steps"] for step in steps] == [1, 2, 3]
    assert steps[-1].details["home_pose_reached"] is True
