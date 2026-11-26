from __future__ import annotations

import numpy as np
from benchmark.libero.config import LiberoEvalConfig, LiberoTaskSpec
from benchmark.libero.remote_env import RemoteLiberoEnv
from benchmark.libero.rpc import pack_message, unpack_message


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    def send(self, message: bytes) -> None:
        self.sent.append(message)

    def recv(self) -> bytes:
        request = unpack_message(self.sent[-1])
        method = request["method"]
        result: dict[str, object]
        if method == "reset":
            result = {
                "observation": _observation(),
                "task_description": "pick the bowl",
                "done": False,
                "success": False,
            }
        elif method == "step":
            result = {
                "observation": _observation(),
                "reward": 1.0,
                "done": True,
                "success": True,
                "info": {},
            }
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


def test_remote_env_preserves_task_and_serial_step(monkeypatch) -> None:
    socket = FakeSocket()
    monkeypatch.setattr("benchmark.libero.remote_env.connect", lambda *args, **kwargs: socket)
    config = LiberoEvalConfig(task=LiberoTaskSpec("libero_spatial", 2, 3, 4), warmup_steps=5)
    env = RemoteLiberoEnv(config)

    observation = env.reset()
    step = env.step([0.0] * 7)
    env.close()

    assert "agentview_image" in observation
    assert env.task_description == "pick the bowl"
    assert step.success is True
    requests = [unpack_message(message) for message in socket.sent]
    assert [request["method"] for request in requests] == ["reset", "step", "close"]
    assert requests[0]["params"] == {
        "suite": "libero_spatial",
        "task_id": 2,
        "seed": 3,
        "init_state_id": 4,
        "warmup_steps": 5,
        "resolution": 256,
    }
    assert socket.closed is True
