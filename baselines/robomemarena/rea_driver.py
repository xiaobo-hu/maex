"""仿真适配层：通过 RoboMemArena RPC 将 REA 的动作调用转为 π0.5 动作块。"""

import base64
from io import BytesIO

import numpy as np
from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.preprocessing import build_policy_input
from PIL import Image


class RoboMemArenaDriver:
    """提供 REA 观察与动作接口，仿真服务管理物理步预算。"""

    id = "robomemarena"
    timing = "turn-based"
    metadata = {"simulator": "RoboMemArena", "controlHz": 20, "actionDimension": 7}
    actions = [{
        "name": "vla_chunk",
        "description": "Execute a frozen RoboMemArena π0.5 action chunk for the given subtask.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        },
    }]

    def __init__(self, options):
        self.env_port = int(options["envPort"])
        self.policy_port = int(options["policyPort"])
        self.max_steps = int(options["maxSteps"])
        self.env = None
        self.policy = None
        self.latest = None

    def reset(self, task=None, seed=50):
        """Reset the selected task on this client's simulator connection."""
        if self.env is not None:
            self.env.close()
        config = RoboMemArenaConfig(task_id=int(task), seed=seed, max_steps=self.max_steps)
        self.env = RemoteRoboMemArenaEnv(config, port=self.env_port)
        self.latest = self.env.reset()
        return self.observe()

    def observe(self):
        """Return the two camera frames and the eight-dimensional policy state."""
        payload = build_policy_input(self.latest, self.env.task_description)
        cameras = (("agentview", "observation/image"), ("wrist", "observation/wrist_image"))
        images = []
        for name, key in cameras:
            with BytesIO() as buffer:
                Image.fromarray(payload[key]).save(buffer, format="JPEG", quality=90)
                images.append({"name": name, "mimeType": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode("ascii")})
        return {"text": self.env.task_description, "images": images, "state": {"eef": payload["observation/state"].tolist()}}

    def step(self, action, execute=False):
        """Execute at most ten actions and report the new observation at the chunk boundary."""
        if not execute:
            return {**self.observe(), "info": {"executedSteps": 0}}
        prompt = action["arguments"]["prompt"].strip()
        if not prompt:
            raise ValueError("vla_chunk requires a nonempty prompt")
        if self.policy is None:
            self.policy = WebsocketPolicyClient("127.0.0.1", self.policy_port)
        payload = build_policy_input(self.latest, prompt)
        actions = np.asarray(self.policy.infer(payload)["actions"])
        if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("π0.5 must return nonempty, finite [N, 7] actions")
        executed_steps = 0
        for command in actions[:10]:
            result = self.env.step(command)
            executed_steps += 1
            self.latest = result.observation
            if result.done:
                break
        return {
            **self.observe(), "terminated": result.done and result.success,
            "truncated": result.done and not result.success, "success": result.success if result.done else None,
            "info": {"executedSteps": executed_steps},
        }

    def stop(self):
        """A step-based simulation stays idle between requests."""
        return {"supported": True}

    def close(self):
        """Release simulator and policy connections."""
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self.policy is not None:
                self.policy.close()
