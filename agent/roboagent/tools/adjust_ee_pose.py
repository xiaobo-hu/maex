"""在 frozen VLA 重试前执行一个受限的局部末端恢复动作。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from agent_core import AgentToolResult
from agent_core.types import AgentToolUpdateCallback

from ..types import Observation
from ..utils.ee_pose import EePoseAdjustmentAction
from .common import result_with_observation


class AdjustEePoseTool:
    """把单轴语义调整交给 Runtime 的闭环 OSC 执行器。"""

    name = "adjust_ee_pose"
    label = "Adjust End-Effector Pose"
    description = (
        "Executes one bounded, contact-free relative end-effector adjustment. Translation "
        "magnitude is in meters; rotation magnitude is in degrees. Returns the applied "
        "displacement and latest observation. Execution success only means the requested "
        "pose was reached; task effect remains unverified. Does not grasp, place, insert, "
        "or open."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "world_z_up",
                    "camera_z_retract",
                    "camera_x_left",
                    "camera_x_right",
                    "camera_rx_pos",
                    "camera_rx_neg",
                    "camera_ry_pos",
                    "camera_ry_neg",
                ],
                "description": "One contact-free translation or wrist-camera rotation axis.",
            },
            "magnitude": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Meters for world_z_up/camera_z_retract/camera_x_left/"
                    "camera_x_right; degrees for camera_rx_* and camera_ry_*. "
                    "Runtime applies configured safety maxima."
                ),
            },
            "decision_summary": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reason for choosing this tool now.",
            },
        },
        "required": ["action", "magnitude"],
        "additionalProperties": False,
    }
    prepare_arguments = None
    execution_mode = "sequential"

    def __init__(
        self,
        adjust: Callable[
            [EePoseAdjustmentAction, float, asyncio.Event | None],
            Awaitable[tuple[dict[str, object], Observation | None]],
        ],
    ) -> None:
        self._adjust = adjust

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        action = cast(EePoseAdjustmentAction, str(params["action"]))
        payload, observation = await self._adjust(
            action,
            float(cast(float, params["magnitude"])),
            cancel_event,
        )
        return result_with_observation(payload, observation)
