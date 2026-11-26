"""通过 OpenAI-compatible API 调用本地多帧视觉 Observer。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace

from openai import AsyncOpenAI

from ..prompts import OBSERVER_SYSTEM_PROMPT
from ..types import ImageFrame, Observation
from ..visual_evidence import build_change_highlight
from .types import ObserverAssessment


class OpenAiObserverAdapter:
    """把多视角观察转换为本地 vLLM 的结构化视觉请求。"""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8002/v1",
        model: str = "robo-observer",
        timeout: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self._model = model

    async def assess(
        self,
        task: str,
        instruction: str,
        observations: Sequence[Observation],
        cancel_event: asyncio.Event | None = None,
        *,
        anchor_index: int | None = None,
        gripper_transition_index: int | None = None,
        previous_observation: str = "",
        previous_instruction: str = "",
    ) -> ObserverAssessment:
        """分析采样器给出的有序 timestep，服务故障时返回空记录。"""
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        window = [item for item in observations if item.rgb or item.wrist_rgb]
        if not window:
            return ObserverAssessment(keyframe=None, observation=None)
        if anchor_index is not None and not 1 <= anchor_index <= len(window):
            return ObserverAssessment(keyframe=None, observation=None)
        if gripper_transition_index is not None and not 1 <= gripper_transition_index <= len(
            window
        ):
            return ObserverAssessment(keyframe=None, observation=None)
        finish_reason = None
        error_stage = "request"
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(
                    task,
                    instruction,
                    window,
                    anchor_index,
                    gripper_transition_index,
                    previous_observation,
                    previous_instruction,
                ),
                response_format=self._response_format(gripper_transition_index),
                temperature=0,
                max_tokens=256,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            error_stage = "parse"
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                return ObserverAssessment(
                    None, None, finish_reason=finish_reason,
                    error_stage="truncation", error_type="OutputTruncated",
                )
            content = choice.message.content
            if not isinstance(content, str):
                raise ValueError("Observer response did not contain text")
            return replace(
                self._parse(content, gripper_transition_index), finish_reason=finish_reason,
            )
        except Exception as error:
            return ObserverAssessment(
                None, None, finish_reason=finish_reason,
                error_stage=error_stage, error_type=type(error).__name__,
            )

    @staticmethod
    def _messages(
        task: str,
        instruction: str,
        observations: Sequence[Observation],
        anchor_index: int | None,
        gripper_transition_index: int | None,
        previous_observation: str,
        previous_instruction: str,
    ) -> list[dict[str, object]]:
        context_lines = [
            f"Global task: {task.strip()}",
        ]
        if previous_instruction.strip():
            context_lines.append(
                f"Previous VLA instruction: {previous_instruction.strip()}"
            )
        context_lines.append(f"Current VLA instruction: {instruction.strip()}")
        if previous_observation.strip():
            context_lines.extend(
                (
                    f"Previous visual observation: {previous_observation.strip()}",
                    "The previous visual observation is unverified model output. "
                    "Use it only for temporal comparison and correct it when images disagree.",
                )
            )
        if gripper_transition_index is None:
            context_lines.append(
                "No gripper open/close transition was detected; keyframe must be null."
            )
        else:
            context_lines.append(
                "Detected gripper open/close transition: "
                f"timestep {gripper_transition_index}. Only this timestep may be a keyframe."
            )
        context_lines.append(
            f"Recent visual context: {len(observations)} timesteps ordered oldest to newest."
        )
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "\n".join(context_lines),
            }
        ]
        for index, observation in enumerate(observations, start=1):
            role = _timestep_role(index, len(observations), gripper_transition_index)
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Timestep {index}/{len(observations)} ({role})\n"
                        "View A — external fixed eye-to-hand camera:"
                    ),
                }
            )
            if observation.rgb is not None:
                content.append(OpenAiObserverAdapter._image_content(observation.rgb))
            if observation.wrist_rgb is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Timestep {index}/{len(observations)} ({role})\n"
                            "View B — wrist-mounted eye-in-hand camera:"
                        ),
                    }
                )
                content.append(OpenAiObserverAdapter._image_content(observation.wrist_rgb))
        if (
            len(observations) >= 2
            and observations[0].rgb is not None
            and observations[-1].rgb is not None
        ):
            highlight = build_change_highlight(observations[0].rgb, observations[-1].rgb)
            if highlight is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Difference highlight over View A — red overlays and boxes "
                            "mark pixel-level change between the START and END "
                            "observations."
                        ),
                    }
                )
                content.append(OpenAiObserverAdapter._image_content(highlight))
        content.append(
            {
                "type": "text",
                "text": "Record only grounded visual evidence and return the required JSON.",
            }
        )
        return [
            {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _image_content(frame: ImageFrame) -> dict[str, object]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{frame.mime_type};base64,{frame.data}"},
        }

    @staticmethod
    def _response_format(gripper_transition_index: int | None) -> dict[str, object]:
        keyframe_schema: dict[str, object]
        if gripper_transition_index is None:
            keyframe_schema = {"type": "null"}
        else:
            keyframe_schema = {
                "anyOf": [
                    {"type": "integer", "enum": [gripper_transition_index]},
                    {"type": "null"},
                ]
            }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "observer_assessment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "keyframe": keyframe_schema,
                        "observation": {
                            "anyOf": [
                                {"type": "string", "minLength": 1},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["keyframe", "observation"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _parse(
        content: str,
        gripper_transition_index: int | None,
    ) -> ObserverAssessment:
        payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {"keyframe", "observation"}:
            raise ValueError("Observer response has invalid fields")
        keyframe = payload["keyframe"]
        observation = payload["observation"]
        if keyframe is not None and (
            not isinstance(keyframe, int)
            or isinstance(keyframe, bool)
            or keyframe != gripper_transition_index
        ):
            raise ValueError("Observer response has invalid keyframe")
        if observation is not None and (
            not isinstance(observation, str) or not observation.strip()
        ):
            raise ValueError("Observer response has invalid observation")
        return ObserverAssessment(
            keyframe=keyframe,
            observation=observation.strip() if observation is not None else None,
        )


def _timestep_role(
    index: int,
    total: int,
    gripper_transition_index: int | None,
) -> str:
    """给首尾窗口中的每个 timestep 一个明确的角色标签。"""
    if total == 1:
        return "START and END observation"
    if index == 1:
        return "START observation"
    if index == total:
        return "END observation"
    if index == gripper_transition_index:
        return "gripper transition candidate"
    return "intermediate observation"
