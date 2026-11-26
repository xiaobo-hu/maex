"""记录 LIBERO Observer 输入窗口和判断，供本地离线标注使用。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from roboagent import ImageFrame, Observation
from roboagent.observer import ObserverAssessment, ObserverPort


class LiberoObserverRecorder:
    """包装 ObserverPort，并把不进入正式 trace 的图像写入本地目录。"""

    def __init__(self, observer: ObserverPort, record_dir: Path) -> None:
        self._observer = observer
        self._record_dir = record_dir
        self._window_index = 0

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
        """调用 Observer 后保存当前窗口；记录失败不改变判断结果。"""
        assessment = await self._observer.assess(
            task,
            instruction,
            observations,
            cancel_event,
            anchor_index=anchor_index,
            gripper_transition_index=gripper_transition_index,
            previous_observation=previous_observation,
            previous_instruction=previous_instruction,
        )
        self._window_index += 1
        try:
            self._record(
                task,
                instruction,
                gripper_transition_index,
                previous_observation,
                previous_instruction,
                observations,
                assessment,
                anchor_index,
            )
        except OSError as error:
            logging.warning("Failed to record Observer window: %s", error)
        return assessment

    def _record(
        self,
        task: str,
        instruction: str,
        gripper_transition_index: int | None,
        previous_observation: str,
        previous_instruction: str,
        observations: Sequence[Observation],
        assessment: ObserverAssessment,
        anchor_index: int | None,
    ) -> None:
        window_name = f"window-{self._window_index:04d}"
        window_dir = self._record_dir / window_name
        window_dir.mkdir(parents=True, exist_ok=True)
        for timestep, observation in enumerate(observations, start=1):
            self._write_frame(window_dir, timestep, "external", observation.rgb)
            self._write_frame(window_dir, timestep, "wrist", observation.wrist_rgb)
        entry = {
            "window": window_name,
            "task": task,
            "instruction": instruction,
            "gripper_transition_index": gripper_transition_index,
            "previous_observation": previous_observation,
            "previous_instruction": previous_instruction,
            "timesteps": len(observations),
            "anchor_index": anchor_index,
            "assessment": asdict(assessment),
        }
        with (self._record_dir / "assessments.jsonl").open("a", encoding="utf-8") as output:
            output.write(f"{json.dumps(entry, ensure_ascii=False)}\n")

    @staticmethod
    def _write_frame(
        window_dir: Path,
        timestep: int,
        view: str,
        frame: ImageFrame | None,
    ) -> None:
        if frame is None:
            return
        suffix = ".jpg" if frame.mime_type == "image/jpeg" else ".png"
        path = window_dir / f"t{timestep:02d}-{view}{suffix}"
        path.write_bytes(base64.b64decode(frame.data))
