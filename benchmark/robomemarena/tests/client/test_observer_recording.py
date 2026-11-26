"""Verify RoboMemArena Observer logs keep images out of JSONL."""

from __future__ import annotations

import asyncio
import base64
import json

from benchmark.robomemarena.client.observer_recording import RoboMemArenaObserverRecorder

from roboagent import ImageFrame, Observation
from roboagent.observer import ObserverAssessment


class FakeObserver:
    async def assess(
        self,
        task,
        instruction,
        observations,
        cancel_event=None,
        *,
        anchor_index=None,
        gripper_transition_index=None,
        previous_observation="",
        previous_instruction="",
    ):
        del (
            task,
            instruction,
            observations,
            cancel_event,
            anchor_index,
            gripper_transition_index,
            previous_observation,
            previous_instruction,
        )
        return ObserverAssessment(1, "The cookies are held.")


def test_recorder_writes_images_and_jsonl(tmp_path) -> None:
    async def run() -> None:
        recorder = RoboMemArenaObserverRecorder(FakeObserver(), tmp_path)
        assessment = await recorder.assess(
            "Move the cookies",
            "put the cookies in the basket",
            [
                Observation(
                    rgb=ImageFrame(base64.b64encode(b"agent").decode("ascii")),
                    wrist_rgb=ImageFrame(base64.b64encode(b"wrist").decode("ascii")),
                )
            ],
            gripper_transition_index=1,
            previous_observation="The cookies were lifted.",
            previous_instruction="approach cookies",
        )

        assert assessment.observation == "The cookies are held."
        assert (tmp_path / "window-0001/t01-external.png").read_bytes() == b"agent"
        assert (tmp_path / "window-0001/t01-wrist.png").read_bytes() == b"wrist"
        entry = json.loads((tmp_path / "assessments.jsonl").read_text())
        assert entry["assessment"]["keyframe"] == 1
        assert entry["gripper_transition_index"] == 1
        assert entry["previous_observation"] == "The cookies were lifted."
        assert entry["previous_instruction"] == "approach cookies"
        assert entry["assessment"]["observation"] == "The cookies are held."
        assert "base64" not in json.dumps(entry)

    asyncio.run(run())
