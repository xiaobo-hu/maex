"""验证本地 Observer 的提示词、输入顺序和结构化输出边界。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboagent import ImageFrame, Observation
from roboagent.memory import build_visual_memory
from roboagent.observer import (
    ExecutionDiagnosticsRunner,
    ObservationEvent,
    ObserverAssessment,
    ObserverSamplingOptions,
    OpenAiObserverAdapter,
    VisualObserver,
    create_observation_stream,
    detect_gripper_keyframes,
    sample_observer_window,
)
from roboagent.prompts import OBSERVER_SYSTEM_PROMPT


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_observer_prompt_is_loaded_from_markdown() -> None:
    assert "robot visual observer" in OBSERVER_SYSTEM_PROMPT
    assert "Do not propose the next action" in OBSERVER_SYSTEM_PROMPT
    assert "remains accurate for the newest images" in OBSERVER_SYSTEM_PROMPT
    assert "verbatim instead of paraphrasing it" in OBSERVER_SYSTEM_PROMPT
    assert "not currently visible" in OBSERVER_SYSTEM_PROMPT
    assert "does not specify its direction" in OBSERVER_SYSTEM_PROMPT
    assert "one or two short sentences" in OBSERVER_SYSTEM_PROMPT
    assert "verbatim-reuse rule below takes precedence" in OBSERVER_SYSTEM_PROMPT


def test_observation_stream_separates_publish_and_read_capabilities() -> None:
    sink, source = create_observation_stream()

    assert hasattr(sink, "publish")
    assert not hasattr(sink, "snapshot_after")
    assert hasattr(source, "snapshot_after")
    assert not hasattr(source, "publish")


def test_visual_memory_deduplicates_by_last_occurrence_without_limit() -> None:
    context = build_visual_memory([None, "", "  ", "A", "B", "A"])
    assert context.summary() == {"observations": ["B", "A"], "keyframes": []}
    assert build_visual_memory(["A", "B", "C", "D"]).observations == (
        "A",
        "B",
        "C",
        "D",
    )


def test_gripper_keyframes_use_command_runs_and_cumulative_qpos_motion() -> None:
    sink, source = create_observation_stream()
    anchor = sink.publish(
        Observation(
            rgb=ImageFrame(data="frame-0"),
            robot_state={
                "gripper_command": -1.0,
                "gripper_qpos": [0.04, -0.04],
            },
        )
    )
    for frame, opening in enumerate((0.035, 0.03, 0.025, 0.02, 0.015), start=1):
        sink.publish(
            Observation(
                rgb=ImageFrame(data=f"frame-{frame}"),
                robot_state={
                    "gripper_command": 1.0,
                    "gripper_qpos": [opening, -opening],
                },
            )
        )
    for frame, opening in enumerate((0.02, 0.025, 0.03, 0.035, 0.04), start=6):
        sink.publish(
            Observation(
                rgb=ImageFrame(data=f"frame-{frame}"),
                robot_state={
                    "gripper_command": -1.0,
                    "gripper_qpos": [opening, -opening],
                },
            )
        )

    keyframes = detect_gripper_keyframes(
        anchor,
        source.snapshot_after(anchor.sequence),
        threshold=0.01,
    )

    assert [(item.event, item.command_sequence, item.sequence) for item in keyframes] == [
        ("gripper_close", 2, 6),
        ("gripper_open", 7, 11),
    ]
    assert [item.robot_state["gripper_qpos"] for item in keyframes] == [
        [0.015, -0.015],
        [0.04, -0.04],
    ]


def test_execution_diagnostics_records_call_states_without_failure_inference() -> None:
    sink, source = create_observation_stream()
    initial = Observation(
        robot_state={
            "eef_position": [0.0, 0.0, 0.3],
            "eef_axis_angle": [0.0, 0.0, 0.0],
            "gripper_command": -1.0,
            "gripper_qpos": [0.04, -0.04],
        }
    )
    sink.publish(initial)
    diagnostics = ExecutionDiagnosticsRunner(
        source,
        gripper_transition_threshold=0.01,
    )
    diagnostics.begin_vla_call(initial)
    sink.publish(
        Observation(
            robot_state={
                "eef_position": [0.0, 0.0, 0.2],
                "eef_axis_angle": [0.0, 0.0, 0.0],
                "gripper_command": 1.0,
                "gripper_qpos": [0.02, -0.02],
            }
        )
    )
    latest = Observation(
        robot_state={
            "eef_position": [0.0, 0.0, 0.25],
            "eef_axis_angle": [0.0, 0.0, 0.0],
            "gripper_command": 1.0,
            "gripper_qpos": [0.001, -0.001],
        }
    )
    sink.publish(latest)

    result = diagnostics.end_vla_call(
        latest,
        chunks_used=2,
        stop_reason="gripper_checkpoint",
    )
    summary = result.summary()

    assert summary["start"] == {
        "sequence": 1,
        "robot_state": {
            **initial.robot_state,
            "gripper_command_state": "open",
            "gripper_width_m": 0.08,
        },
    }
    assert summary["end"] == {
        "sequence": 3,
        "robot_state": {
            **latest.robot_state,
            "gripper_command_state": "close",
            "gripper_width_m": 0.002,
        },
    }
    assert summary["gripper_events"][0]["event"] == "gripper_close"
    assert summary["gripper_events"][0]["sequence"] == 3
    assert summary["gripper_events"][0]["robot_state"][
        "gripper_command_state"
    ] == "close"
    assert summary["chunks_used"] == 2
    assert summary["stop_reason"] == "gripper_checkpoint"
    assert "failed" not in str(summary)
    assert "recovery" not in str(summary)


def test_gripper_command_prevents_qpos_only_keyframe_fallback() -> None:
    anchor = ObservationEvent(
        sequence=1,
        captured_at=0.0,
        observation=Observation(
            robot_state={"gripper_command": 1.0, "gripper_qpos": [0.0, 0.0]}
        ),
    )
    events = (
        ObservationEvent(
            sequence=2,
            captured_at=1.0,
            observation=Observation(
                rgb=ImageFrame(data="rgb"),
                robot_state={"gripper_command": 1.0, "gripper_qpos": [0.006, 0.006]},
            ),
        ),
    )

    window = sample_observer_window(
        anchor,
        events,
        ObserverSamplingOptions(
            max_timesteps=2,
            min_new_observations=1,
            gripper_transition_threshold=0.01,
        ),
    )

    assert window.gripper_transition_sequence is None


def test_openai_observer_sends_ordered_views_and_parses_json() -> None:
    async def run() -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=_response(
                            '{"keyframe":2,'
                            '"observation":"The bowl is held by the gripper."}'
                        )
                    )
                )
            )
        )
        observations = [
            Observation(
                rgb=ImageFrame(data="main-1"),
                wrist_rgb=ImageFrame(data="wrist-1"),
            ),
            Observation(
                rgb=ImageFrame(data="main-2"),
                wrist_rgb=ImageFrame(data="wrist-2"),
            ),
        ]
        with patch("roboagent.observer.openai_adapter.AsyncOpenAI", return_value=client):
            adapter = OpenAiObserverAdapter(
                base_url="http://127.0.0.1:8002/v1",
                model="robo-observer",
            )
            assessment = await adapter.assess(
                "Move the bowl",
                "grasp the bowl",
                observations,
                anchor_index=1,
                gripper_transition_index=2,
                previous_observation="The gripper is near the bowl.",
                previous_instruction="approach the bowl",
            )

        assert assessment.keyframe == 2
        assert assessment.observation == "The bowl is held by the gripper."
        request = client.chat.completions.create.await_args.kwargs
        assert request["temperature"] == 0
        assert request["max_tokens"] == 256
        schema = request["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["keyframe"]["anyOf"][0]["enum"] == [2]
        user_content = request["messages"][1]["content"]
        assert "Previous visual observation: The gripper is near the bowl." in user_content[0][
            "text"
        ]
        assert "Only this timestep may be a keyframe" in user_content[0]["text"]
        assert "Previous VLA instruction: approach the bowl" in user_content[0]["text"]
        assert "Current VLA instruction: grasp the bowl" in user_content[0]["text"]
        assert "ordered oldest to newest" in user_content[0]["text"]
        assert "START observation" in user_content[1]["text"]
        assert "external fixed eye-to-hand" in user_content[1]["text"]
        assert "START observation" in user_content[3]["text"]
        assert "wrist-mounted eye-in-hand" in user_content[3]["text"]
        assert "END observation" in user_content[5]["text"]
        image_urls = [
            item["image_url"]["url"]
            for item in user_content
            if item["type"] == "image_url"
        ]
        # 伪 base64 无法解码成图像,差分图不生成,只保留两个 timestep 的双视角原图。
        assert image_urls == [
            "data:image/png;base64,main-1",
            "data:image/png;base64,wrist-1",
            "data:image/png;base64,main-2",
            "data:image/png;base64,wrist-2",
        ]

    asyncio.run(run())


def test_sampler_selects_start_gripper_and_end_events_without_anchor() -> None:
    sink, source = create_observation_stream()
    anchor = sink.publish(
        Observation(
            rgb=ImageFrame(data="frame-0"),
            robot_state={"gripper_qpos": [0.0, 0.0]},
        )
    )
    for frame in range(1, 11):
        qpos = [1.0, 1.0] if frame >= 5 else [0.0, 0.0]
        sink.publish(
            Observation(
                rgb=ImageFrame(data=f"frame-{frame}"),
                robot_state={"gripper_qpos": qpos},
            )
        )

    window = sample_observer_window(
        anchor,
        source.snapshot_after(anchor.sequence),
        ObserverSamplingOptions(
            max_timesteps=5,
            min_new_observations=10,
            gripper_transition_threshold=0.1,
        ),
    )

    # 首尾窗口:START(seq2)+ 夹爪转变(seq6)+ END(seq11),不再携带 anchor 帧。
    assert [event.sequence for event in window.events] == [2, 6, 11]
    assert window.anchor_sequence is None
    assert window.gripper_transition_sequence == 6
    assert window.range_start == 2
    assert window.range_end == 11


def test_runner_maps_existing_gripper_transition_to_keyframe() -> None:
    class GripperObserver:
        def __init__(self) -> None:
            self.transition_index: int | None = None
            self.called = asyncio.Event()

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
                previous_observation,
                previous_instruction,
            )
            self.transition_index = gripper_transition_index
            self.called.set()
            return ObserverAssessment(gripper_transition_index, "The gripper closes.")

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(
            Observation(
                rgb=ImageFrame(data="f0"),
                robot_state={"gripper_qpos": [0.0, 0.0]},
            )
        )
        observer = GripperObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(
                max_timesteps=3,
                min_new_observations=2,
                gripper_transition_threshold=0.1,
            ),
        )
        await runner.start("Grasp the object")
        runner.set_instruction("grasp the object")
        sink.publish(
            Observation(
                rgb=ImageFrame(data="f1"),
                robot_state={"gripper_qpos": [0.0, 0.0]},
            )
        )
        transition = sink.publish(
            Observation(
                rgb=ImageFrame(data="f2"),
                robot_state={"gripper_qpos": [1.0, 1.0]},
            )
        )
        await observer.called.wait()
        await runner.stop()

        # 无 anchor 后窗口只含 [f1, f2],夹爪转变帧是第 2 个 timestep。
        assert observer.transition_index == 2
        assert runner.trace[0]["gripper_transition_sequence"] == transition.sequence
        assert runner.trace[0]["keyframe_sequence"] == transition.sequence

    asyncio.run(run())


def test_runner_is_single_flight_and_consumes_non_overlapping_new_ranges() -> None:
    class BlockingObserver:
        def __init__(self) -> None:
            self.calls: list[tuple[list[int], int | None]] = []
            self.previous_observations: list[str] = []
            self.active = 0
            self.max_active = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

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
                cancel_event,
                gripper_transition_index,
                previous_instruction,
            )
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.previous_observations.append(previous_observation)
            self.calls.append(
                ([int(item.robot_state["frame"]) for item in observations], anchor_index)
            )
            try:
                if len(self.calls) == 1:
                    self.started.set()
                    await self.release.wait()
                return ObserverAssessment(None, "The bowl is held by the gripper.")
            finally:
                self.active -= 1

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(Observation(rgb=ImageFrame(data="f0"), robot_state={"frame": 0}))
        observer = BlockingObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=3, min_new_observations=2),
        )
        await runner.start("Move the bowl")
        instruction_id = runner.set_instruction(
            "grasp the bowl",
            step_id="grasp-bowl",
        )
        assert runner.set_instruction(
            "grasp the bowl",
            step_id="grasp-bowl",
        ) == instruction_id
        sink.publish(Observation(rgb=ImageFrame(data="f1"), robot_state={"frame": 1}))
        sink.publish(Observation(rgb=ImageFrame(data="f2"), robot_state={"frame": 2}))
        await observer.started.wait()
        sink.publish(Observation(rgb=ImageFrame(data="f3"), robot_state={"frame": 3}))
        sink.publish(Observation(rgb=ImageFrame(data="f4"), robot_state={"frame": 4}))
        assert observer.max_active == 1
        observer.release.set()
        for _ in range(20):
            if len(observer.calls) == 2:
                break
            await asyncio.sleep(0)
        await runner.stop()

        assert observer.calls == [([1, 2], None), ([3, 4], None)]
        assert observer.previous_observations == ["", "The bowl is held by the gripper."]
        assert observer.max_active == 1
        assert [(item["range_start"], item["range_end"]) for item in runner.trace] == [
            (2, 3),
            (4, 5),
        ]
        assert {item["step_id"] for item in runner.trace} == {"grasp-bowl"}
        assert runner.trace[-1]["observation"] == "The bowl is held by the gripper."

    asyncio.run(run())


def test_runner_returns_each_tail_window_with_its_own_call() -> None:
    class FrameObserver:
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
                cancel_event,
                anchor_index,
                gripper_transition_index,
                previous_observation,
                previous_instruction,
            )
            frame = observations[-1].robot_state["frame"]
            return ObserverAssessment(None, f"Observed frame {frame}.")

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(Observation(rgb=ImageFrame(data="f0"), robot_state={"frame": 0}))
        runner = VisualObserver(
            FrameObserver(),
            source,
            sampling=ObserverSamplingOptions(max_timesteps=3, min_new_observations=3),
        )
        await runner.start("Move the object")

        # 短调用达不到中途阈值：tail 仍必须在本次调用返回。
        sink.publish(Observation(rgb=ImageFrame(data="f1"), robot_state={"frame": 1}))
        runner.begin_vla_call("grasp the object", step_id="grasp")
        sink.publish(Observation(rgb=ImageFrame(data="f2"), robot_state={"frame": 2}))
        first = await runner.end_vla_call()
        assert first.observations == ("Observed frame 2.",)
        assert first.keyframes == ()

        sink.publish(Observation(rgb=ImageFrame(data="f3"), robot_state={"frame": 3}))
        runner.begin_vla_call("grasp the object", step_id="grasp")
        sink.publish(Observation(rgb=ImageFrame(data="f4"), robot_state={"frame": 4}))
        second = await runner.end_vla_call()
        assert second.observations == ("Observed frame 4.",)
        await runner.stop()

        assert [(event["range_start"], event["range_end"]) for event in runner.trace] == [
            (2, 3),
            (4, 5),
        ]
        assert all(event["is_tail"] for event in runner.trace)

    asyncio.run(run())


def test_dynamic_windows_cover_frames_during_inference_without_chunk_queue() -> None:
    async def run() -> None:
        sink, source = create_observation_stream()
        calls = []
        started = [asyncio.Event() for _ in range(3)]
        releases = [asyncio.Event() for _ in range(3)]

        class FrameObserver:
            async def assess(self, task, instruction, observations, **kwargs):
                index = len(calls)
                calls.append(([o.rgb.data for o in observations], kwargs))
                started[index].set()
                await releases[index].wait()
                return ObserverAssessment(None, f"window-{index}")

        runner = VisualObserver(
            FrameObserver(), source,
            sampling=ObserverSamplingOptions(min_new_observations=2),
        )
        def publish(frame):
            sink.publish(Observation(rgb=ImageFrame(data=f"f{frame}")))

        publish(0)
        await runner.start("Move the object")
        runner.begin_vla_call("move", step_id="move")
        publish(1)
        publish(2)
        await asyncio.wait_for(started[0].wait(), 2)
        for frame in range(3, 21):
            publish(frame)
        assert len(calls) == 1
        releases[0].set()
        await asyncio.wait_for(started[1].wait(), 2)
        # 推理期间的帧组成下一窗口，其起点是上次输入末帧。
        assert calls[1][0] == ["f2", "f20"]
        publish(21)
        releases[1].set()
        # 后续只有一帧，也应立即评估，不再等首窗阈值。
        await asyncio.wait_for(started[2].wait(), 2)
        ending = asyncio.create_task(runner.end_vla_call())
        await asyncio.sleep(0)
        assert not ending.done()
        releases[2].set()
        memory = await asyncio.wait_for(ending, 2)
        assert [c[0] for c in calls] == [["f0", "f2"], ["f2", "f20"], ["f20", "f21"]]
        assert memory.observations == ("window-0", "window-1", "window-2")
        assert [c[1]["previous_observation"] for c in calls] == ["", "window-0", "window-1"]
        runner.begin_vla_call("move", step_id="move")
        assert (await runner.end_vla_call()).observations == ()
        await runner.stop()

    asyncio.run(run())


def test_runner_attaches_keyframe_images_without_separate_describe() -> None:
    class CanonicalObserver:
        def __init__(self) -> None:
            self.calls: list[tuple[list[int], int | None]] = []

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
                cancel_event,
                anchor_index,
                previous_observation,
                previous_instruction,
            )
            frames = [int(item.robot_state["frame"]) for item in observations]
            self.calls.append((frames, gripper_transition_index))
            if gripper_transition_index is None:
                return ObserverAssessment(None, f"Observed frame {frames[-1]}.")
            frame = frames[gripper_transition_index - 1]
            return ObserverAssessment(
                gripper_transition_index,
                f"Canonical keyframe {frame}.",
            )

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(
            Observation(
                rgb=ImageFrame(data="f0"),
                wrist_rgb=ImageFrame(data="w0"),
                robot_state={"frame": 0},
            )
        )
        observer = CanonicalObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=3, min_new_observations=10),
        )
        await runner.start("Move the object")
        runner.begin_vla_call("grasp the object", step_id="grasp")
        sink.publish(Observation(rgb=ImageFrame(data="f1"), robot_state={"frame": 1}))
        canonical = sink.publish(
            Observation(
                rgb=ImageFrame(data="f2"),
                wrist_rgb=ImageFrame(data="w2"),
                robot_state={"frame": 2},
            )
        )

        memory = await runner.end_vla_call(
            ((canonical.sequence, "gripper_close"),)
        )
        # 不再为关键帧发起单独评估；tail 窗口在本调用内完成。
        assert observer.calls == [([0, 2], None)]
        assert len(memory.keyframes) == 1
        keyframe = memory.keyframes[0]
        assert keyframe.sequence == canonical.sequence
        assert keyframe.rgb == canonical.observation.rgb
        assert keyframe.wrist_rgb == canonical.observation.wrist_rgb
        # 尾部描述归入 VisualMemory，关键帧本身仍不重复描述。
        assert keyframe.observation is None
        assert memory.observations == ("Observed frame 2.",)

        await runner.stop()
        assert observer.calls == [([0, 2], None)]

    asyncio.run(run())


def test_runner_ignores_adjustment_frames_but_uses_final_frame_as_next_vla_anchor() -> None:
    class FrameObserver:
        def __init__(self) -> None:
            self.frames: list[list[str]] = []

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
                cancel_event,
                anchor_index,
                gripper_transition_index,
                previous_observation,
                previous_instruction,
            )
            frames = [str(item.robot_state["frame"]) for item in observations]
            self.frames.append(frames)
            return ObserverAssessment(None, f"Observed {frames[-1]}.")

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(Observation(rgb=ImageFrame(data="initial"), robot_state={"frame": "initial"}))
        observer = FrameObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=3, min_new_observations=1),
        )
        await runner.start("Move the object")

        sink.publish(Observation(rgb=ImageFrame(data="adjust-1"), robot_state={"frame": "adjust-1"}))
        sink.publish(Observation(rgb=ImageFrame(data="adjust-2"), robot_state={"frame": "adjust-final"}))
        await asyncio.sleep(0)
        assert observer.frames == []

        runner.begin_vla_call("grasp the object", step_id="grasp")
        sink.publish(Observation(rgb=ImageFrame(data="vla"), robot_state={"frame": "vla"}))
        await runner.end_vla_call()
        await runner.stop()

        # 恢复动作不单独触发观察，但其最终状态作为下一窗口 START。
        assert observer.frames == [["adjust-final", "vla"]]

    asyncio.run(run())


def test_runner_marks_late_result_stale_after_instruction_switch() -> None:
    class SwitchingObserver:
        def __init__(self) -> None:
            self.instructions: list[str] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

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
                observations,
                cancel_event,
                anchor_index,
                gripper_transition_index,
                previous_observation,
                previous_instruction,
            )
            self.instructions.append(instruction)
            if len(self.instructions) == 1:
                self.started.set()
                await self.release.wait()
            return ObserverAssessment(None, f"Context for {instruction}.")

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(Observation(rgb=ImageFrame(data="f0")))
        observer = SwitchingObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=3, min_new_observations=2),
        )
        await runner.start("Move the bowl")
        runner.set_instruction("old instruction")
        sink.publish(Observation(rgb=ImageFrame(data="f1")))
        sink.publish(Observation(rgb=ImageFrame(data="f2")))
        await observer.started.wait()

        runner.set_instruction("new instruction")
        sink.publish(Observation(rgb=ImageFrame(data="f3")))
        sink.publish(Observation(rgb=ImageFrame(data="f4")))
        observer.release.set()
        for _ in range(20):
            if len(observer.instructions) == 2:
                break
            await asyncio.sleep(0)
        await runner.stop()

        assert observer.instructions == ["old instruction", "new instruction"]
        assert [entry["stale"] for entry in runner.trace] == [True, False]
        assert [entry["instruction_id"] for entry in runner.trace] == [1, 2]
        assert runner.trace[1]["previous_instruction"] == "old instruction"

    asyncio.run(run())


def test_runner_preserves_previous_observation_across_step_switch() -> None:
    class ContextObserver:
        def __init__(self) -> None:
            self.inputs: list[tuple[str, str]] = []

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
            del task, observations, cancel_event, anchor_index, gripper_transition_index
            self.inputs.append((previous_observation, previous_instruction))
            observation = (
                "The cookie jar is near the basket; tomato sauce remains on the table."
                if instruction == "place cookies"
                else "Tomato sauce remains on the table."
            )
            return ObserverAssessment(None, observation)

    async def wait_for_calls(observer, count: int) -> None:
        for _ in range(20):
            if len(observer.inputs) >= count:
                return
            await asyncio.sleep(0)

    async def run() -> None:
        sink, source = create_observation_stream()
        sink.publish(Observation(rgb=ImageFrame(data="f0")))
        observer = ContextObserver()
        runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=2, min_new_observations=1),
        )
        await runner.start("Place both objects")
        runner.set_instruction("place cookies", step_id="cookies")
        sink.publish(Observation(rgb=ImageFrame(data="f1")))
        await wait_for_calls(observer, 1)

        runner.set_instruction("place tomato sauce", step_id="tomato")
        sink.publish(Observation(rgb=ImageFrame(data="f2")))
        await wait_for_calls(observer, 2)
        sink.publish(Observation(rgb=ImageFrame(data="f3")))
        await wait_for_calls(observer, 3)
        await runner.stop()

        assert observer.inputs == [
            ("", ""),
            (
                "The cookie jar is near the basket; tomato sauce remains on the table.",
                "place cookies",
            ),
            ("Tomato sauce remains on the table.", ""),
        ]

    asyncio.run(run())


def test_openai_observer_degrades_invalid_keyframe_to_empty_record() -> None:
    async def run() -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=_response(
                            '{"keyframe":3,"observation":"The bowl is held."}'
                        )
                    )
                )
            )
        )
        with patch("roboagent.observer.openai_adapter.AsyncOpenAI", return_value=client):
            adapter = OpenAiObserverAdapter(
                base_url="http://127.0.0.1:8002/v1",
                model="robo-observer",
            )
            assessment = await adapter.assess(
                "Move the bowl",
                "grasp the bowl",
                [Observation(rgb=ImageFrame(data="main"))],
                gripper_transition_index=None,
            )

        assert assessment.keyframe is None
        assert assessment.observation is None

    asyncio.run(run())


@pytest.mark.parametrize(
    ('content', 'finish_reason', 'request_error', 'expected_stage', 'expected_error'),
    [
        ('{"keyframe":null,"observation":null}', 'stop', False, None, None),
        ('{"keyframe":null,"observation":"The bottle tilted."}', 'stop', False, None, None),
        ('{"keyframe":null,"observation":"The bottle', 'length', False, 'truncation', 'OutputTruncated'),
        ('{"keyframe":null,"observation":null}', 'length', False, 'truncation', 'OutputTruncated'),
        ('{"keyframe":', 'stop', False, 'parse', 'JSONDecodeError'),
        ('{"keyframe":null,"observation":42}', 'stop', False, 'parse', 'ValueError'),
        ('', None, True, 'request', 'ConnectionError'),
    ],
)
def test_observer_records_finish_reason_and_failure_kind(
    tmp_path, content, finish_reason, request_error, expected_stage, expected_error,
) -> None:
    """正常 null、截断、解析错误与断连在 trace 和磁盘记录中可区分。"""
    import json

    from benchmark.robomemarena.client.observer_recording import RoboMemArenaObserverRecorder

    async def run() -> None:
        response = _response(content)
        response.choices[0].finish_reason = finish_reason
        create = AsyncMock(return_value=response)
        if request_error:
            create.side_effect = ConnectionError('offline')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('roboagent.observer.openai_adapter.AsyncOpenAI', return_value=client):
            adapter = OpenAiObserverAdapter()
        sink, source = create_observation_stream()
        runner = VisualObserver(RoboMemArenaObserverRecorder(adapter, tmp_path), source)
        await runner.start('Pour')
        runner.begin_vla_call('pour sauce', step_id='pour')
        sink.publish(Observation(rgb=ImageFrame(data='YWJj')))
        await runner.end_vla_call()
        await runner.stop()
        event = runner.trace[0]
        entry = json.loads((tmp_path / 'assessments.jsonl').read_text())['assessment']
        for actual in (event, entry):
            assert actual['finish_reason'] == finish_reason
            assert actual['error_stage'] == expected_stage
            assert actual['error_type'] == expected_error
            if expected_error:
                assert actual['observation'] is None
        assert create.call_args.kwargs['max_tokens'] == 256

    asyncio.run(run())
