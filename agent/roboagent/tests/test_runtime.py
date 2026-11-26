from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from importlib.resources import files

import pytest

from ai import (
    AssistantMessage,
    ImageContent,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    UserMessage,
    create_models,
)
from ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call
from roboagent import (
    DEFAULT_SYSTEM_PROMPT,
    ActionChunk,
    EePoseAdjustmentConfig,
    ExecutionFeedback,
    ImageFrame,
    Observation,
    RoboAgent,
    RoboAgentOptions,
    TaskExecutionSkill,
)
from roboagent.memory import (
    EpisodeMemory,
    ExecutionDiagnostics,
    ExecutionState,
    GripperEventRecord,
    VisualKeyframe,
    VisualMemory,
    VlaWorkingMemory,
)
from roboagent.observer import (
    ObserverAssessment,
    ObserverSamplingOptions,
    VisualObserver,
    create_observation_stream,
)
from roboagent.prompts.robomemarena import task_primitive_reference
from roboagent.recovery import RecoveryState
from roboagent.tools.adjust_ee_pose import AdjustEePoseTool
from roboagent.tools.advance_task_plan import AdvanceTaskPlanTool
from roboagent.tools.close_gripper import CloseGripperTool
from roboagent.tools.create_task_plan import CreateTaskPlanTool
from roboagent.tools.execute_vla import ExecuteVlaTool
from roboagent.tools.finish import FinishTool
from roboagent.tools.observe import ObserveTool
from roboagent.tools.open_gripper import OpenGripperTool
from roboagent.tools.repair_task_plan import RepairTaskPlanTool
from roboagent.tools.restage_vla import RestageVlaTool
from roboagent.tools.return_to_home import ReturnToHomeTool
from roboagent.visual_evidence import (
    build_decision_visual_evidence,
    build_reflection_visual_evidence,
)


def test_episode_context_compacts_old_vla_and_preserves_latest_evidence() -> None:
    memory = EpisodeMemory()
    initial = UserMessage(
        content=[
            TextContent(text="task"),
            ImageContent(data="initial-image", mime_type="image/png"),
        ]
    )
    earlier = ToolResultMessage(
        tool_call_id="earlier",
        tool_name="execute_vla",
        content=[
            TextContent(text="earlier result"),
            ImageContent(data="earlier-image", mime_type="image/png"),
        ],
        details={
            "status": "running",
            "step_id": "pick-cookie",
            "stop_reason": "chunk_budget",
            "chunks_used": 24,
            "working_memory": {"visual": {"observations": ["old"]}},
        },
    )
    text_only = ToolResultMessage(
        tool_call_id="text-only",
        tool_name="advance_task_plan",
        content=[TextContent(text="advanced")],
    )
    latest = ToolResultMessage(
        tool_call_id="latest",
        tool_name="execute_vla",
        content=[
            TextContent(text="latest result"),
            ImageContent(data="latest-image", mime_type="image/png"),
        ],
        details={"status": "running", "working_memory": {"visual": {}}},
    )

    earlier_call = faux_assistant_message(
        faux_tool_call(
            "execute_vla",
            {
                "step_id": "pick-cookie",
                "decision_summary": "Retry the running step.",
            },
            tool_call_id="earlier",
        ),
        stop_reason="toolUse",
    )
    memory.append_tool_event(
        call_id="earlier",
        tool="execute_vla",
        arguments={
            "step_id": "pick-cookie",
            "decision_summary": "Retry the running step.",
        },
        details=earlier.details,
        is_error=False,
    )
    memory.append_tool_event(
        call_id="text-only",
        tool="advance_task_plan",
        arguments={},
        details=text_only.details,
        is_error=False,
    )
    transformed = memory.compact_transcript_for_model(
        [initial, earlier_call, earlier, text_only, latest]
    )

    assert [item.type for item in transformed[0].content] == ["text"]
    assert json.loads(transformed[2].content[0].text) == {
        "episode_memory_ref": "earlier"
    }
    compact_records = json.loads(transformed[3].content[0].text)["episode_memory"]
    assert compact_records[0] == {
        "call_id": "earlier",
        "tool": "execute_vla",
        "step_id": "pick-cookie",
        "decision_summary": "Retry the running step.",
        "status": "running",
        "stop_reason": "chunk_budget",
        "chunks_used": 24,
    }
    assert compact_records[1] == {
        "call_id": "text-only",
        "tool": "advance_task_plan",
    }
    assert "working_memory" not in transformed[2].content[0].text
    assert [item.type for item in transformed[3].content] == ["text"]
    assert [item.type for item in transformed[4].content] == ["text", "image"]
    assert [item.type for item in initial.content] == ["text", "image"]
    assert [item.type for item in earlier.content] == ["text", "image"]


def test_episode_context_keeps_latest_vla_across_new_text_only_result() -> None:
    memory = EpisodeMemory()
    latest_physical = ToolResultMessage(
        tool_call_id="physical",
        tool_name="execute_vla",
        content=[
            TextContent(text="result"),
            ImageContent(data="latest-image", mime_type="image/png"),
        ],
    )
    later_text = ToolResultMessage(
        tool_call_id="plan",
        tool_name="advance_task_plan",
        content=[TextContent(text="advanced")],
    )

    transformed = memory.compact_transcript_for_model([latest_physical, later_text])

    assert [item.type for item in transformed[0].content] == ["text", "image"]


def test_visual_evidence_adapter_keeps_labeled_dual_view_evidence() -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    content = [
        TextContent(text="deterministic diagnostics"),
        TextContent(text="VLA call start observation at sequence 1."),
        ImageContent(data=png, mime_type="image/png"),
        ImageContent(data=png, mime_type="image/png"),
        TextContent(text="Keyframe gripper-keyframe-5."),
        ImageContent(data=png, mime_type="image/png"),
        ImageContent(data=png, mime_type="image/png"),
        TextContent(text="VLA call end/latest observation at sequence 10."),
        ImageContent(data=png, mime_type="image/png"),
        ImageContent(data=png, mime_type="image/png"),
        TextContent(text="Observer advisory"),
    ]

    adapted = build_reflection_visual_evidence(content)

    assert sum(isinstance(item, ImageContent) for item in adapted) == 6
    assert adapted[0].text == "deterministic diagnostics"
    assert adapted[3] is content[2]
    assert adapted[7] is content[8]
    assert adapted[9] is content[3]
    assert adapted[13] is content[9]
    labels = [item.text for item in adapted if isinstance(item, TextContent)]
    evidence_labels = labels[2:]
    assert evidence_labels[0].startswith("START external")
    assert evidence_labels[1].startswith("CHANGE HIGHLIGHT over the END external")
    assert evidence_labels[2].startswith("END external")
    assert evidence_labels[3].startswith("START wrist")
    assert evidence_labels[4].startswith("CHANGE HIGHLIGHT over the END wrist")
    assert evidence_labels[5].startswith("END wrist")
    assert any(
        "END external image" in label and "CHANGE HIGHLIGHT" in label
        for label in labels
    )
    assert any(
        "END wrist image" in label and "CHANGE HIGHLIGHT" in label
        for label in labels
    )
    assert all(
        "Keyframe" not in item.text
        for item in adapted
        if isinstance(item, TextContent)
    )
    assert labels[-1] == "Observer advisory"
    decision_evidence = build_decision_visual_evidence(content, adapted)
    assert sum(isinstance(item, ImageContent) for item in decision_evidence) == 6
    assert decision_evidence[0] is content[0]
    assert decision_evidence[-1] is content[-1]


def test_visual_evidence_adapter_removes_reflection_anchors() -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    start = {"sequence": 1, "robot_state": {"gripper_command": -1.0}}
    end = {"sequence": 10, "robot_state": {"gripper_command": 1.0}}
    content = [
        TextContent(
            text=json.dumps(
                {
                    "current_step": {
                        "id": "step-1",
                        "instruction": "open top drawer",
                        "status": "running",
                    },
                    "execution_diagnostics": {
                        "start": start,
                        "gripper_events": [{"event": "close", "sequence": 5}],
                        "end": end,
                        "chunks_used": 24,
                        "stop_reason": "chunk_budget",
                    },
                    "result": {
                        "status": "running",
                        "stop_reason": "chunk_budget",
                        "recovery_notice": "Review repeated attempts.",
                    },
                    "plan_progress": {"running": "step-1"},
                    "latest_observation": {"task_success": False},
                }
            )
        ),
        TextContent(text="VLA call start observation at sequence 1."),
        ImageContent(data=png, mime_type="image/png"),
        TextContent(text="VLA call end/latest observation at sequence 10."),
        ImageContent(data=png, mime_type="image/png"),
    ]

    adapted = build_reflection_visual_evidence(content)
    header = json.loads(adapted[0].text)

    assert header == {
        "current_step": {"id": "step-1", "instruction": "open top drawer"},
        "gripper_command_convention": (
            "gripper_command > 0 means close; gripper_command < 0 means open"
        ),
        "execution_diagnostics": {
            "start": start,
            "gripper_events": [{"event": "close", "sequence": 5}],
            "end": end,
        },
    }
    assert "running" not in adapted[0].text
    assert "chunk_budget" not in adapted[0].text
    assert "recovery_notice" not in adapted[0].text


@pytest.mark.parametrize(
    ("start_label", "end_label"),
    [
        ("RESTAGE START observation.", "RESTAGE END/latest observation."),
        (
            "RETURN-TO-HOME START observation.",
            "RETURN-TO-HOME END/latest observation.",
        ),
    ],
)
def test_visual_evidence_adapter_supports_recovery_labels(
    start_label: str,
    end_label: str,
) -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    content = [
        TextContent(text='{"status": "success"}'),
        TextContent(text=start_label),
        ImageContent(data=png, mime_type="image/png"),
        ImageContent(data=png, mime_type="image/png"),
        TextContent(text=end_label),
        ImageContent(data=png, mime_type="image/png"),
        ImageContent(data=png, mime_type="image/png"),
    ]

    adapted = build_reflection_visual_evidence(content)

    labels = [item.text for item in adapted if isinstance(item, TextContent)]
    assert json.loads(adapted[0].text)["status"] == "success"
    assert sum(isinstance(item, ImageContent) for item in adapted) == 6
    assert labels[2].startswith("START external")
    assert labels[3].startswith("CHANGE HIGHLIGHT over the END external")
    assert labels[4].startswith("END external")
    assert labels[5].startswith("START wrist")
    assert labels[6].startswith("CHANGE HIGHLIGHT over the END wrist")
    assert labels[7].startswith("END wrist")


def test_recovery_state_only_counts_vla_attempts_per_step() -> None:
    state = RecoveryState()

    assert state.record_vla_attempt("step-1") == 1
    assert state.record_vla_attempt("step-1") == 2
    assert state.attempts_for("step-1") == 2
    assert state.attempts_for("step-2") == 0

    assert state.record_vla_attempt("step-1") == 3
    assert state.attempts_for("step-1") == 3

    assert state.record_vla_attempt("step-2") == 1
    assert state.attempts_for("step-1") == 0
    state.clear()
    assert state.attempts_for("step-2") == 0


def test_episode_memory_is_appended_once_and_not_derived_from_transcript() -> None:
    memory = EpisodeMemory()
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="execute_vla",
        content=[TextContent(text="large raw result")],
        details={"status": "running", "chunks_used": 3},
    )

    assert memory.summary() == []
    memory.compact_transcript_for_model([result, result])
    assert memory.summary() == []

    for _ in range(2):
        memory.append_tool_event(
            call_id="call-1",
            tool="execute_vla",
            arguments={"step_id": "step-1"},
            details=result.details,
            is_error=False,
        )
    assert memory.summary() == [
        {
            "call_id": "call-1",
            "tool": "execute_vla",
            "step_id": "step-1",
            "status": "running",
            "chunks_used": 3,
        }
    ]


def test_default_prompt_requires_task_plan_before_execute_vla() -> None:
    system_prompt_text = (
        files("roboagent.prompts").joinpath("system.md").read_text(encoding="utf-8").strip()
    )
    reflection_prompt_text = (
        files("roboagent.prompts")
        .joinpath("state_reflection.md")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert system_prompt_text == DEFAULT_SYSTEM_PROMPT
    assert "Use one provided tool at a time" in DEFAULT_SYSTEM_PROMPT
    assert "You own the TaskPlan" in DEFAULT_SYSTEM_PROMPT
    assert "include a concise `decision_summary`" in DEFAULT_SYSTEM_PROMPT
    assert "When receiving a new task, prioritize creating the TaskPlan" in (
        DEFAULT_SYSTEM_PROMPT
    )
    assert "RoboMemArena π0.5" in DEFAULT_SYSTEM_PROMPT
    assert "TaskPlan step boundaries, order, and VLA instruction wording" in (
        DEFAULT_SYSTEM_PROMPT
    )
    assert "checkpoint-era language rather than task truth" in DEFAULT_SYSTEM_PROMPT
    assert "current running instruction's observable end state" in DEFAULT_SYSTEM_PROMPT
    assert "advance exactly that step before executing the next one" in DEFAULT_SYSTEM_PROMPT
    assert "later steps or the whole task" in DEFAULT_SYSTEM_PROMPT
    assert "stop_reason=chunk_budget" in DEFAULT_SYSTEM_PROMPT
    assert "stop_reason=gripper_checkpoint" not in DEFAULT_SYSTEM_PROMPT
    assert "This result does not change the TaskPlan" in DEFAULT_SYSTEM_PROMPT
    assert "The VLA is primarily for contact-rich operations" in DEFAULT_SYSTEM_PROMPT
    assert "Recovery tools are optional" in DEFAULT_SYSTEM_PROMPT
    assert "not failure judgments" in DEFAULT_SYSTEM_PROMPT
    assert "by themselves never require a particular recovery" in DEFAULT_SYSTEM_PROMPT
    assert "An Active Skill may require one recovery tool" in DEFAULT_SYSTEM_PROMPT
    assert "prefer `restage_vla`" not in DEFAULT_SYSTEM_PROMPT
    assert "call `return_to_home` at most once" not in DEFAULT_SYSTEM_PROMPT
    assert "Do not assume hidden state" in DEFAULT_SYSTEM_PROMPT
    assert "An attempted action or gripper event alone does not prove completion" in (
        DEFAULT_SYSTEM_PROMPT
    )
    assert "single ordinary Observer" in DEFAULT_SYSTEM_PROMPT
    assert "low-priority temporal context" in DEFAULT_SYSTEM_PROMPT
    assert "multiple consecutive windows" in DEFAULT_SYSTEM_PROMPT
    assert "physically selected gripper keyframe" in DEFAULT_SYSTEM_PROMPT
    assert "requested object is visibly at the requested destination" not in (
        DEFAULT_SYSTEM_PROMPT
    )
    assert "re-examine the available before/after evidence" in DEFAULT_SYSTEM_PROMPT
    assert "physical state uncertain" in DEFAULT_SYSTEM_PROMPT
    assert "multiple consecutive windows" in reflection_prompt_text
    assert "re-examine the supplied start/end comparison" in reflection_prompt_text
    assert "report the step state as uncertain" in reflection_prompt_text
    assert "explicitly state what the available Observer" in reflection_prompt_text
    assert "Never claim that there is no conflict" in reflection_prompt_text


def test_tool_descriptions_only_define_each_tool_function() -> None:
    assert "exactly one running step" in CreateTaskPlanTool.description
    assert "pick-and-place" not in CreateTaskPlanTool.description
    assert "Does not move the robot" in AdvanceTaskPlanTool.description
    assert "earlier completed" in RepairTaskPlanTool.description
    assert "Does not move the robot" in RepairTaskPlanTool.description
    assert "Runs VLA action chunks" in ExecuteVlaTool.description
    assert "TaskPlan" not in ExecuteVlaTool.description
    assert "pending" not in ExecuteVlaTool.description
    assert "stop_reason" not in ExecuteVlaTool.description
    assert "contact-free" in AdjustEePoseTool.description
    assert "Does not grasp, place" in AdjustEePoseTool.description
    assert "execute_vla" not in AdjustEePoseTool.description
    assert "TaskPlan" not in AdjustEePoseTool.description
    assert "does not prove alignment" in CloseGripperTool.description
    assert "before pulling" in CloseGripperTool.description
    assert "does not prove that an object was safely released" in (
        OpenGripperTool.description
    )
    assert "TaskPlan" not in OpenGripperTool.description
    assert "5 cm" in RestageVlaTool.description
    assert "preserving the current gripper command" in RestageVlaTool.description
    assert "magnitude" not in RestageVlaTool.parameters["properties"]
    assert "seven joint positions recorded at the start" in ReturnToHomeTool.description
    assert "joint-space target" in ReturnToHomeTool.description
    assert "pose" not in ReturnToHomeTool.parameters["properties"]
    assert "does not advance the environment" in ObserveTool.description
    assert "TaskPlan" not in FinishTool.description
    for tool in (
        ExecuteVlaTool,
        AdjustEePoseTool,
        CloseGripperTool,
        RestageVlaTool,
        OpenGripperTool,
        ReturnToHomeTool,
        AdvanceTaskPlanTool,
        RepairTaskPlanTool,
    ):
        assert "decision_summary" in tool.parameters["properties"]


def test_robomemarena_primitive_reference_uses_checkpoint_labels() -> None:
    reference = task_primitive_reference(
        "Pick and place cookies into the basket, then pick and place tomato sauce "
        "into the same basket."
    )

    assert reference is not None
    assert "task 1, advisory" in reference
    assert "1. pick cookies" in reference
    assert "2. place cookies into basket" in reference
    assert "3. pick tomato sauce" in reference
    assert "4. place tomato into basket" in reference
    assert task_primitive_reference("A non-RoboMemArena task") is None


def test_robomemarena_task11_reference_preserves_butter_identity() -> None:
    reference = task_primitive_reference(
        "Put cookies into the top drawer and put butter into another drawer."
    )

    assert reference is not None
    assert "pick butter" in reference
    assert "place butter" in reference
    assert "chocolate" not in reference


def test_execute_vla_labels_keyframe_images_before_latest_observation() -> None:
    async def execute(step_id, cancel_event):
        del step_id, cancel_event
        start = Observation(
            rgb=ImageFrame(data="start-external"),
            wrist_rgb=ImageFrame(data="start-wrist"),
        )
        keyframe_observation = Observation(
            rgb=ImageFrame(data="keyframe-external"),
            wrist_rgb=ImageFrame(data="keyframe-wrist"),
            robot_state={
                "eef_position": [0.1, 0.2, 0.3],
                "gripper_qpos": [0.01, -0.01],
                "gripper_command": 1.0,
                "openpi_state": [9.0] * 8,
            },
        )
        visual = VisualMemory(
            observations=("The object appears held.",),
            keyframes=(
                VisualKeyframe(
                    sequence=12,
                    event="gripper_close",
                    rgb=keyframe_observation.rgb,
                    wrist_rgb=keyframe_observation.wrist_rgb,
                    observation="The object appears held.",
                ),
            ),
        )
        latest = Observation(
            rgb=ImageFrame(data="latest-external"),
            wrist_rgb=ImageFrame(data="latest-wrist"),
            robot_state={
                "eef_position": [0.4, 0.5, 0.6],
                "gripper_qpos": [0.001, -0.001],
                "gripper_command": -1.0,
            },
        )
        execution = ExecutionDiagnostics(
            start=ExecutionState(
                sequence=8,
                robot_state={
                    "eef_position": [0.0, 0.0, 0.0],
                    "gripper_qpos": [0.04, -0.04],
                    "gripper_command": -1.0,
                },
            ),
            gripper_events=(
                GripperEventRecord(
                    sequence=12,
                    command_sequence=8,
                    event="gripper_close",
                    command=1.0,
                    robot_state={
                        "eef_position": [0.1, 0.2, 0.3],
                        "gripper_qpos": [0.01, -0.01],
                        "gripper_command": 1.0,
                    },
                ),
            ),
            end=ExecutionState(sequence=12, robot_state=latest.robot_state),
            chunks_used=1,
            stop_reason="gripper_checkpoint",
        )
        memory = VlaWorkingMemory(
            visual=visual,
            execution=execution,
            start_observation=start,
        )
        return {
            "status": "running",
            "active_step": {
                "id": "step-1",
                "instruction": "pick up the object",
                "status": "running",
            },
            "plan_progress": {
                "completed": [],
                "running": "step-1",
                "pending": [],
            },
            "working_memory": memory.summary(),
        }, latest, memory

    async def run() -> None:
        result = await ExecuteVlaTool(execute).execute("call-1", {"step_id": "step-1"})

        assert [type(item) for item in result.content] == [
            TextContent,
            TextContent,
            ImageContent,
            ImageContent,
            TextContent,
            ImageContent,
            ImageContent,
            TextContent,
        ]
        assert "start observation at sequence 8" in result.content[1].text
        assert "end/latest observation at sequence 12" in result.content[4].text
        assert [result.content[index].data for index in (2, 3, 5, 6)] == [
            "start-external",
            "start-wrist",
            "latest-external",
            "latest-wrist",
        ]
        tool_payload = json.loads(result.content[0].text)
        assert tool_payload["current_step"]["id"] == "step-1"
        assert tool_payload["execution_diagnostics"] == (
            result.details["working_memory"]["execution"]
        )
        assert tool_payload["latest_observation"]["robot_state"] == {
            "eef_position": [0.4, 0.5, 0.6],
            "gripper_qpos": [0.001, -0.001],
            "gripper_command": -1.0,
            "gripper_command_state": "open",
            "gripper_width_m": pytest.approx(0.002),
        }
        assert "working_memory" not in tool_payload
        assert "The object appears held." not in result.content[0].text
        assert result.content[-1].text == (
            "Observer advisory — may contain visual recognition errors.\n"
            "Use only as a temporal hint after examining the direct images.\n\n"
            "- Keyframe sequence 12: The object appears held."
        )
        assert result.details["observation"]["robot_state"] == (
            tool_payload["latest_observation"]["robot_state"]
        )
        keyframe = result.details["working_memory"]["execution"]["gripper_events"][0]
        assert keyframe["robot_state"] == {
            "eef_position": [0.1, 0.2, 0.3],
            "gripper_qpos": [0.01, -0.01],
            "gripper_command": 1.0,
        }
        visual_keyframe = result.details["working_memory"]["visual"]["keyframes"][0]
        assert "robot_state" not in visual_keyframe
        assert "views" not in keyframe
        assert "keyframe-external" not in json.dumps(result.details)

    asyncio.run(run())


def test_execute_vla_labels_latest_images_without_keyframes() -> None:
    async def execute(step_id, cancel_event):
        del step_id, cancel_event
        latest = Observation(rgb=ImageFrame(data="latest-external"))
        execution = ExecutionDiagnostics(
            end=ExecutionState(sequence=21, robot_state={}),
        )
        memory = VlaWorkingMemory(execution=execution)
        return {"status": "running"}, latest, memory

    async def run() -> None:
        result = await ExecuteVlaTool(execute).execute("call-1", {"step_id": "step-1"})

        assert [type(item) for item in result.content] == [
            TextContent,
            TextContent,
            ImageContent,
        ]
        assert "end/latest observation at sequence 21" in result.content[1].text
        assert result.content[2].data == "latest-external"

    asyncio.run(run())


@dataclass
class FakeRobot:
    observation: Observation
    success_after: int = 1
    action_calls: list[ActionChunk] = field(default_factory=list)
    active_calls: int = 0
    max_active_calls: int = 0
    observation_sink: object | None = None
    home_calls: list[float] = field(default_factory=list)

    async def observe(self, cancel_event=None):
        del cancel_event
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0)
            return self.observation
        finally:
            self.active_calls -= 1

    async def execute_action_chunk(self, actions, cancel_event=None):
        del cancel_event
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0)
            self.action_calls.append(actions)
            self.observation.task_success = len(self.action_calls) >= self.success_after
            if self.observation_sink is not None:
                self.observation_sink.publish(self.observation)
            await asyncio.sleep(0)
            return ExecutionFeedback(
                status="success" if self.observation.task_success else "running",
                observation=self.observation,
                executed_steps=2,
            )
        finally:
            self.active_calls -= 1

    async def return_to_home(self, gripper_command, cancel_event=None):
        del cancel_event
        self.home_calls.append(float(gripper_command))
        state = dict(self.observation.robot_state)
        state["gripper_command"] = float(gripper_command)
        self.observation = Observation(robot_state=state, task_success=False)
        return ExecutionFeedback(
            status="running",
            observation=self.observation,
            executed_steps=4,
            details={
                "status": "success",
                "controller": "JOINT_POSITION",
                "joint_positions_before": [0.1] * 7,
                "joint_positions_target": [0.0] * 7,
                "joint_positions_after": [0.0] * 7,
                "max_joint_error_rad": 0.0,
                "home_pose_reached": True,
            },
        )


class FakeVla:
    def __init__(self):
        self.instructions: list[str] = []

    async def infer(self, instruction, observation, cancel_event=None):
        del observation, cancel_event
        self.instructions.append(instruction)
        return ActionChunk(actions=[[0.1, 0.2]])


@dataclass
class OscFakeRobot(FakeRobot):
    """只为 Runtime 测试实现当前 OSC action 的确定性一步响应。"""

    async def execute_action_chunk(self, actions, cancel_event=None):
        if actions.metadata.get("source") not in {
            "adjust_ee_pose",
            "restage_vla",
        }:
            return await super().execute_action_chunk(actions, cancel_event)
        del cancel_event
        self.action_calls.append(actions)
        action = actions.actions[0]
        state = dict(self.observation.robot_state)
        position = list(state["eef_position"])
        axis_angle = list(state["eef_axis_angle"])
        state["eef_position"] = [
            position[index] + action[index] * 0.05 for index in range(3)
        ]
        state["eef_axis_angle"] = [
            axis_angle[index] + action[index + 3] * 0.5 for index in range(3)
        ]
        state["gripper_command"] = action[6]
        self.observation = Observation(
            rgb=ImageFrame(data="adjusted-external"),
            wrist_rgb=ImageFrame(data="adjusted-wrist"),
            robot_state=state,
            task_success=False,
        )
        if self.observation_sink is not None:
            self.observation_sink.publish(self.observation)
        await asyncio.sleep(0)
        return ExecutionFeedback(
            status="running",
            observation=self.observation,
            executed_steps=1,
        )


class FakeFinalizer:
    def __init__(self) -> None:
        self.calls = 0

    async def finalize_episode(self, cancel_event=None):
        del cancel_event
        self.calls += 1
        return Observation(task_success=True)


class FakeObserver:
    def __init__(self, robot: FakeRobot):
        self.robot = robot
        self.calls: list[tuple[str, str, list[Observation]]] = []

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
            cancel_event,
            anchor_index,
            gripper_transition_index,
            previous_observation,
            previous_instruction,
        )
        self.calls.append((task, instruction, list(observations)))
        return ObserverAssessment(
            keyframe=None,
            observation="The gripper moved toward the mug.",
        )


class FailingObserver:
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
        raise ConnectionError("observer is offline")


class BlockingObserver:
    def __init__(self, robot: FakeRobot):
        self.robot = robot
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

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
        self.calls += 1
        if self.calls == 1:
            assert len(self.robot.action_calls) == 1
            self.started.set()
            await self.release.wait()
        return ObserverAssessment(None, "The robot is moving toward the target.")


def plan_response(
    instruction: str,
    *,
    step_id: str = "step-1",
    pending_instruction: str | None = None,
):
    steps = [
        {
            "id": step_id,
            "instruction": instruction,
            "status": "running",
        }
    ]
    if pending_instruction is not None:
        steps.append(
            {
                "id": "step-2",
                "instruction": pending_instruction,
                "status": "pending",
            }
        )
    return faux_assistant_message(
        faux_tool_call("create_task_plan", {"steps": steps}, tool_call_id="create-plan"),
        stop_reason="toolUse",
    )


def make_agent(
    robot,
    vla,
    *responses,
    observer=None,
    finalizer=None,
    max_vla_chunks=24,
    include_observer_text=True,
    auto_reflections=True,
    max_turns=12,
    task_execution_skill=None,
    ee_pose_adjustment=None,
    plan_advance_approval=None,
):
    models = create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    scripted_responses = []
    for response in responses:
        scripted_responses.append(response)
        if (
            auto_reflections
            and isinstance(response, AssistantMessage)
            and any(
                getattr(block, "name", None)
                in {
                    "execute_vla",
                    "restage_vla",
                    "adjust_ee_pose",
                    "return_to_home",
                }
                for block in response.content
            )
        ):
            scripted_responses.append(
                faux_assistant_message(
                    "Grounded State Reflection: The latest physical evidence was "
                    "reviewed. The current step status is uncertain. No important "
                    "conflict is visible."
                )
            )
    faux.set_responses(scripted_responses)
    models.set_provider(faux.provider)
    observer_runner = None
    if observer is not None:
        sink, source = create_observation_stream()
        robot.observation_sink = sink
        observer_runner = VisualObserver(
            observer,
            source,
            sampling=ObserverSamplingOptions(max_timesteps=5, min_new_observations=1),
        )
    options = RoboAgentOptions(
        model=model,
        robot=robot,
        vla=vla,
        stream_fn=models.stream_simple,
        system_prompt="Use one robot tool at a time. Finish only after observing the result.",
        stream_options=SimpleStreamOptions(tool_choice="required"),
        observer=observer_runner,
        include_observer_text=include_observer_text,
        finalizer=finalizer,
        max_vla_chunks=max_vla_chunks,
        max_turns=max_turns,
        task_execution_skill=task_execution_skill,
        plan_advance_approval=plan_advance_approval,
    )
    if ee_pose_adjustment is not None:
        options.ee_pose_adjustment = ee_pose_adjustment
    return RoboAgent(options)


def test_physical_tool_inserts_grounded_reflection_before_decision() -> None:
    async def run() -> None:
        captured: list[str] = []
        png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )

        def reflect(context, options, state, model):
            del state, model
            assert context.tools is None
            assert options.tool_choice is None
            assert "state-reflection turn" in context.system_prompt
            reflection_text = "\n".join(
                item.text
                for message in context.messages
                for item in getattr(message, "content", [])
                if isinstance(item, TextContent)
            )
            assert "chunk_budget" not in reflection_text
            assert '"status": "running"' not in reflection_text
            assert "gripper_command > 0 means close" in reflection_text
            assert "gripper_command_state" in context.system_prompt
            assert "partial motion" in context.system_prompt
            assert sum(
                isinstance(item, ImageContent)
                for message in context.messages
                for item in getattr(message, "content", [])
            ) == 6
            captured.append("reflection")
            return faux_assistant_message(
                "Grounded State Reflection: Compared with the start view, the drawer "
                "is visibly farther outward. The current step appears complete because "
                "the latest external view shows an opening gap. No important conflict "
                "is visible."
            )

        def decide(context, options, state, model):
            del state, model
            assert context.tools is not None
            assert {tool.name for tool in context.tools} >= {
                "execute_vla",
                "advance_task_plan",
                "repair_task_plan",
                "adjust_ee_pose",
                "finish",
            }
            assert options.tool_choice == "required"
            assert isinstance(context.messages[-1], UserMessage)
            assert "Latest Grounded State Reflection:" in context.messages[-1].content
            captured.append("decision")
            return faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Reflection path checked."},
                ),
                stop_reason="toolUse",
            )

        agent = make_agent(
            FakeRobot(
                Observation(
                    rgb=ImageFrame(data=png),
                    wrist_rgb=ImageFrame(data=png),
                    task_success=False,
                )
            ),
            FakeVla(),
            plan_response("open the drawer"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            reflect,
            decide,
            auto_reflections=False,
        )

        result = await agent.run("Open the drawer")

        assert captured == ["reflection", "decision"]
        assert result.turns == 4
        assert result.details["state_reflections"] == [
            {
                "after_call_id": result.details["tool_trace"][1]["call_id"],
                "tool": "execute_vla",
                "text": (
                    "Grounded State Reflection: Compared with the start view, the drawer "
                    "is visibly farther outward. The current step appears complete because "
                    "the latest external view shows an opening gap. No important conflict "
                    "is visible."
                ),
            }
        ]

    asyncio.run(run())


def test_active_skill_is_visible_to_decisions_but_not_reflection() -> None:
    async def run() -> None:
        marker = "UNIQUE_ACTIVE_SKILL_MARKER"

        def create_plan(context, options, state, model):
            del options, state, model
            assert marker in context.system_prompt
            return plan_response("open the drawer")

        def reflect(context, options, state, model):
            del options, state, model
            assert marker not in context.system_prompt
            return faux_assistant_message(
                "Grounded State Reflection: The physical change is uncertain. The "
                "current step is uncertain. No conflict is visible."
            )

        def decide(context, options, state, model):
            del options, state, model
            assert marker in context.system_prompt
            return faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Skill visibility checked."},
                ),
                stop_reason="toolUse",
            )

        skill = TaskExecutionSkill(
            skill_id="embodied-task-execution",
            version="v1",
            content=marker,
            content_sha256="hash",
            source_path="/tmp/v1/SKILL.md",
        )
        agent = make_agent(
            FakeRobot(Observation(task_success=False)),
            FakeVla(),
            create_plan,
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            reflect,
            decide,
            auto_reflections=False,
            task_execution_skill=skill,
        )

        result = await agent.run("Open the drawer")

        assert result.details["task_execution_skill"]["version"] == "v1"

    asyncio.run(run())


def test_run_injects_only_matching_primitive_reference() -> None:
    async def run() -> None:
        captured_text: list[str] = []

        def capture_initial_context(context, options, state, model):
            del options, state, model
            message = context.messages[-1]
            captured_text.extend(
                block.text for block in message.content if isinstance(block, TextContent)
            )
            return plan_response("pick cookies")

        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            capture_initial_context,
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": True, "summary": "Reference injection checked."},
                ),
                stop_reason="toolUse",
            ),
        )

        await agent.run(
            "Pick and place cookies into the basket, then pick and place tomato sauce "
            "into the same basket."
        )

        reference_blocks = [
            text for text in captured_text if text.startswith("RoboMemArena π0.5")
        ]
        assert len(reference_blocks) == 1
        assert "1. pick cookies" in reference_blocks[0]
        assert "4. place tomato into basket" in reference_blocks[0]
        assert "task 2" not in reference_blocks[0]

    asyncio.run(run())


def test_execute_vla_then_finish() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        vla = FakeVla()
        agent = make_agent(
            robot,
            vla,
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call(
                    "execute_vla",
                    {
                        "step_id": "step-1",
                        "decision_summary": "The running step still requires physical action.",
                    },
                    tool_call_id="execute-1",
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": True, "summary": "The mug is grasped."},
                    tool_call_id="finish-1",
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Pick up the mug")

        assert result.status == "success"
        assert len(robot.action_calls) == 1
        assert vla.instructions == ["pick up the mug"]
        assert result.tool_calls == 3
        assert "observe" not in {tool.name for tool in agent._agent.state.tools}
        trace = result.details["tool_trace"]
        assert [entry["name"] for entry in trace] == [
            "create_task_plan",
            "execute_vla",
            "finish",
        ]
        assert trace[1]["decision_summary"] == (
            "The running step still requires physical action."
        )
        episode_memory = result.details["episode_memory"]
        assert [event["tool"] for event in episode_memory] == [
            "create_task_plan",
            "execute_vla",
            "finish",
        ]
        assert episode_memory[1]["decision_summary"] == (
            "The running step still requires physical action."
        )
        execute_result = dict(trace[1]["result"])
        working_memory = execute_result.pop("working_memory")
        assert execute_result == {
            "status": "success",
            "step_id": "step-1",
            "chunks_used": 1,
            "executed_steps": 2,
            "details": {},
            "execution_status": "success",
            "task_effect_status": "verified_task_success",
            "workspace_violation": False,
            "consecutive_vla_attempts": 1,
            "active_step": {
                "id": "step-1",
                "instruction": "pick up the mug",
                "status": "running",
            },
            "plan_progress": {
                "completed": [],
                "running": "step-1",
                "pending": [],
            },
            "observation": robot.observation.summary(),
        }
        assert working_memory == {
            "visual": {"observations": [], "keyframes": []},
            "execution": {
                "start": {"sequence": None, "robot_state": {}},
                "gripper_events": [],
                "end": {"sequence": None, "robot_state": {}},
                "chunks_used": 1,
                "stop_reason": None,
            },
        }
        assert result.details["task_plan"]["steps"][0]["status"] == "running"

    asyncio.run(run())


def test_adjust_ee_pose_runs_after_vla_and_returns_latest_multimodal_state() -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                rgb=ImageFrame(data="initial-external"),
                wrist_rgb=ImageFrame(data="initial-wrist"),
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": 1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.03},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Stopped after recovery test."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Pick up the mug")

        assert result.status == "failure"
        assert [entry["name"] for entry in result.details["tool_trace"]] == [
            "create_task_plan",
            "execute_vla",
            "adjust_ee_pose",
            "finish",
        ]
        adjustment = result.details["tool_trace"][2]
        assert adjustment["result"]["status"] == "success"
        assert adjustment["result"]["execution_status"] == "success"
        assert adjustment["result"]["task_effect_status"] == "unverified"
        assert adjustment["result"]["applied_translation_m"] == pytest.approx(
            [0.0, 0.0, 0.03]
        )
        assert adjustment["result"]["applied_rotation_deg"] is None
        assert "working_memory" not in adjustment["result"]
        assert adjustment["result"]["observation"]["robot_state"][
            "gripper_command"
        ] == 1.0
        assert adjustment["image_count"] == 2
        assert robot.action_calls[-1].metadata == {"source": "adjust_ee_pose"}

    asyncio.run(run())


def test_adjust_ee_pose_returns_workspace_rejection_without_robot_motion() -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": 1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        workspace = EePoseAdjustmentConfig(
            workspace_min_m=(-0.2, -0.2, 0.2),
            workspace_max_m=(0.2, 0.2, 0.52),
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.03},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Blocked."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
            ee_pose_adjustment=workspace,
        )

        result = await agent.run("Pick up the mug")

        adjustment = result.details["tool_trace"][2]["result"]
        assert adjustment["status"] == "out_of_workspace"
        assert adjustment["executed_steps"] == 0
        assert adjustment["target_pose"]["position_m"] == pytest.approx([0.0, 0.0, 0.53])
        assert adjustment["workspace"] == {
            "min_m": [-0.2, -0.2, 0.2],
            "max_m": [0.2, 0.2, 0.52],
        }
        assert len(robot.action_calls) == 1

    asyncio.run(run())


def test_adjust_ee_pose_stops_when_observed_pose_crosses_workspace() -> None:
    @dataclass
    class DriftingRobot(OscFakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            if actions.metadata.get("source") != "adjust_ee_pose":
                return await super().execute_action_chunk(actions, cancel_event)
            del cancel_event
            self.action_calls.append(actions)
            state = dict(self.observation.robot_state)
            state["eef_position"] = [0.0, 0.0, 0.53]
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
            )

    async def run() -> None:
        robot = DriftingRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": 1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.01},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Stopped."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
            ee_pose_adjustment=EePoseAdjustmentConfig(
                workspace_min_m=(-0.2, -0.2, 0.2),
                workspace_max_m=(0.2, 0.2, 0.52),
            ),
        )

        result = await agent.run("Pick up the mug")

        adjustment = result.details["tool_trace"][2]["result"]
        assert adjustment["status"] == "out_of_workspace"
        assert adjustment["executed_steps"] == 1
        assert "motion was stopped" in adjustment["reason"]
        assert len(robot.action_calls) == 2

    asyncio.run(run())


def test_execute_vla_stops_remaining_chunk_after_observed_workspace_violation() -> None:
    @dataclass
    class BoundaryRobot(FakeRobot):
        crossed_workspace_once: bool = False

        async def execute_action_chunk(self, actions, cancel_event=None):
            del cancel_event
            assert len(actions.actions) == 1
            self.action_calls.append(actions)
            if self.crossed_workspace_once or len(self.action_calls) == 1:
                x_position = 0.1
            else:
                x_position = 0.21
                self.crossed_workspace_once = True
            state = dict(self.observation.robot_state)
            state["eef_position"] = [x_position, 0.0, 0.5]
            state["gripper_command"] = actions.actions[0][6]
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
            )

        async def return_to_home(self, gripper_command, cancel_event=None):
            del cancel_event
            self.home_calls.append(float(gripper_command))
            state = dict(self.observation.robot_state)
            state["eef_position"] = [0.0, 0.0, 0.5]
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=4,
                details={
                    "status": "success",
                    "controller": "JOINT_POSITION",
                    "home_pose_reached": True,
                },
            )

    class ThreeActionVla(FakeVla):
        async def infer(self, instruction, observation, cancel_event=None):
            del observation, cancel_event
            self.instructions.append(instruction)
            return ActionChunk(actions=[[0.0] * 6 + [-1.0] for _ in range(3)])

    async def run() -> None:
        robot = BoundaryRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        vla = ThreeActionVla()
        agent = make_agent(
            robot,
            vla,
            plan_response("move safely"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "This must be blocked while unsafe."},
                    tool_call_id="unsafe-finish",
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="safe-home"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Stopped."}),
                stop_reason="toolUse",
            ),
            max_turns=16,
            max_vla_chunks=1,
            ee_pose_adjustment=EePoseAdjustmentConfig(
                workspace_min_m=(-0.2, -0.2, 0.2),
                workspace_max_m=(0.2, 0.2, 0.8),
            ),
        )

        result = await agent.run("Move safely")

        execution = result.details["tool_trace"][1]["result"]
        assert execution["status"] == "out_of_workspace"
        assert execution["chunks_used"] == 1
        assert execution["executed_steps"] == 2
        assert execution["eef_pose"]["position_m"] == pytest.approx(
            [0.21, 0.0, 0.5]
        )
        assert execution["workspace"] == {
            "min_m": [-0.2, -0.2, 0.2],
            "max_m": [0.2, 0.2, 0.8],
        }
        assert execution["workspace_violation"] is True
        unsafe_finish = result.details["tool_trace"][2]
        assert unsafe_finish["status"] == "blocked"
        assert "allows only return_to_home" in unsafe_finish["reason"]
        home = result.details["tool_trace"][3]["result"]
        assert home["status"] == "success"
        assert home["workspace_violation"] is False
        resumed_execution = result.details["tool_trace"][4]["result"]
        assert resumed_execution["status"] == "running"
        assert resumed_execution["workspace_violation"] is False
        assert result.details["workspace_violation"] is False
        assert result.details["safety_stop"] is False
        assert len(robot.action_calls) == 5
        assert all(len(chunk.actions) == 1 for chunk in robot.action_calls)
        assert vla.instructions == ["move safely", "move safely"]

    asyncio.run(run())


def test_failed_home_after_workspace_violation_stops_episode() -> None:
    @dataclass
    class FailedHomeRobot(FakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            del cancel_event
            self.action_calls.append(actions)
            state = dict(self.observation.robot_state)
            state["eef_position"] = [0.21, 0.0, 0.5]
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
            )

        async def return_to_home(self, gripper_command, cancel_event=None):
            del cancel_event
            self.home_calls.append(float(gripper_command))
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
                details={
                    "status": "controller_failed",
                    "controller": "JOINT_POSITION",
                    "home_pose_reached": False,
                },
            )

    class OneActionVla(FakeVla):
        async def infer(self, instruction, observation, cancel_event=None):
            del observation, cancel_event
            self.instructions.append(instruction)
            return ActionChunk(actions=[[0.0] * 6 + [-1.0]])

    async def run() -> None:
        robot = FailedHomeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            OneActionVla(),
            plan_response("move safely"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="failed-home"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "This call must not execute."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
            ee_pose_adjustment=EePoseAdjustmentConfig(
                workspace_min_m=(-0.2, -0.2, 0.2),
                workspace_max_m=(0.2, 0.2, 0.8),
            ),
        )

        result = await agent.run("Move safely")

        assert result.status == "safety_stop"
        assert [entry["name"] for entry in result.details["tool_trace"]] == [
            "create_task_plan",
            "execute_vla",
            "return_to_home",
        ]
        home = result.details["tool_trace"][2]["result"]
        assert home["status"] == "controller_failed"
        assert home["safety_stop"] is True
        assert home["workspace_violation"] is True
        assert result.details["workspace_violation"] is True
        assert result.details["safety_stop"] is True
        assert robot.home_calls == [-1.0]

    asyncio.run(run())


def test_adjust_ee_pose_reuses_last_valid_gripper_command() -> None:
    @dataclass
    class CommandDroppingRobot(OscFakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            if actions.metadata.get("source") == "adjust_ee_pose":
                return await super().execute_action_chunk(actions, cancel_event)
            del cancel_event
            self.action_calls.append(actions)
            state = dict(self.observation.robot_state)
            state.pop("gripper_command", None)
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
            )

    async def run() -> None:
        robot = CommandDroppingRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.03},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": False, "summary": "Recovery tested."}
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Pick up the mug")

        adjustment = result.details["tool_trace"][2]["result"]
        assert adjustment["status"] == "success"
        assert adjustment["applied_translation_m"] == pytest.approx([0.0, 0.0, 0.03])
        assert robot.action_calls[-1].actions[0][6] == -1.0

    asyncio.run(run())


@pytest.mark.parametrize("gripper_command", [-1.0, 1.0])
def test_restage_vla_retracts_fixed_distance_and_preserves_gripper(
    gripper_command,
) -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                rgb=ImageFrame(data="initial-external"),
                wrist_rgb=ImageFrame(data="initial-wrist"),
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": gripper_command,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("open the drawer"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("restage_vla", {}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Restage checked."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Open the drawer")

        restage = result.details["tool_trace"][2]
        assert restage["name"] == "restage_vla"
        assert restage["result"]["status"] == "success"
        assert restage["result"]["requested_distance_m"] == 0.05
        assert restage["result"]["requested_magnitude"] == 0.05
        assert restage["result"]["applied_translation_m"] == pytest.approx(
            [0.0, 0.0, -0.05]
        )
        assert restage["result"]["ee_pose_before"]["position_m"] == [0.0, 0.0, 0.5]
        assert restage["result"]["ee_pose_after"]["position_m"] == pytest.approx(
            [0.0, 0.0, 0.45]
        )
        assert restage["result"]["observation"]["robot_state"][
            "gripper_command"
        ] == gripper_command
        assert restage["image_count"] == 4
        assert any("RESTAGE START" in text for text in restage["text"])
        assert any("RESTAGE END/latest" in text for text in restage["text"])
        assert robot.action_calls[-1].metadata == {"source": "restage_vla"}
        assert robot.action_calls[-1].actions[0][6] == gripper_command
        assert result.details["state_reflections"][-1]["tool"] == "restage_vla"
        restage_event = result.details["episode_memory"][2]
        assert restage_event["tool"] == "restage_vla"
        assert restage_event["requested_distance_m"] == 0.05
        assert restage_event["applied_translation_m"] == pytest.approx(
            [0.0, 0.0, -0.05]
        )

    asyncio.run(run())


def test_vla_attempt_notices_are_advisory_and_never_block_execution() -> None:
    def execute_response():
        return faux_assistant_message(
            faux_tool_call("execute_vla", {"step_id": "step-1"}),
            stop_reason="toolUse",
        )

    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                rgb=ImageFrame(data="external"),
                wrist_rgb=ImageFrame(data="wrist"),
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("open the drawer"),
            *(execute_response() for _ in range(9)),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Advisory notices checked."},
                ),
                stop_reason="toolUse",
            ),
            max_turns=24,
            max_vla_chunks=1,
        )

        result = await agent.run("Open the drawer")

        execute_entries = [
            entry
            for entry in result.details["tool_trace"]
            if entry["name"] == "execute_vla"
        ]
        assert len(execute_entries) == 9
        assert all(entry["status"] == "completed" for entry in execute_entries)
        results = [entry["result"] for entry in execute_entries]
        assert [item["consecutive_vla_attempts"] for item in results] == list(
            range(1, 10)
        )
        assert all("recovery_notice" not in results[index] for index in (0, 1))
        assert all("inspect whether" in results[index]["recovery_notice"] for index in (2, 3))
        assert all("remain optional" in results[index]["recovery_notice"] for index in (4, 5, 6))
        assert all("High VLA usage" in results[index]["recovery_notice"] for index in (7, 8))

    asyncio.run(run())


def test_return_to_home_does_not_control_vla_or_repeat_permissions() -> None:
    @dataclass
    class StalledHomeRobot(OscFakeRobot):
        async def return_to_home(self, gripper_command, cancel_event=None):
            del cancel_event
            self.home_calls.append(float(gripper_command))
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
                details={
                    "status": "blocked",
                    "controller": "JOINT_POSITION",
                    "home_pose_reached": False,
                },
            )

    def execute_response():
        return faux_assistant_message(
            faux_tool_call("execute_vla", {"step_id": "step-1"}),
            stop_reason="toolUse",
        )

    async def run() -> None:
        robot = StalledHomeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("open the drawer"),
            execute_response(),
            faux_assistant_message(
                faux_tool_call("restage_vla", {}),
                stop_reason="toolUse",
            ),
            execute_response(),
            execute_response(),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="repeat-home"),
                stop_reason="toolUse",
            ),
            execute_response(),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": False, "summary": "Gate checked."}
                ),
                stop_reason="toolUse",
            ),
            max_turns=24,
            max_vla_chunks=1,
        )

        result = await agent.run("Open the drawer")

        homes = [
            entry
            for entry in result.details["tool_trace"]
            if entry["name"] == "return_to_home"
        ]
        assert [entry["status"] for entry in homes] == ["completed", "completed"]
        assert [entry["result"]["status"] for entry in homes] == [
            "blocked",
            "blocked",
        ]
        assert all(entry["result"]["home_pose_reached"] is False for entry in homes)
        attempts = [
            entry["result"]["consecutive_vla_attempts"]
            for entry in result.details["tool_trace"]
            if entry["name"] == "execute_vla" and entry["status"] == "completed"
        ]
        assert attempts == [1, 2, 3, 4]

    asyncio.run(run())


def test_blocked_zero_displacement_restage_does_not_reset_retry_streak() -> None:
    async def run() -> None:
        robot = FakeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(robot, FakeVla(), max_vla_chunks=1)
        await agent._create_task_plan(
            [{"id": "step-1", "instruction": "open drawer", "status": "running"}],
            None,
        )
        for _ in range(3):
            await agent._execute_vla("step-1", None)

        payload, _, _ = await agent._restage_vla(None)

        assert payload["status"] == "blocked"
        assert payload["applied_translation_m"] == [0.0, 0.0, 0.0]
        assert agent._recovery_state.attempts_for("step-1") == 3

    asyncio.run(run())


def test_restage_vla_is_blocked_before_current_step_executes_vla() -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            )
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("open the drawer"),
            faux_assistant_message(
                faux_tool_call("restage_vla", {}, tool_call_id="premature-restage"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "No VLA action was attempted."},
                ),
                stop_reason="toolUse",
            ),
            auto_reflections=False,
        )

        result = await agent.run("Open the drawer")

        restage = result.details["tool_trace"][1]
        assert restage["status"] == "blocked"
        assert "only after execute_vla" in restage["reason"]
        assert robot.action_calls == []

    asyncio.run(run())


def test_return_to_home_is_available_after_first_vla_attempt() -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.001645,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("open the drawer"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="early-home"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Early home was accepted."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Open the drawer")

        home = result.details["tool_trace"][2]
        assert home["name"] == "return_to_home"
        assert home["status"] == "completed"
        assert home["result"]["status"] == "success"
        assert home["result"]["controller"] == "JOINT_POSITION"
        assert home["result"]["joint_positions_target"] == [0.0] * 7
        assert home["result"]["home_pose_reached"] is True
        assert robot.home_calls == [-1.0]

    asyncio.run(run())


def test_return_to_home_is_available_once_immediately_after_plan_repair() -> None:
    async def run() -> None:
        robot = FakeRobot(
            Observation(
                robot_state={"gripper_command": -1.0},
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response(
                "pick milk",
                step_id="pick",
                pending_instruction="pour milk",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "pick"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "pick", "next_step_id": "step-2"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-2"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "repair_task_plan",
                    {"running_step_id": "step-2", "reopen_step_id": "pick"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="repair-home"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("return_to_home", {}, tool_call_id="repeat-home"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": False, "summary": "Gate checked."}
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Pick milk, then pour it")

        homes = [
            entry
            for entry in result.details["tool_trace"]
            if entry["name"] == "return_to_home"
        ]
        assert [entry["status"] for entry in homes] == ["completed", "blocked"]
        assert homes[0]["result"]["status"] == "success"
        assert "only after execute_vla" in homes[1]["reason"]
        assert robot.home_calls == [-1.0]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("tool_name", "command", "qpos_before", "qpos_after"),
    [
        ("open_gripper", -1.0, [0.01, -0.01], [0.04, -0.04]),
        ("close_gripper", 1.0, [0.04, -0.04], [0.01, -0.01]),
    ],
)
def test_gripper_tool_changes_state_after_vla_attempt(
    tool_name: str,
    command: float,
    qpos_before: list[float],
    qpos_after: list[float],
) -> None:
    @dataclass
    class GripperRecoveryRobot(FakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            if actions.metadata.get("source") != tool_name:
                return await super().execute_action_chunk(actions, cancel_event)
            del cancel_event
            self.action_calls.append(actions)
            state = dict(self.observation.robot_state)
            state["gripper_command"] = actions.actions[-1][6]
            state["gripper_qpos"] = qpos_after
            self.observation = Observation(
                rgb=ImageFrame(data="opened-external"),
                wrist_rgb=ImageFrame(data="opened-wrist"),
                robot_state=state,
                task_success=False,
            )
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=len(actions.actions),
            )

    async def run() -> None:
        robot = GripperRecoveryRobot(
            Observation(
                robot_state={
                    "gripper_command": -command,
                    "gripper_qpos": qpos_before,
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up and place the mug in the basket"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(tool_name, {}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": False, "summary": "Recovery tested."}
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Pick up and place the mug in the basket")

        assert [entry["name"] for entry in result.details["tool_trace"]] == [
            "create_task_plan",
            "execute_vla",
            tool_name,
            "finish",
        ]
        recovery = result.details["tool_trace"][2]
        assert recovery["result"]["status"] == "success"
        assert recovery["result"]["execution_status"] == "success"
        assert recovery["result"]["task_effect_status"] == "unverified"
        assert recovery["result"]["command"] == command
        assert recovery["result"]["executed_steps"] == 5
        assert recovery["result"]["gripper_qpos_before"] == qpos_before
        assert recovery["result"]["gripper_qpos_after"] == qpos_after
        assert recovery["result"]["observation"]["robot_state"][
            "gripper_command"
        ] == command
        assert recovery["image_count"] == 2
        gripper_actions = [
            chunk
            for chunk in robot.action_calls
            if chunk.metadata == {"source": tool_name}
        ]
        assert len(gripper_actions) == 5
        assert all(len(chunk.actions) == 1 for chunk in gripper_actions)
        assert all(chunk.actions[0][:6] == [0.0] * 6 for chunk in gripper_actions)
        assert all(chunk.actions[0][6] == command for chunk in gripper_actions)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("tool_name", "command"),
    [("open_gripper", -1.0), ("close_gripper", 1.0)],
)
def test_gripper_tool_stops_after_observed_workspace_violation(
    tool_name: str,
    command: float,
) -> None:
    @dataclass
    class GripperDriftRobot(FakeRobot):
        gripper_steps: int = 0

        async def execute_action_chunk(self, actions, cancel_event=None):
            if actions.metadata.get("source") != tool_name:
                del cancel_event
                self.action_calls.append(actions)
                return ExecutionFeedback(
                    status="running",
                    observation=self.observation,
                    executed_steps=1,
                )
            del cancel_event
            self.action_calls.append(actions)
            self.gripper_steps += 1
            state = dict(self.observation.robot_state)
            state["eef_position"] = [
                0.1 if self.gripper_steps == 1 else 0.21,
                0.0,
                0.5,
            ]
            state["gripper_command"] = command
            self.observation = Observation(robot_state=state, task_success=False)
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=1,
            )

    async def run() -> None:
        robot = GripperDriftRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -command,
                    "gripper_qpos": [0.02, -0.02],
                },
                task_success=False,
            ),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("grasp the handle"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(tool_name, {}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Workspace guard tested."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
            ee_pose_adjustment=EePoseAdjustmentConfig(
                workspace_min_m=(-0.2, -0.2, 0.2),
                workspace_max_m=(0.2, 0.2, 0.8),
            ),
        )

        result = await agent.run("Grasp the handle")

        recovery = result.details["tool_trace"][2]["result"]
        assert recovery["status"] == "out_of_workspace"
        assert recovery["execution_status"] == "out_of_workspace"
        assert recovery["task_effect_status"] == "unverified"
        assert recovery["executed_steps"] == 2
        assert recovery["eef_pose"]["position_m"] == [0.21, 0.0, 0.5]
        assert robot.gripper_steps == 2
        assert len(
            [
                chunk
                for chunk in robot.action_calls
                if chunk.metadata == {"source": tool_name}
            ]
        ) == 2

    asyncio.run(run())


@pytest.mark.parametrize("tool_name", ["open_gripper", "close_gripper"])
def test_gripper_tool_is_blocked_before_current_step_executes_vla(
    tool_name: str,
) -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up and place the mug in the basket"),
            faux_assistant_message(
                faux_tool_call(tool_name, {}, tool_call_id="premature-gripper"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "No VLA action was attempted."},
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Pick up and place the mug in the basket")

        recovery = result.details["tool_trace"][1]
        assert recovery["status"] == "blocked"
        assert "only after execute_vla" in recovery["reason"]
        assert robot.action_calls == []

    asyncio.run(run())


def test_adjust_ee_pose_is_blocked_before_current_step_executes_vla() -> None:
    async def run() -> None:
        robot = OscFakeRobot(
            Observation(
                robot_state={
                    "eef_position": [0.0, 0.0, 0.5],
                    "eef_axis_angle": [0.0, 0.0, 0.0],
                    "gripper_command": -1.0,
                },
                task_success=False,
            )
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.03},
                    tool_call_id="premature-adjustment",
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "No VLA action was attempted."},
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Pick up the mug")

        adjustment = result.details["tool_trace"][1]
        assert adjustment["status"] == "blocked"
        assert "only after execute_vla" in adjustment["reason"]
        assert robot.action_calls == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("robot_state", "expected_status"),
    [
        ({"gripper_command": 1.0}, "controller_failed"),
        (
            {
                "eef_position": [0.0, 0.0, 0.5],
                "eef_axis_angle": [0.0, 0.0, 0.0],
                "gripper_command": 1.0,
            },
            "blocked",
        ),
    ],
)
def test_adjust_ee_pose_reports_missing_state_and_stalled_controller(
    robot_state, expected_status
) -> None:
    async def run() -> None:
        robot = FakeRobot(
            Observation(robot_state=robot_state, task_success=False),
            success_after=99,
        )
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "adjust_ee_pose",
                    {"action": "world_z_up", "magnitude": 0.03},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "Recovery could not continue."},
                ),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Pick up the mug")

        adjustment = result.details["tool_trace"][2]["result"]
        assert adjustment["status"] == expected_status

    asyncio.run(run())


def test_text_only_stop_gets_finish_follow_up() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            faux_assistant_message("The task is not complete yet."),
            faux_assistant_message("I am still stopping without a tool."),
            faux_assistant_message(
                faux_tool_call(
                    "finish",
                    {"success": False, "summary": "The task is incomplete."},
                    tool_call_id="finish-follow-up",
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Try the task")

        assert result.status == "failure"
        assert result.tool_calls == 1
        assert result.turns == 3

    asyncio.run(run())


def test_observer_consumes_stream_and_only_records_assessment() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(rgb=ImageFrame(data="frame"), task_success=False))
        observer = FakeObserver(robot)

        def reflect(context, options, state, model):
            del options, state, model
            reflection_text = "\n".join(
                item.text
                for message in context.messages
                for item in getattr(message, "content", [])
                if isinstance(item, TextContent)
            )
            assert "The gripper moved toward the mug." in reflection_text
            return faux_assistant_message(
                "Grounded State Reflection: The current tail observation was reviewed."
            )

        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            reflect,
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            observer=observer,
            auto_reflections=False,
        )

        result = await agent.run("Pick up the mug")

        assert result.status == "success"
        assert observer.calls == [("Pick up the mug", "pick up the mug", [robot.observation])]
        assert "observer_events" not in result.details["tool_trace"][1]["result"]
        # chunk 评估必须进入本次 execute_vla 的 feedback。
        assert result.details["tool_trace"][1]["result"]["working_memory"]["visual"] == {
            "observations": ["The gripper moved toward the mug."],
            "keyframes": [],
        }
        tool_payload = json.loads(result.details["tool_trace"][1]["text"][0])
        assert "working_memory" not in tool_payload
        assert tool_payload["execution_diagnostics"] == {
            "start": {"sequence": None, "robot_state": {}},
            "gripper_events": [],
            "end": {"sequence": 1, "robot_state": {}},
            "chunks_used": 1,
            "stop_reason": None,
        }
        assert any(
            item.startswith("Observer advisory")
            for item in result.details["tool_trace"][1]["text"]
        )
        assert result.details["tool_trace"][1]["image_count"] == 2
        event = result.details["observer_trace"][0]
        assert "chunk" not in event
        assert event["range_start"] == 1
        assert event["range_end"] == 1
        assert event["input_sequences"] == [1]
        assert event["keyframe_sequence"] is None
        assert event["is_tail"] is True
        assert event["observation"] == "The gripper moved toward the mug."
        assert event["latency_ms"] >= 0

    asyncio.run(run())


def test_observer_ordinary_text_can_be_disabled_for_ablation() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(rgb=ImageFrame(data="frame"), task_success=False))
        observer = FakeObserver(robot)
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            observer=observer,
            include_observer_text=False,
        )

        result = await agent.run("Pick up the mug")

        visual = result.details["tool_trace"][1]["result"]["working_memory"]["visual"]
        assert visual == {
            "observations": [],
            "keyframes": [],
        }
        assert result.details["observer_trace"][0]["observation"] == (
            "The gripper moved toward the mug."
        )
        assert result.details["tool_trace"][1]["image_count"] == 2

    asyncio.run(run())


def test_observer_failure_does_not_fail_episode() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(rgb=ImageFrame(data="frame"), task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            observer=FailingObserver(),
        )

        result = await agent.run("Pick up the mug")

        assert result.status == "success"
        assert result.details["tool_trace"][1]["result"]["working_memory"]["visual"] == {
            "observations": [],
            "keyframes": [],
        }
        event = result.details["observer_trace"][0]
        assert event["keyframe_sequence"] is None
        assert event["observation"] is None

    asyncio.run(run())


def test_observer_does_not_block_the_next_action_chunk() -> None:
    class CoordinatedVla(FakeVla):
        def __init__(self, observer_started: asyncio.Event) -> None:
            super().__init__()
            self._observer_started = observer_started

        async def infer(self, instruction, observation, cancel_event=None):
            if self.instructions:
                await self._observer_started.wait()
            return await super().infer(instruction, observation, cancel_event)

    async def run() -> None:
        robot = FakeRobot(
            Observation(rgb=ImageFrame(data="frame"), task_success=False),
            success_after=2,
        )
        observer = BlockingObserver(robot)
        agent = make_agent(
            robot,
            CoordinatedVla(observer.started),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            observer=observer,
        )

        run_task = asyncio.create_task(agent.run("Pick up the mug"))
        await observer.started.wait()
        for _ in range(10):
            if len(robot.action_calls) == 2:
                break
            await asyncio.sleep(0)

        assert len(robot.action_calls) == 2
        assert not run_task.done()
        observer.release.set()
        result = await run_task
        assert result.status == "success"
        # 进行中的评估不阻塞动作，尾部必须在本次 feedback 前补齐。
        assert len(result.details["observer_trace"]) == 2
        assert [
            event["observation"] for event in result.details["observer_trace"]
        ] == [
            "The robot is moving toward the target.",
            "The robot is moving toward the target.",
        ]
        assert result.details["observer_trace"][1]["is_tail"] is True

    asyncio.run(run())


@pytest.mark.parametrize("terminal_after", [None, 197, 0])
def test_dynamic_observer_drains_before_reflection(terminal_after) -> None:
    """慢推理期间继续执行动作；积累的帧合为尾部窗口并进入同轮 Reflection。"""
    async def run() -> None:
        executed = asyncio.Event()
        release = asyncio.Event()
        expected_actions = 200 if terminal_after is None else terminal_after

        class ChunkVla(FakeVla):
            async def infer(self, instruction, observation, cancel_event=None):
                self.instructions.append(instruction)
                return ActionChunk(actions=[[0.1, 0.2]] * (10 if expected_actions else 0))

        class StreamRobot(FakeRobot):
            async def execute_action_chunk(self, actions, cancel_event=None):
                self.action_calls.append(actions)
                count = len(self.action_calls)
                self.observation = Observation(
                    rgb=ImageFrame(data=f"frame-{count}"),
                    task_success=count == terminal_after,
                )
                self.observation_sink.publish(self.observation)
                if count == expected_actions:
                    executed.set()
                await asyncio.sleep(0)
                return ExecutionFeedback(
                    status="success" if self.observation.task_success else "running",
                    observation=self.observation,
                    executed_steps=1,
                )

        class SlowObserver:
            def __init__(self):
                self.calls = []
                self.active = 0
                self.max_active = 0

            async def assess(self, task, instruction, observations, **kwargs):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(([o.rgb.data for o in observations], kwargs))
                index = len(self.calls)
                try:
                    await release.wait()
                    await asyncio.sleep(0)
                    return ObserverAssessment(None, f"window-{index} observed")
                finally:
                    self.active -= 1

        robot = StreamRobot(Observation(rgb=ImageFrame(data="frame-0")))
        observer = SlowObserver()

        def reflect(context, options, state, model):
            text = "\n".join(
                item.text for message in context.messages
                for item in getattr(message, "content", []) if isinstance(item, TextContent)
            )
            if expected_actions:
                assert "window-2 observed" in text
                assert len(observer.calls) == 2
            else:
                assert observer.calls == []
            return faux_assistant_message("Grounded State Reflection: Observations reviewed.")

        agent = make_agent(
            robot, ChunkVla(), plan_response("pour tomato sauce"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}), stop_reason="toolUse",
            ),
            reflect,
            faux_assistant_message(
                faux_tool_call("finish", {"success": terminal_after == 197, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            observer=observer, max_vla_chunks=20, auto_reflections=False,
        )
        robot.observation_sink.publish(robot.observation)
        run_task = asyncio.create_task(agent.run("Pour tomato sauce twice"))
        if expected_actions:
            await asyncio.wait_for(executed.wait(), timeout=5)
            # 第一次观察仍阻塞时，所有物理动作已执行，Reflection 必须继续等待。
            assert not run_task.done()
        release.set()
        result = await asyncio.wait_for(run_task, timeout=5)
        trace = result.details["observer_trace"]
        assert len(robot.action_calls) == expected_actions
        assert len(trace) == (2 if expected_actions else 0)
        if expected_actions:
            assert observer.max_active == 1
            first_frames, first_kwargs = observer.calls[0]
            last_frames, last_kwargs = observer.calls[1]
            assert first_frames[0] == "frame-0"
            assert last_frames == [first_frames[-1], f"frame-{expected_actions}"]
            assert trace[0]["range_end"] == trace[1]["range_start"]
            assert trace[1]["range_end"] == expected_actions + 1
            assert first_kwargs["previous_observation"] == ""
            assert last_kwargs["previous_observation"] == "window-1 observed"
            visual = result.details["tool_trace"][1]["result"]["working_memory"]["visual"]
            assert visual["observations"] == ["window-1 observed", "window-2 observed"]

    asyncio.run(run())


def test_multiple_tool_calls_only_execute_first() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        vla = FakeVla()
        agent = make_agent(
            robot,
            vla,
            plan_response("move"),
            faux_assistant_message(
                [
                    faux_tool_call("execute_vla", {"step_id": "step-1"}, tool_call_id="first"),
                    faux_tool_call(
                        "execute_vla", {"step_id": "unsafe-step"}, tool_call_id="second"
                    ),
                ],
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": False, "summary": "Stopped safely."}, tool_call_id="finish"
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Move once")

        assert result.status == "failure"
        assert len(robot.action_calls) == 1
        assert vla.instructions == ["move"]

    asyncio.run(run())


def test_robot_action_after_task_success_is_blocked() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        vla = FakeVla()
        agent = make_agent(
            robot,
            vla,
            plan_response("move"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}, tool_call_id="move"),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "execute_vla", {"step_id": "step-1"}, tool_call_id="unsafe"
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "finish", {"success": True, "summary": "Done."}, tool_call_id="finish"
                ),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Move once")

        assert result.status == "success"
        assert len(robot.action_calls) == 1
        assert result.details["tool_trace"][2]["status"] == "blocked"
        assert result.details["tool_trace"][2]["reason"].startswith(
            "The environment already reports task success"
        )

    asyncio.run(run())


def test_two_step_plan_is_advanced_only_by_agent_tool_calls() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False), success_after=2)
        vla = FakeVla()
        agent = make_agent(
            robot,
            vla,
            plan_response("place cookies", step_id="cookies", pending_instruction="place sauce"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "cookies"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "cookies", "next_step_id": "step-2"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-2"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "step-2", "next_step_id": None},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Place both items")

        assert result.status == "success"
        assert vla.instructions == ["place cookies", "place sauce"]
        assert [step["status"] for step in result.details["task_plan"]["steps"]] == [
            "completed",
            "completed",
        ]

    asyncio.run(run())


def test_rejected_plan_transition_allows_corrected_second_call() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False), success_after=2)
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("place cookies", step_id="cookies", pending_instruction="place sauce"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "cookies"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "cookies", "next_step_id": "cookies"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "cookies", "next_step_id": "step-2"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-2"}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Place both items")

        rejected = result.details["tool_trace"][2]
        assert rejected["status"] == "rejected"
        assert rejected["result"]["valid_next_step_ids"] == ["step-2"]
        rejected_memory = result.details["episode_memory"][2]
        assert rejected_memory["status"] == "rejected"
        assert rejected_memory["task_plan"]["steps"][0]["status"] == "running"
        assert rejected_memory["retry_instruction"].startswith("Re-read")
        assert result.details["tool_trace"][3]["status"] == "completed"
        assert result.details["task_plan"]["steps"][1]["status"] == "running"

    asyncio.run(run())


def test_completed_plan_uses_optional_environment_finalizer() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False), success_after=100)
        finalizer = FakeFinalizer()
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("place cookies"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "step-1", "next_step_id": None},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": True, "summary": "Done."}),
                stop_reason="toolUse",
            ),
            finalizer=finalizer,
            max_vla_chunks=1,
        )

        result = await agent.run("Place cookies")

        assert result.status == "success"
        assert finalizer.calls == 1

    asyncio.run(run())


def test_new_running_step_requires_its_own_action_before_advance() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False), success_after=100)
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("place cookies", step_id="cookies", pending_instruction="place sauce"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "cookies"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "cookies", "next_step_id": "step-2"},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "step-2", "next_step_id": None},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-2"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Smoke test done."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=1,
        )

        result = await agent.run("Place both items")

        assert len(robot.action_calls) == 2
        assert result.details["tool_trace"][2]["status"] == "completed"
        assert result.details["tool_trace"][3]["status"] == "rejected"
        assert "has not executed a robot action" in (
            result.details["tool_trace"][3]["result"]["reason"]
        )

    asyncio.run(run())


def test_execute_vla_requires_plan_and_running_step_id() -> None:
    async def run_without_plan() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Blocked."}),
                stop_reason="toolUse",
            ),
        )
        result = await agent.run("Move")
        assert robot.action_calls == []
        assert result.details["tool_trace"][0]["status"] == "blocked"
        assert result.details["tool_trace"][0]["reason"].startswith("No robot action")
        assert result.details["tool_trace"][1]["status"] == "completed"

    async def run_with_wrong_step_id() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("move left"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "wrong-step"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Blocked."}),
                stop_reason="toolUse",
            ),
        )
        result = await agent.run("Move")
        assert robot.action_calls == []
        assert result.details["tool_trace"][1]["status"] == "blocked"
        assert "running TaskPlan step_id" in result.details["tool_trace"][1]["reason"]

    async def run_with_completed_plan() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("move"),
            faux_assistant_message(
                faux_tool_call(
                    "advance_task_plan",
                    {"completed_step_id": "step-1", "next_step_id": None},
                ),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Done."}),
                stop_reason="toolUse",
            ),
        )
        result = await agent.run("Move")
        assert len(robot.action_calls) == 1
        assert result.details["tool_trace"][1]["status"] == "rejected"
        assert "has not executed a robot action" in (
            result.details["tool_trace"][1]["result"]["reason"]
        )

    asyncio.run(run_without_plan())
    asyncio.run(run_with_wrong_step_id())
    asyncio.run(run_with_completed_plan())


def test_chunk_budget_returns_running_without_advancing_plan() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False), success_after=100)
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("move carefully"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Budget reached."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=24,
        )

        result = await agent.run("Move carefully")

        assert len(robot.action_calls) == 24
        execute_result = result.details["tool_trace"][1]["result"]
        assert execute_result["status"] == "running"
        assert execute_result["stop_reason"] == "chunk_budget"
        assert execute_result["step_id"] == "step-1"
        assert execute_result["chunks_used"] == 24
        assert execute_result["executed_steps"] == 48
        assert result.details["task_plan"]["steps"][0]["status"] == "running"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("initial_command", "next_command"),
    [(-1.0, 1.0), (1.0, -1.0)],
)
def test_gripper_transition_does_not_interrupt_vla_chunks(
    initial_command, next_command
) -> None:
    @dataclass
    class GripperRobot(FakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            del cancel_event
            self.action_calls.append(actions)
            command = actions.actions[-1][6]
            qpos = [0.02, -0.02] if command > 0 else [0.04, -0.04]
            self.observation = Observation(
                robot_state={
                    "gripper_command": command,
                    "gripper_qpos": qpos,
                },
                task_success=False,
            )
            return ExecutionFeedback(
                status="running",
                observation=self.observation,
                executed_steps=len(actions.actions),
            )

    class GripperVla(FakeVla):
        async def infer(self, instruction, observation, cancel_event=None):
            del observation, cancel_event
            self.instructions.append(instruction)
            command = initial_command if len(self.instructions) == 1 else next_command
            return ActionChunk(actions=[[0.0] * 6 + [command] for _ in range(10)])

    async def run() -> None:
        robot = GripperRobot(
            Observation(
                robot_state={
                    "gripper_command": initial_command,
                    "gripper_qpos": (
                        [0.02, -0.02] if initial_command > 0 else [0.04, -0.04]
                    ),
                },
                task_success=False,
            ),
            success_after=100,
        )
        agent = make_agent(
            robot,
            GripperVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                faux_tool_call("finish", {"success": False, "summary": "Checked."}),
                stop_reason="toolUse",
            ),
            max_vla_chunks=3,
        )

        result = await agent.run("Pick up the mug")

        assert len(robot.action_calls) == 30
        execute_result = result.details["tool_trace"][1]["result"]
        assert execute_result["status"] == "running"
        assert execute_result["stop_reason"] == "chunk_budget"
        assert "event" not in execute_result
        assert execute_result["chunks_used"] == 3
        assert all(len(chunk.actions) == 1 for chunk in robot.action_calls)

    asyncio.run(run())


def test_terminal_robot_failure_stops_agent_without_another_model_turn() -> None:
    @dataclass
    class TerminalRobot(FakeRobot):
        async def execute_action_chunk(self, actions, cancel_event=None):
            del cancel_event
            self.action_calls.append(actions)
            return ExecutionFeedback(
                status="failure",
                observation=self.observation,
                executed_steps=1,
            )

    async def run() -> None:
        robot = TerminalRobot(Observation(task_success=False), success_after=100)
        agent = make_agent(
            robot,
            FakeVla(),
            plan_response("pick up the mug"),
            faux_assistant_message(
                faux_tool_call("execute_vla", {"step_id": "step-1"}),
                stop_reason="toolUse",
            ),
        )

        result = await agent.run("Pick up the mug")

        assert result.status == "failure"
        assert result.summary == "Robot environment reached a terminal failure state."
        assert [entry["name"] for entry in result.details["tool_trace"]] == [
            "create_task_plan",
            "execute_vla",
        ]
        assert len(robot.action_calls) == 1

    asyncio.run(run())


def test_task_plan_rejects_invalid_creation_and_duplicate_creation() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(robot, FakeVla())
        valid_step = {
            "id": "step-1",
            "instruction": "move",
            "status": "running",
        }

        duplicate = await agent._create_task_plan([valid_step, valid_step], None)
        assert duplicate["status"] == "rejected"
        assert "unique" in duplicate["reason"]
        multiple_running = await agent._create_task_plan(
            [valid_step, {**valid_step, "id": "step-2", "status": "running"}],
            None,
        )
        assert multiple_running["status"] == "rejected"
        assert "exactly one running" in multiple_running["reason"]
        empty = await agent._create_task_plan(
            [{**valid_step, "instruction": "  "}], None
        )
        assert empty["status"] == "rejected"
        assert "must not be empty" in empty["reason"]
        non_contiguous = await agent._create_task_plan(
            [
                {**valid_step, "status": "pending"},
                {**valid_step, "id": "step-2", "status": "running"},
            ],
            None,
        )
        assert non_contiguous["status"] == "rejected"
        assert "before the running step" in non_contiguous["reason"]

        await agent._create_task_plan([valid_step], None)
        duplicate_plan = await agent._create_task_plan([valid_step], None)
        assert duplicate_plan["status"] == "rejected"
        assert duplicate_plan["task_plan"] == {"steps": [valid_step]}
        assert robot.action_calls == []

    asyncio.run(run())


def test_invalid_plan_advancement_is_atomic() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(robot, FakeVla(), max_vla_chunks=1)
        await agent._create_task_plan(
            [
                {
                    "id": "step-1",
                    "instruction": "move",
                    "status": "running",
                },
                {
                    "id": "step-2",
                    "instruction": "place",
                    "status": "pending",
                },
            ],
            None,
        )
        await agent._execute_vla("step-1", None)
        original_plan = agent._task_plan.summary()

        wrong_running = await agent._advance_task_plan("step-2", None, None)
        assert wrong_running["status"] == "rejected"
        missing_next = await agent._advance_task_plan("step-1", None, None)
        assert missing_next["status"] == "rejected"
        assert missing_next["valid_next_step_ids"] == ["step-2"]
        unknown = await agent._advance_task_plan("step-1", "missing", None)
        assert unknown["status"] == "rejected"
        assert agent._task_plan.summary() == original_plan

        await agent._advance_task_plan("step-1", "step-2", None)
        stale = await agent._advance_task_plan("step-1", None, None)
        assert stale["status"] == "rejected"
        assert [step.status for step in agent._task_plan.steps] == [
            "completed",
            "running",
        ]
        assert len(robot.action_calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("approved", [True, False])
def test_plan_advance_optional_human_approval(approved: bool) -> None:
    async def run() -> None:
        calls: list[tuple[str, str | None, str]] = []

        async def approval(current, next_step, reflection):
            calls.append(
                (
                    current.id,
                    next_step.id if next_step is not None else None,
                    reflection,
                )
            )
            return approved, "" if approved else "The drawer is still closed."

        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(
            robot,
            FakeVla(),
            max_vla_chunks=1,
            plan_advance_approval=approval,
        )
        await agent._create_task_plan(
            [
                {"id": "open", "instruction": "open drawer", "status": "running"},
                {"id": "pick", "instruction": "pick object", "status": "pending"},
            ],
            None,
        )
        await agent._execute_vla("open", None)
        result = await agent._advance_task_plan("open", "pick", None)
        details = agent._result_details()

        assert calls == [("open", "pick", "")]
        assert details["human_intervention"] is True
        assert details["supervision_events"] == [
            {
                "tool": "advance_task_plan",
                "completed_step_id": "open",
                "next_step_id": "pick",
                "decision": "approved" if approved else "rejected",
                "feedback": "" if approved else "The drawer is still closed.",
            }
        ]
        if approved:
            assert result["status"] == "updated"
            assert [step.status for step in agent._task_plan.steps] == [
                "completed",
                "running",
            ]
        else:
            assert result["status"] == "rejected"
            assert result["reason"] == "The drawer is still closed."
            assert result["human_feedback"] == "The drawer is still closed."
            assert "choose an available tool" in result["retry_instruction"]
            assert [step.status for step in agent._task_plan.steps] == [
                "running",
                "pending",
            ]

    asyncio.run(run())


def test_task_plan_repair_reopens_completed_prerequisite_atomically() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation(task_success=False))
        agent = make_agent(robot, FakeVla(), max_vla_chunks=1)
        await agent._create_task_plan(
            [
                {"id": "pick", "instruction": "pick milk", "status": "running"},
                {"id": "lift", "instruction": "lift milk", "status": "pending"},
                {"id": "pour", "instruction": "pour milk", "status": "pending"},
                {"id": "place", "instruction": "place milk", "status": "pending"},
            ],
            None,
        )
        await agent._execute_vla("pick", None)
        await agent._advance_task_plan("pick", "lift", None)
        await agent._execute_vla("lift", None)
        await agent._advance_task_plan("lift", "pour", None)
        original_plan = agent._task_plan.summary()

        no_action = await agent._repair_task_plan("pour", "pick", None)
        assert no_action["status"] == "rejected"
        assert "executed a robot action" in no_action["reason"]
        assert agent._task_plan.summary() == original_plan

        await agent._execute_vla("pour", None)
        for running_id, reopen_id, error in (
            ("pick", "pick", "Only the running step"),
            ("pour", "missing", "Unknown TaskPlan step"),
            ("pour", "place", "must precede"),
        ):
            rejected = await agent._repair_task_plan(running_id, reopen_id, None)
            assert rejected["status"] == "rejected"
            assert error in rejected["reason"]
            assert agent._task_plan.summary() == original_plan

        result = await agent._repair_task_plan("pour", "pick", None)

        assert result["status"] == "repaired"
        assert [step.status for step in agent._task_plan.steps] == [
            "running",
            "pending",
            "pending",
            "pending",
        ]
        assert result["active_step"]["id"] == "pick"
        assert result["plan_progress"] == {
            "completed": [],
            "running": "pick",
            "pending": ["lift", "pour", "place"],
        }
        invalid_advance = await agent._advance_task_plan("pick", "lift", None)
        assert invalid_advance["status"] == "rejected"

    asyncio.run(run())


def test_physical_port_calls_are_serialized() -> None:
    async def run() -> None:
        robot = FakeRobot(Observation())
        agent = make_agent(robot, FakeVla())
        await asyncio.gather(agent._observe(None), agent._observe(None))
        assert robot.max_active_calls == 1

    asyncio.run(run())
