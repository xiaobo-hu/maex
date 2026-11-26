"""RoboAgent 运行时组合：一个 Agent Loop、受限工具和机器人/VLA 端口。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TypeAlias, cast

from agent_core import (
    Agent,
    AgentContext,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentOptions,
    AgentState,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    Context,
    ImageContent,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

from .memory import (
    EpisodeMemory,
    VisualMemory,
    VlaWorkingMemory,
)
from .observer import ExecutionDiagnosticsRunner, VisualObserver
from .ports import EpisodeFinalizerPort, RobotPort, VlaPolicyPort
from .prompts import DEFAULT_SYSTEM_PROMPT, STATE_REFLECTION_PROMPT
from .prompts.robomemarena import task_primitive_reference
from .recovery import RecoveryState
from .skills import TaskExecutionSkill, apply_task_execution_skill
from .tools import (
    AdjustEePoseTool,
    AdvanceTaskPlanTool,
    CloseGripperTool,
    CreateTaskPlanTool,
    ExecuteVlaTool,
    FinishTool,
    OpenGripperTool,
    RepairTaskPlanTool,
    RestageVlaTool,
    ReturnToHomeTool,
)
from .types import (
    ActionChunk,
    EpisodeResult,
    EpisodeStatus,
    Observation,
    TaskPlan,
    TaskStep,
    TaskStepStatus,
)
from .utils.ee_pose import (
    ROTATION_ACTIONS,
    TRANSLATION_ACTIONS,
    EePose,
    EePoseAdjustmentAction,
    EePoseAdjustmentConfig,
    applied_rotation_deg,
    build_adjustment_target,
    build_osc_action,
    pose_from_observation,
    primary_error,
    target_is_reached,
    vector_subtract,
    workspace_motion_is_allowed,
    workspace_violation_distance,
)
from .visual_evidence import (
    build_decision_visual_evidence,
    build_reflection_visual_evidence,
    sanitize_reflection_content,
)


def _finite_number(value: object) -> float | None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _gripper_qpos(observation: Observation) -> tuple[float, ...] | None:
    value = observation.robot_state.get("gripper_qpos")
    if not isinstance(value, list | tuple) or not value:
        return None
    items = tuple(_finite_number(item) for item in value)
    if any(item is None for item in items):
        return None
    return cast(tuple[float, ...], items)


def _gripper_event(
    chunk: ActionChunk,
    previous_command: float | None,
) -> tuple[str | None, float | None]:
    """从完整 action chunk 中提取最后一次夹爪命令边沿。"""
    active_sign = (
        None
        if previous_command is None or previous_command == 0.0
        else (1 if previous_command > 0 else -1)
    )
    event: str | None = None
    latest_command = previous_command
    for action in chunk.actions:
        if len(action) <= 6:
            continue
        command = _finite_number(action[6])
        if command is None or command == 0.0:
            continue
        sign = 1 if command > 0 else -1
        if active_sign is not None and sign != active_sign:
            event = "close" if sign > 0 else "open"
        active_sign = sign
        latest_command = command
    return event, latest_command


_REFLECTION_TOOL_NAMES = {
    "execute_vla",
    "adjust_ee_pose",
    "restage_vla",
    "return_to_home",
}
_RESTAGE_DISTANCE_M = 0.05

PlanAdvanceApproval: TypeAlias = Callable[
    [TaskStep, TaskStep | None, str], Awaitable[tuple[bool, str]]
]


@dataclass(slots=True)
class RoboAgentOptions:
    """RoboAgent 的最小构造配置。"""

    model: Model
    robot: RobotPort
    vla: VlaPolicyPort
    stream_fn: StreamFn
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 12
    max_vla_chunks: int = 24
    stream_options: SimpleStreamOptions | None = None
    observer: VisualObserver | None = None
    include_observer_text: bool = True
    finalizer: EpisodeFinalizerPort | None = None
    gripper_checkpoint_threshold: float = 0.01
    ee_pose_adjustment: EePoseAdjustmentConfig = field(
        default_factory=EePoseAdjustmentConfig
    )
    task_execution_skill: TaskExecutionSkill | None = None
    plan_advance_approval: PlanAdvanceApproval | None = None


class RoboAgent:
    """使用单一 Agent Loop 完成一个机器人 episode。"""

    def __init__(self, options: RoboAgentOptions) -> None:
        if options.max_turns < 1 or options.max_vla_chunks < 1:
            raise ValueError("max_turns and max_vla_chunks must be at least 1")
        if (
            not math.isfinite(options.gripper_checkpoint_threshold)
            or options.gripper_checkpoint_threshold < 0
        ):
            raise ValueError("gripper_checkpoint_threshold must be non-negative")
        self._options = options
        self._decision_system_prompt = apply_task_execution_skill(
            options.system_prompt,
            options.task_execution_skill,
        )
        self._latest_observation: Observation | None = None
        self._last_gripper_command: float | None = None
        self._workspace_violation = False
        self._finish_status: str | None = None
        self._finish_summary = ""
        self._turns = 0
        self._tool_calls = 0
        self._tool_history: list[str] = []
        self._tool_trace: list[dict[str, object]] = []
        self._episode_memory = EpisodeMemory()
        self._recovery_state = RecoveryState()
        self._decision_tools: list[AgentTool] = []
        self._reflection_source: tuple[str, str] | None = None
        self._pre_reflection_context: AgentContext[object] | None = None
        self._pending_decision_prompt: str | None = None
        self._state_reflections: list[dict[str, str]] = []
        self._supervision_events: list[dict[str, object]] = []
        self._task_plan: TaskPlan | None = None
        self._current_step_has_executed_action = False
        self._return_home_after_plan_repair_available = False
        self._running = False
        self._execution_diagnostics = ExecutionDiagnosticsRunner(
            options.observer.source if options.observer is not None else None,
            gripper_transition_threshold=options.gripper_checkpoint_threshold,
        )
        # AgentTool 的 async 外壳不能改变机器人动作的串行语义；所有物理端口
        # 调用都经过这把锁，禁止未来的异步扩展意外并发操作同一机器人。
        self._physical_lock = asyncio.Lock()
        self._agent: Agent[object] = self._build_agent()
        self._agent.subscribe(self._record_agent_event)

    def _build_agent(self) -> Agent[object]:
        tools: list[AgentTool] = [
            CreateTaskPlanTool(self._create_task_plan),
            ExecuteVlaTool(self._execute_vla),
            RestageVlaTool(self._restage_vla),
            AdjustEePoseTool(self._adjust_ee_pose),
            CloseGripperTool(self._close_gripper),
            OpenGripperTool(self._open_gripper),
            ReturnToHomeTool(self._return_to_home),
            AdvanceTaskPlanTool(self._advance_task_plan),
            RepairTaskPlanTool(self._repair_task_plan),
            FinishTool(self._finish),
        ]
        self._decision_tools = list(tools)
        return Agent(
            AgentOptions(
                initial_state=AgentState(
                    model=self._options.model,
                    system_prompt=self._decision_system_prompt,
                    tools=tools,
                ),
                stream_fn=self._stream_fn,
                transform_context=self._episode_memory.compact_transcript_for_model,
                before_tool_call=self._before_tool_call,
                prepare_next_turn_with_context=self._prepare_next_turn,
                should_stop_after_turn=self._should_stop_after_turn,
                tool_execution="sequential",
                stream_options=self._options.stream_options,
            )
        )

    async def run(self, task: str) -> EpisodeResult:
        """开始一次 episode，并返回结构化终态。"""
        if self._running:
            raise RuntimeError("RoboAgent is already running")
        self._running = True
        self._finish_status = None
        self._finish_summary = ""
        self._turns = 0
        self._tool_calls = 0
        self._tool_history = []
        self._tool_trace = []
        self._episode_memory.clear()
        self._recovery_state.clear()
        self._reflection_source = None
        self._pre_reflection_context = None
        self._pending_decision_prompt = None
        self._state_reflections = []
        self._supervision_events = []
        self._task_plan = None
        self._current_step_has_executed_action = False
        self._return_home_after_plan_repair_available = False
        self._last_gripper_command = None
        self._workspace_violation = False
        self._latest_observation = await self._observe(None)
        task = task.strip()
        if self._options.observer is not None:
            await self._options.observer.start(task)
        self._agent.reset()
        try:
            prompt_content: list[TextContent | ImageContent] = [TextContent(text=task)]
            primitive_reference = task_primitive_reference(task)
            if primitive_reference is not None:
                prompt_content.append(TextContent(text=primitive_reference))
            prompt_content.extend(self._latest_observation.llm_content())
            await self._agent.prompt(UserMessage(content=prompt_content))
        finally:
            if self._options.observer is not None:
                await self._options.observer.stop()
            self._running = False
        return self._result_from_agent()

    def abort(self) -> None:
        """请求取消当前运行，并把取消信号传给 Provider、VLA 和机器人端口。"""
        self._agent.abort()

    def _stream_fn(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]:
        """无工具 Reflection 回合不向 Provider 发送强制工具选择参数。"""
        resolved = options
        if not context.tools and options is not None and options.tool_choice is not None:
            resolved = options.model_copy()
            resolved.tool_choice = None
        return self._options.stream_fn(model, context, resolved)

    async def _observe(self, cancel_event: asyncio.Event | None) -> Observation:
        async with self._physical_lock:
            observation = await self._options.robot.observe(cancel_event)
        self._remember_observation(observation)
        return observation

    async def _create_task_plan(
        self,
        step_payloads: list[dict[str, object]],
        cancel_event: asyncio.Event | None,
    ) -> dict[str, object]:
        """保存 Agent 显式创建的计划，不推断或改写任何步骤状态。"""
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        if self._task_plan is not None:
            return self._plan_rejection(
                "TaskPlan has already been created for this episode.",
                tool_name="create_task_plan",
            )
        steps: list[TaskStep] = []
        seen_ids: set[str] = set()
        for payload in step_payloads:
            step_id = str(payload["id"]).strip()
            instruction = str(payload["instruction"]).strip()
            status_value = str(payload["status"]).strip()
            if not step_id or not instruction:
                return self._plan_rejection(
                    "TaskPlan step fields must not be empty.",
                    tool_name="create_task_plan",
                )
            if step_id in seen_ids:
                return self._plan_rejection(
                    f"TaskPlan step id must be unique: {step_id}.",
                    tool_name="create_task_plan",
                )
            if status_value not in {"pending", "running", "completed"}:
                return self._plan_rejection(
                    f"Invalid TaskPlan step status: {status_value}.",
                    tool_name="create_task_plan",
                )
            seen_ids.add(step_id)
            steps.append(
                TaskStep(
                    id=step_id,
                    instruction=instruction,
                    status=cast(TaskStepStatus, status_value),
                )
            )
        invariant_error = self._task_plan_invariant_error(steps)
        if invariant_error is not None:
            return self._plan_rejection(
                invariant_error,
                tool_name="create_task_plan",
            )
        self._task_plan = TaskPlan(steps=steps)
        return {
            "status": "created",
            "task_plan": self._task_plan.summary(),
            **self._plan_context(),
        }

    async def _advance_task_plan(
        self,
        completed_step_id: str,
        next_step_id: str | None,
        cancel_event: asyncio.Event | None,
    ) -> dict[str, object]:
        """原子应用 Agent 指定的当前完成项和下一运行项。"""
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        plan = self._task_plan
        if plan is None:
            return self._plan_rejection(
                "No TaskPlan exists.", tool_name="advance_task_plan"
            )
        invariant_error = self._task_plan_invariant_error(plan.steps)
        if invariant_error is not None:
            return self._plan_rejection(
                f"Current TaskPlan is inconsistent: {invariant_error}",
                tool_name="advance_task_plan",
            )
        running = self._running_task_step(plan)
        if running.id != completed_step_id:
            return self._plan_rejection(
                f"Only the running step can be completed: expected {running.id}.",
                tool_name="advance_task_plan",
            )
        if not self._current_step_has_executed_action:
            return self._plan_rejection(
                "The current running step has not executed a robot action. Call "
                "execute_vla before advancing the TaskPlan.",
                tool_name="advance_task_plan",
            )
        running_index = plan.steps.index(running)
        expected_next = next(
            (step for step in plan.steps[running_index + 1 :] if step.status == "pending"),
            None,
        )
        next_step = None
        if next_step_id is not None:
            next_step = next(
                (step for step in plan.steps if step.id == next_step_id),
                None,
            )
            if next_step is None:
                return self._plan_rejection(
                    f"Unknown next TaskPlan step: {next_step_id}.",
                    tool_name="advance_task_plan",
                )
            if expected_next is None or next_step.id != expected_next.id:
                expected = expected_next.id if expected_next is not None else "null"
                return self._plan_rejection(
                    f"next_step_id must be the immediate pending step: {expected}.",
                    tool_name="advance_task_plan",
                )
        elif expected_next is not None:
            return self._plan_rejection(
                f"next_step_id is required while pending steps remain; expected "
                f"{expected_next.id}.",
                tool_name="advance_task_plan",
            )

        updated_steps = [
            replace(step, status="completed")
            if step.id == completed_step_id
            else replace(step, status="running")
            if next_step is not None and step.id == next_step.id
            else step
            for step in plan.steps
        ]
        invariant_error = self._task_plan_invariant_error(
            updated_steps, allow_terminal=True
        )
        if invariant_error is not None:
            return self._plan_rejection(
                f"Requested transition would create an inconsistent TaskPlan: "
                f"{invariant_error}",
                tool_name="advance_task_plan",
            )
        approval = self._options.plan_advance_approval
        if approval is not None:
            latest_reflection = (
                self._state_reflections[-1]["text"] if self._state_reflections else ""
            )
            approved, feedback = await approval(
                running,
                next_step,
                latest_reflection,
            )
            feedback = feedback.strip()
            self._supervision_events.append(
                {
                    "tool": "advance_task_plan",
                    "completed_step_id": running.id,
                    "next_step_id": next_step.id if next_step is not None else None,
                    "decision": "approved" if approved else "rejected",
                    "feedback": feedback,
                }
            )
            if not approved:
                reason = feedback or (
                    "The current step is not complete. Re-evaluate the physical state."
                )
                rejection = self._plan_rejection(
                    reason,
                    tool_name="advance_task_plan",
                )
                rejection["human_feedback"] = reason
                rejection["retry_instruction"] = (
                    "Use the human feedback to re-evaluate the physical state and choose "
                    "an available tool. Do not advance this step until it is complete."
                )
                return rejection
        self._task_plan = TaskPlan(steps=updated_steps)
        self._current_step_has_executed_action = False
        self._return_home_after_plan_repair_available = False
        self._recovery_state.clear()
        return {
            "status": "updated",
            "task_plan": self._task_plan.summary(),
            **self._plan_context(),
        }

    async def _repair_task_plan(
        self,
        running_step_id: str,
        reopen_step_id: str,
        cancel_event: asyncio.Event | None,
    ) -> dict[str, object]:
        """原子暂停当前步骤并重开一个已完成的前置步骤。"""
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        plan = self._task_plan
        if plan is None:
            return self._plan_rejection(
                "No TaskPlan exists.", tool_name="repair_task_plan"
            )
        invariant_error = self._task_plan_invariant_error(plan.steps)
        if invariant_error is not None:
            return self._plan_rejection(
                f"Current TaskPlan is inconsistent: {invariant_error}",
                tool_name="repair_task_plan",
            )
        running = self._running_task_step(plan)
        if running.id != running_step_id:
            return self._plan_rejection(
                f"Only the running step can be paused: expected {running.id}.",
                tool_name="repair_task_plan",
            )
        if not self._current_step_has_executed_action:
            return self._plan_rejection(
                "The current running step has not executed a robot action. Call "
                "execute_vla before repairing the TaskPlan.",
                tool_name="repair_task_plan",
            )
        running_index = next(
            index for index, step in enumerate(plan.steps) if step.id == running.id
        )
        reopen_index = next(
            (
                index
                for index, step in enumerate(plan.steps)
                if step.id == reopen_step_id
            ),
            None,
        )
        if reopen_index is None:
            return self._plan_rejection(
                f"Unknown TaskPlan step to reopen: {reopen_step_id}.",
                tool_name="repair_task_plan",
            )
        reopen_step = plan.steps[reopen_index]
        if reopen_index >= running_index:
            return self._plan_rejection(
                "The reopened TaskPlan step must precede the running step.",
                tool_name="repair_task_plan",
            )
        if reopen_step.status != "completed":
            return self._plan_rejection(
                "The reopened TaskPlan step must be completed.",
                tool_name="repair_task_plan",
            )

        updated_steps = [
            step
            if index < reopen_index
            else replace(step, status="running")
            if index == reopen_index
            else replace(step, status="pending")
            for index, step in enumerate(plan.steps)
        ]
        invariant_error = self._task_plan_invariant_error(updated_steps)
        if invariant_error is not None:
            return self._plan_rejection(
                f"Requested repair would create an inconsistent TaskPlan: "
                f"{invariant_error}",
                tool_name="repair_task_plan",
            )
        self._task_plan = TaskPlan(steps=updated_steps)
        self._current_step_has_executed_action = False
        self._return_home_after_plan_repair_available = True
        self._recovery_state.clear()
        return {
            "status": "repaired",
            "task_plan": self._task_plan.summary(),
            **self._plan_context(),
        }

    async def _execute_vla(
        self, step_id: str, cancel_event: asyncio.Event | None
    ) -> tuple[dict[str, object], Observation | None, VlaWorkingMemory]:
        step = self._running_task_step(self._require_task_plan())
        if step_id != step.id:
            raise ValueError("execute_vla step_id must identify the running TaskPlan step")
        self._return_home_after_plan_repair_available = False
        instruction = step.instruction
        observer = self._options.observer
        payload: dict[str, object] = {}
        observation = self._latest_observation
        start_observation = observation
        self._execution_diagnostics.begin_vla_call(observation)
        if observer is not None:
            observer.begin_vla_call(instruction, step_id=step.id)
        try:
            payload, observation = await self._execute_vla_chunks(
                step,
                cancel_event,
            )
        finally:
            chunks_used = payload.get("chunks_used")
            stop_reason = payload.get("stop_reason")
            execution = self._execution_diagnostics.end_vla_call(
                observation,
                chunks_used=(
                    chunks_used
                    if isinstance(chunks_used, int) and not isinstance(chunks_used, bool)
                    else 0
                ),
                stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            )
            visual = (
                await observer.end_vla_call(
                    tuple(
                        (event.sequence, event.event)
                        for event in execution.gripper_events
                    )
                )
                if observer is not None
                else VisualMemory()
            )
            if not self._options.include_observer_text:
                visual = VisualMemory(keyframes=visual.keyframes)
        working_memory = VlaWorkingMemory(
            visual=visual,
            execution=execution,
            start_observation=start_observation,
        )
        self._add_physical_outcome_fields(payload, observation)
        chunks_used = payload.get("chunks_used")
        if (
            isinstance(chunks_used, int)
            and not isinstance(chunks_used, bool)
            and chunks_used > 0
        ):
            retry_streak = self._recovery_state.record_vla_attempt(step.id)
            payload["consecutive_vla_attempts"] = retry_streak
            if 3 <= retry_streak < 5:
                payload["recovery_notice"] = (
                    f"This task step has used {retry_streak} VLA executions. Before "
                    "retrying, inspect whether the latest execution made progress."
                )
            elif 5 <= retry_streak < 8:
                payload["recovery_notice"] = (
                    f"This task step has used {retry_streak} VLA executions without a "
                    "plan transition. Reconsider the current state or strategy before "
                    "another retry; recovery tools remain optional."
                )
            elif retry_streak >= 8:
                payload["recovery_notice"] = (
                    f"High VLA usage on the current task step: {retry_streak} executions. "
                    "Do not blindly repeat unchanged behavior; continue VLA only when "
                    "the latest evidence supports that choice."
                )
        return {
            **payload,
            "working_memory": working_memory.summary(),
        }, observation, working_memory

    async def _execute_vla_chunks(
        self,
        step: TaskStep,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None]:
        """逐动作执行 VLA chunk，并用真实 post-step EEF 截停越界动作。"""
        instruction = step.instruction
        observation = self._latest_observation or await self._observe(cancel_event)
        executed_steps = 0
        workspace_error = self._vla_workspace_error(observation)
        if workspace_error is not None:
            return self._with_plan_context(
                {
                    **workspace_error,
                    "step_id": step.id,
                    "chunks_used": 0,
                    "executed_steps": 0,
                }
            ), observation
        for chunk_index in range(self._options.max_vla_chunks):
            if cancel_event is not None and cancel_event.is_set():
                return self._with_plan_context(
                    {
                        "status": "cancelled",
                        "step_id": step.id,
                        "chunks_used": chunk_index,
                        "executed_steps": executed_steps,
                    }
                ), observation
            async with self._physical_lock:
                actions = await self._options.vla.infer(instruction, observation, cancel_event)
                for action in actions.actions:
                    if cancel_event is not None and cancel_event.is_set():
                        return self._with_plan_context(
                            {
                                "status": "cancelled",
                                "step_id": step.id,
                                "chunks_used": chunk_index + 1,
                                "executed_steps": executed_steps,
                            }
                        ), observation
                    action_chunk = ActionChunk(
                        actions=[action],
                        metadata=actions.metadata,
                    )
                    feedback = await self._options.robot.execute_action_chunk(
                        action_chunk,
                        cancel_event,
                    )
                    executed_steps += feedback.executed_steps
                    if feedback.executed_steps > 0:
                        self._current_step_has_executed_action = True
                        _, action_command = _gripper_event(
                            action_chunk,
                            self._last_gripper_command,
                        )
                        if action_command is not None:
                            self._last_gripper_command = action_command
                    observation = feedback.observation or await self._options.robot.observe(
                        cancel_event
                    )
                    self._remember_observation(observation)
                    workspace_error = self._vla_workspace_error(observation)
                    if workspace_error is not None:
                        return self._with_plan_context(
                            {
                                **workspace_error,
                                "step_id": step.id,
                                "chunks_used": chunk_index + 1,
                                "executed_steps": executed_steps,
                            }
                        ), observation
                    if feedback.status != "running":
                        self._set_terminal_feedback(feedback.status)
                        return self._with_plan_context(
                            {
                                "status": feedback.status,
                                "step_id": step.id,
                                "chunks_used": chunk_index + 1,
                                "executed_steps": executed_steps,
                                "details": feedback.details,
                            }
                        ), observation
                    if observation.task_success is True:
                        self._agent.steer(
                            UserMessage(
                                content=(
                                    "The environment reports task_success=true. "
                                    "Do not execute another robot action; call "
                                    "finish(success=true) now."
                                )
                            )
                        )
                        return self._with_plan_context(
                            {
                                "status": "success",
                                "step_id": step.id,
                                "chunks_used": chunk_index + 1,
                                "executed_steps": executed_steps,
                            }
                        ), observation
        return self._with_plan_context(
            {
                "status": "running",
                "stop_reason": "chunk_budget",
                "step_id": step.id,
                "chunks_used": self._options.max_vla_chunks,
                "executed_steps": executed_steps,
            }
        ), observation

    def _vla_workspace_error(
        self,
        observation: Observation,
    ) -> dict[str, object] | None:
        """配置 AABB 时，用实际 EEF observation 生成 VLA 硬停止结果。"""
        return self._observed_workspace_error(
            observation,
            action_name="VLA execution",
            violation_reason=(
                "The observed EEF position left the configured workspace; "
                "the remaining VLA action chunk was stopped."
            ),
        )

    def _observed_workspace_error(
        self,
        observation: Observation,
        *,
        action_name: str,
        violation_reason: str,
    ) -> dict[str, object] | None:
        """配置 AABB 时，把真实 EEF observation 转成通用硬停止结果。"""
        config = self._options.ee_pose_adjustment
        if config.workspace_min_m is None or config.workspace_max_m is None:
            return None
        try:
            pose = pose_from_observation(observation)
        except ValueError as error:
            return {
                "status": "controller_failed",
                "reason": f"Cannot verify the EEF workspace during {action_name}: {error}",
                "workspace": self._workspace_summary(),
            }
        if workspace_violation_distance(pose, config) == 0.0:
            return None
        return {
            "status": "out_of_workspace",
            "reason": violation_reason,
            "eef_pose": pose.summary(),
            "workspace": self._workspace_summary(),
        }

    async def _adjust_ee_pose(
        self,
        action: EePoseAdjustmentAction,
        magnitude: float,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None]:
        payload, observation = await self._execute_ee_adjustment(
            action,
            magnitude,
            cancel_event,
            source="adjust_ee_pose",
        )
        return payload, observation

    async def _restage_vla(
        self,
        cancel_event: asyncio.Event | None,
    ) -> tuple[
        dict[str, object],
        Observation | None,
        Observation | None,
    ]:
        """沿当前腕部相机负 Z 轴固定回撤 5 cm，并保留前后观察。"""
        before = self._latest_observation or await self._observe(cancel_event)
        payload, after = await self._execute_ee_adjustment(
            "camera_z_retract",
            _RESTAGE_DISTANCE_M,
            cancel_event,
            source="restage_vla",
        )
        payload["requested_distance_m"] = _RESTAGE_DISTANCE_M
        return payload, before, after

    async def _execute_ee_adjustment(
        self,
        action: EePoseAdjustmentAction,
        magnitude: float,
        cancel_event: asyncio.Event | None,
        *,
        source: str,
    ) -> tuple[dict[str, object], Observation | None]:
        """用现有 7D OSC action 闭环执行一次单轴局部恢复动作。"""
        observation = self._latest_observation or await self._observe(cancel_event)
        config = self._options.ee_pose_adjustment
        unit = "meter" if action in TRANSLATION_ACTIONS else "degree"
        before: EePose | None = None
        after: EePose | None = None
        try:
            before = pose_from_observation(observation)
        except ValueError as error:
            payload = self._adjustment_payload(
                "controller_failed",
                action,
                magnitude,
                unit,
                before,
                after,
                None,
                reason=str(error),
            )
            self._add_physical_outcome_fields(payload, observation)
            return payload, observation
        try:
            target = build_adjustment_target(before, action, magnitude, config)
        except ValueError as error:
            payload = self._adjustment_payload(
                "blocked",
                action,
                magnitude,
                unit,
                before,
                before,
                None,
                reason=str(error),
            )
            self._add_physical_outcome_fields(payload, observation)
            return payload, observation
        status, observation, after, executed_steps, reason = await self._drive_ee_to_target(
            observation,
            before,
            target,
            cancel_event,
            source=source,
            progress_error=lambda current: primary_error(current, target, action),
            minimum_progress=(
                config.translation_tolerance_m * 0.1
                if action in TRANSLATION_ACTIONS
                else config.rotation_tolerance_deg * 0.1
            ),
        )
        payload = self._adjustment_payload(
            status,
            action,
            magnitude,
            unit,
            before,
            after,
            target,
            executed_steps=executed_steps,
            reason=reason,
        )
        self._add_physical_outcome_fields(payload, observation)
        return payload, observation

    async def _return_to_home(
        self,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None, Observation | None]:
        """保持环境和夹爪状态，把机械臂恢复到 episode 初始关节位姿。"""
        self._return_home_after_plan_repair_available = False
        recovering_workspace_violation = self._workspace_violation
        observation = self._latest_observation or await self._observe(cancel_event)
        before_observation = observation
        command = _finite_number(observation.robot_state.get("gripper_command"))
        if command is None:
            command = self._last_gripper_command
        if command is None:
            payload = {
                "status": "controller_failed",
                "home_pose_reached": False,
                "reason": "The latest observation has no finite gripper_command to preserve.",
            }
            if recovering_workspace_violation:
                self._set_workspace_safety_stop(payload, str(payload["reason"]))
            self._add_physical_outcome_fields(payload, observation)
            return self._with_plan_context(payload), before_observation, observation
        command = max(-1.0, min(1.0, command))

        executed_home_steps = 0
        try:
            async with self._physical_lock:
                feedback = await self._options.robot.return_to_home(command, cancel_event)
            executed_home_steps = feedback.executed_steps
            observation = feedback.observation or await self._observe(cancel_event)
            self._remember_observation(observation)
        except Exception as error:
            if not recovering_workspace_violation:
                raise
            reason = f"return_to_home failed during workspace recovery: {error}"
            payload = {
                "status": "controller_failed",
                "home_pose_reached": False,
                "executed_steps": executed_home_steps,
                "reason": reason,
            }
            self._set_workspace_safety_stop(payload, reason)
            self._add_physical_outcome_fields(payload, observation)
            return self._with_plan_context(payload), before_observation, observation
        if feedback.status != "running":
            self._set_terminal_feedback(feedback.status)
        status = str(feedback.details.get("status", "controller_failed"))
        payload = {
            **feedback.details,
            "status": status,
            "executed_steps": feedback.executed_steps,
        }
        if recovering_workspace_violation:
            workspace_error = self._observed_workspace_error(
                observation,
                action_name="return_to_home verification",
                violation_reason=(
                    "The EEF remains outside the configured workspace after "
                    "return_to_home."
                ),
            )
            home_reached = feedback.details.get("home_pose_reached") is True
            if status == "success" and home_reached and workspace_error is None:
                self._workspace_violation = False
            else:
                reason = (
                    str(workspace_error["reason"])
                    if workspace_error is not None
                    else f"return_to_home did not complete successfully: {status}."
                )
                self._set_workspace_safety_stop(payload, reason)
        self._add_physical_outcome_fields(payload, observation)
        return self._with_plan_context(payload), before_observation, observation

    async def _drive_ee_to_target(
        self,
        observation: Observation,
        before: EePose,
        target: EePose,
        cancel_event: asyncio.Event | None,
        *,
        source: str,
        progress_error: Callable[[EePose], float],
        minimum_progress: float,
    ) -> tuple[str, Observation, EePose, int, str | None]:
        """复用 OSC 闭环到达一个 Runtime 生成的安全 EEF target。"""
        config = self._options.ee_pose_adjustment
        if not workspace_motion_is_allowed(before, target, config):
            current_distance = workspace_violation_distance(before, config)
            reason = (
                "The target EEF position is outside the configured workspace."
                if current_distance == 0.0
                else "The current EEF position is outside the configured workspace and "
                "the proposed target does not move it toward safety."
            )
            return (
                "out_of_workspace",
                observation,
                before,
                0,
                reason,
            )
        command = _finite_number(observation.robot_state.get("gripper_command"))
        if command is None:
            command = self._last_gripper_command
        if command is None:
            return (
                "controller_failed",
                observation,
                before,
                0,
                "The latest observation has no finite gripper_command to preserve.",
            )

        previous_error = progress_error(before)
        stalled_steps = 0
        executed_steps = 0
        current = before
        previous_workspace_distance = workspace_violation_distance(before, config)
        async with self._physical_lock:
            for _ in range(config.max_control_steps):
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError
                if target_is_reached(current, target, config):
                    break
                osc_action = build_osc_action(current, target, float(command), config)
                feedback = await self._options.robot.execute_action_chunk(
                    ActionChunk(actions=[osc_action], metadata={"source": source}),
                    cancel_event,
                )
                executed_steps += feedback.executed_steps
                if feedback.observation is None or feedback.executed_steps < 1:
                    return (
                        "controller_failed",
                        observation,
                        current,
                        executed_steps,
                        "The robot controller returned no executed observation.",
                    )
                observation = feedback.observation
                self._remember_observation(observation)
                try:
                    current = pose_from_observation(observation)
                except ValueError as error:
                    return (
                        "controller_failed",
                        observation,
                        current,
                        executed_steps,
                        str(error),
                    )
                current_workspace_distance = workspace_violation_distance(current, config)
                if current_workspace_distance > 0.0 and (
                    previous_workspace_distance == 0.0
                    or current_workspace_distance >= previous_workspace_distance
                ):
                    return (
                        "out_of_workspace",
                        observation,
                        current,
                        executed_steps,
                        "The observed EEF position crossed or moved farther outside the "
                        "configured workspace; motion was stopped.",
                    )
                previous_workspace_distance = current_workspace_distance
                if observation.task_success is True:
                    return (
                        "success",
                        observation,
                        current,
                        executed_steps,
                        "The environment reports global task success.",
                    )
                remaining_error = progress_error(current)
                stalled_steps = (
                    stalled_steps + 1
                    if previous_error - remaining_error < minimum_progress
                    else 0
                )
                previous_error = remaining_error
                if feedback.status in {"failure", "timeout", "cancelled"}:
                    self._set_terminal_feedback(feedback.status)
                    return (
                        "controller_failed",
                        observation,
                        current,
                        executed_steps,
                        "The robot environment stopped during pose adjustment.",
                    )
                if stalled_steps >= 3:
                    return (
                        "blocked",
                        observation,
                        current,
                        executed_steps,
                        "The EEF pose stopped making measurable progress.",
                    )

        if not target_is_reached(current, target, config):
            return (
                "controller_failed",
                observation,
                current,
                executed_steps,
                "The EEF pose did not converge within the control-step budget.",
            )
        return "success", observation, current, executed_steps, None

    async def _open_gripper(
        self,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None]:
        """保持 EEF 控制量为零并短暂发送明确的开夹爪命令。"""
        return await self._execute_gripper_command(-1.0, "open_gripper", cancel_event)

    async def _close_gripper(
        self,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None]:
        """保持 EEF 控制量为零并短暂发送明确的闭夹爪命令。"""
        return await self._execute_gripper_command(1.0, "close_gripper", cancel_event)

    async def _execute_gripper_command(
        self,
        command: float,
        source: str,
        cancel_event: asyncio.Event | None,
    ) -> tuple[dict[str, object], Observation | None]:
        """逐步执行夹爪命令，并在每个真实 post-step observation 后检查 EEF。"""
        observation = self._latest_observation or await self._observe(cancel_event)
        qpos_before = _gripper_qpos(observation)
        executed_steps = 0
        workspace_error = self._observed_workspace_error(
            observation,
            action_name=source,
            violation_reason=(
                f"The observed EEF position is outside the configured workspace; "
                f"{source} was not started."
            ),
        )
        if workspace_error is not None:
            payload = {
                **workspace_error,
                "command": command,
                "executed_steps": 0,
                "gripper_qpos_before": list(qpos_before) if qpos_before else None,
                "gripper_qpos_after": list(qpos_before) if qpos_before else None,
            }
            self._add_physical_outcome_fields(payload, observation)
            return self._with_plan_context(payload), observation

        status = "controller_failed"
        async with self._physical_lock:
            for _ in range(5):
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    self._set_terminal_feedback(status)
                    break
                action = ActionChunk(
                    actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, command]],
                    metadata={"source": source},
                )
                feedback = await self._options.robot.execute_action_chunk(
                    action,
                    cancel_event,
                )
                executed_steps += feedback.executed_steps
                observation = feedback.observation or await self._options.robot.observe(
                    cancel_event
                )
                self._remember_observation(observation)
                if feedback.executed_steps > 0:
                    self._last_gripper_command = command
                workspace_error = self._observed_workspace_error(
                    observation,
                    action_name=source,
                    violation_reason=(
                        "The observed EEF position left the configured workspace; "
                        f"the remaining {source} control steps were stopped."
                    ),
                )
                if workspace_error is not None:
                    qpos_after = _gripper_qpos(observation)
                    payload = {
                        **workspace_error,
                        "command": command,
                        "executed_steps": executed_steps,
                        "gripper_qpos_before": (
                            list(qpos_before) if qpos_before else None
                        ),
                        "gripper_qpos_after": (
                            list(qpos_after) if qpos_after else None
                        ),
                    }
                    self._add_physical_outcome_fields(payload, observation)
                    return self._with_plan_context(payload), observation
                if feedback.status != "running":
                    self._set_terminal_feedback(feedback.status)
                    status = feedback.status
                    break
            else:
                status = "success" if executed_steps > 0 else "controller_failed"

        qpos_after = _gripper_qpos(observation)
        payload = {
            "status": status,
            "command": command,
            "executed_steps": executed_steps,
            "gripper_qpos_before": list(qpos_before) if qpos_before else None,
            "gripper_qpos_after": list(qpos_after) if qpos_after else None,
        }
        self._add_physical_outcome_fields(payload, observation)
        return self._with_plan_context(payload), observation

    def _add_physical_outcome_fields(
        self,
        payload: dict[str, object],
        observation: Observation | None,
    ) -> None:
        """区分控制执行状态与官方环境验证的任务效果。"""
        payload["execution_status"] = payload.get("status", "unknown")
        payload["task_effect_status"] = (
            "verified_task_success"
            if observation is not None and observation.task_success is True
            else "unverified"
        )
        payload["workspace_violation"] = self._workspace_violation

    def _set_workspace_safety_stop(
        self,
        payload: dict[str, object],
        reason: str,
    ) -> None:
        """记录越界恢复失败，并要求 Agent loop 在本轮后立即终止。"""
        self._workspace_violation = True
        self._finish_status = "safety_stop"
        self._finish_summary = reason
        payload["safety_stop"] = True
        payload["safety_stop_reason"] = reason

    def _adjustment_payload(
        self,
        status: str,
        action: EePoseAdjustmentAction,
        magnitude: float,
        unit: str,
        before: EePose | None,
        after: EePose | None,
        target: EePose | None,
        *,
        executed_steps: int = 0,
        reason: str | None = None,
    ) -> dict[str, object]:
        """构造不含图像 payload、可直接写入工具 trace 的恢复结果。"""
        translation = None
        rotation = None
        if before is not None and after is not None:
            if action in TRANSLATION_ACTIONS:
                translation = list(vector_subtract(after.position, before.position))
            elif action in ROTATION_ACTIONS:
                rotation = list(applied_rotation_deg(before, after))
        payload: dict[str, object] = {
            "status": status,
            "action": action,
            "requested_magnitude": magnitude,
            "unit": unit,
            "applied_translation_m": translation,
            "applied_rotation_deg": rotation,
            "ee_pose_before": before.summary() if before is not None else None,
            "ee_pose_after": after.summary() if after is not None else None,
            "target_pose": target.summary() if target is not None else None,
            "executed_steps": executed_steps,
        }
        workspace = self._workspace_summary()
        if workspace is not None:
            payload["workspace"] = workspace
        if reason is not None:
            payload["reason"] = reason
        return self._with_plan_context(payload)

    def _workspace_summary(self) -> dict[str, list[float]] | None:
        """返回 Agent 可审计的绝对 EEF 工作区配置。"""
        config = self._options.ee_pose_adjustment
        if config.workspace_min_m is None or config.workspace_max_m is None:
            return None
        return {
            "min_m": list(config.workspace_min_m),
            "max_m": list(config.workspace_max_m),
        }

    async def _finish(
        self, success: bool, summary: str, cancel_event: asyncio.Event | None
    ) -> tuple[dict[str, object], Observation | None]:
        observation = await self._observe(cancel_event)
        if (
            success
            and observation.task_success is False
            and self._options.finalizer is not None
            and self._task_plan is not None
            and all(step.status == "completed" for step in self._task_plan.steps)
        ):
            # 通用 Runtime 不理解 benchmark 的等待步数，只在计划完成后委托环境适配层
            # 推进其官方终态验证；该调用会改变物理状态，因此仍受物理锁保护。
            async with self._physical_lock:
                observation = await self._options.finalizer.finalize_episode(cancel_event)
            self._remember_observation(observation)
        if success and observation.task_success is False:
            status = "failure"
        elif success and observation.task_success is None:
            status = "uncertain"
        elif success:
            status = "success"
        else:
            status = "failure"
        self._finish_status = status
        self._finish_summary = summary
        return self._with_plan_context(
            {"status": status, "summary": summary}
        ), observation

    def _remember_observation(self, observation: Observation) -> None:
        """保留最新观察和最后一个可复用的夹爪命令。"""
        self._latest_observation = observation
        command = _finite_number(observation.robot_state.get("gripper_command"))
        if command is not None:
            self._last_gripper_command = command
        config = self._options.ee_pose_adjustment
        if config.workspace_min_m is None or config.workspace_max_m is None:
            return
        try:
            pose = pose_from_observation(observation)
        except ValueError:
            return
        if workspace_violation_distance(pose, config) > 0.0:
            self._workspace_violation = True

    def _set_terminal_feedback(self, status: str) -> None:
        """RobotPort 明确终止时直接结束 episode，不再请求模型决策。"""
        summaries = {
            "failure": "Robot environment reached a terminal failure state.",
            "timeout": "Robot environment timed out.",
            "cancelled": "Robot execution was cancelled.",
        }
        if status in summaries:
            self._finish_status = status
            self._finish_summary = summaries[status]

    async def _before_tool_call(
        self,
        context: BeforeToolCallContext[object],
        cancel_event: asyncio.Event | None,
    ) -> BeforeToolCallResult | None:
        del cancel_event
        self._record_pending_reflection(context.assistant_message)
        self._tool_calls += 1
        self._tool_history.append(context.tool_call.name)
        if (
            self._workspace_violation
            and context.tool_call.name != "return_to_home"
        ):
            return self._block_tool_call(
                context.tool_call.id,
                "The EEF is outside the configured workspace. Runtime safety lock "
                "allows only return_to_home; no other tool can execute until joint-home "
                "recovery succeeds.",
            )
        if (
            context.tool_call.name
            in {
                "execute_vla",
                "restage_vla",
                "adjust_ee_pose",
                "close_gripper",
                "open_gripper",
                "return_to_home",
            }
            and self._latest_observation is not None
            and self._latest_observation.task_success is True
            and not (
                self._workspace_violation
                and context.tool_call.name == "return_to_home"
            )
        ):
            return self._block_tool_call(
                context.tool_call.id,
                "The environment already reports task success; call finish(success=true).",
            )
        if context.tool_call.name == "execute_vla":
            if self._task_plan is None:
                return self._block_tool_call(
                    context.tool_call.id,
                    "No robot action executed because no TaskPlan exists. Create the full "
                    "TaskPlan before calling execute_vla; do not treat this attempt as "
                    "completed.",
                )
            running = next(
                (step for step in self._task_plan.steps if step.status == "running"),
                None,
            )
            if running is None:
                return self._block_tool_call(
                    context.tool_call.id,
                    "TaskPlan has no running step; call finish instead of executing VLA.",
                )
            step_id = str(context.args.get("step_id", "")).strip()
            if step_id != running.id:
                return self._block_tool_call(
                    context.tool_call.id,
                    f"execute_vla must use the running TaskPlan step_id: {running.id}",
                )
        if context.tool_call.name in {
            "restage_vla",
            "adjust_ee_pose",
            "close_gripper",
            "open_gripper",
            "return_to_home",
        }:
            if self._task_plan is None and not (
                self._workspace_violation
                and context.tool_call.name == "return_to_home"
            ):
                return self._block_tool_call(
                    context.tool_call.id,
                    "No recovery action was executed because no TaskPlan exists.",
                )
            if not self._current_step_has_executed_action and not (
                (
                    self._workspace_violation
                    or self._return_home_after_plan_repair_available
                )
                and context.tool_call.name == "return_to_home"
            ):
                return self._block_tool_call(
                    context.tool_call.id,
                    f"{context.tool_call.name} is available only after execute_vla has "
                    "attempted the current running step.",
                )
        calls = [block for block in context.assistant_message.content if hasattr(block, "name")]
        if calls and context.tool_call.id != calls[0].id:
            return self._block_tool_call(
                context.tool_call.id,
                "Only one robot tool call is allowed per assistant turn.",
            )
        return None

    def _block_tool_call(self, tool_call_id: str, reason: str) -> BeforeToolCallResult:
        entry = self._find_trace_entry(tool_call_id)
        if entry is not None:
            entry.update(status="blocked", reason=reason)
        return BeforeToolCallResult(block=True, reason=reason)

    def _record_agent_event(self, event: AgentEvent[object]) -> None:
        """记录不含图像 payload 的工具调用事件，供终端展示和持久化。"""
        if isinstance(event, ToolExecutionStartEvent):
            entry: dict[str, object] = {
                "call_id": event.tool_call_id,
                "name": event.tool_name,
                "args": event.args,
                "status": "running",
            }
            decision_summary = event.args.get("decision_summary")
            if isinstance(decision_summary, str) and decision_summary.strip():
                entry["decision_summary"] = decision_summary.strip()
            self._tool_trace.append(entry)
        elif isinstance(event, ToolExecutionEndEvent):
            trace_entry = self._find_trace_entry(event.tool_call_id)
            if trace_entry is None:
                return
            if trace_entry.get("status") != "blocked":
                details = event.result.details
                rejected = (
                    isinstance(details, dict) and details.get("status") == "rejected"
                )
                trace_entry["status"] = (
                    "error" if event.is_error else "rejected" if rejected else "completed"
                )
            trace_entry["is_error"] = event.is_error
            trace_entry["result"] = event.result.details
            texts = [
                content.text for content in event.result.content if isinstance(content, TextContent)
            ]
            if texts:
                trace_entry["text"] = texts
            image_count = sum(
                isinstance(content, ImageContent) for content in event.result.content
            )
            if image_count:
                trace_entry["image_count"] = image_count
            arguments = trace_entry.get("args")
            self._episode_memory.append_tool_event(
                call_id=event.tool_call_id,
                tool=event.tool_name,
                arguments=arguments if isinstance(arguments, dict) else {},
                details=event.result.details,
                is_error=event.is_error,
            )

    def _find_trace_entry(self, tool_call_id: str) -> dict[str, object] | None:
        return next(
            (entry for entry in reversed(self._tool_trace) if entry["call_id"] == tool_call_id),
            None,
        )

    def _result_details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "tool_history": list(self._tool_history),
            "tool_trace": [dict(entry) for entry in self._tool_trace],
            "episode_memory": self._episode_memory.summary(),
            "state_reflections": [dict(item) for item in self._state_reflections],
            "workspace_violation": self._workspace_violation,
            "safety_stop": self._finish_status == "safety_stop",
        }
        if self._options.observer is not None:
            details["observer_trace"] = self._options.observer.trace
        if self._task_plan is not None:
            details["task_plan"] = self._task_plan.summary()
        if self._options.task_execution_skill is not None:
            details["task_execution_skill"] = self._options.task_execution_skill.summary()
        if self._options.plan_advance_approval is not None:
            details["human_intervention"] = bool(self._supervision_events)
            details["supervision_events"] = [
                dict(item) for item in self._supervision_events
            ]
        return details

    def _require_task_plan(self) -> TaskPlan:
        plan = self._task_plan
        if plan is None:
            raise RuntimeError("TaskPlan must be created before robot execution")
        return plan

    def _with_plan_context(self, payload: dict[str, object]) -> dict[str, object]:
        return {**payload, **self._plan_context()}

    def _plan_context(self) -> dict[str, object]:
        """返回给 Planner 的精简计划状态，不重复整份历史。"""
        plan = self._task_plan
        if plan is None:
            return {
                "active_step": None,
                "plan_progress": {"completed": [], "running": None, "pending": []},
            }
        running = next((step for step in plan.steps if step.status == "running"), None)
        return {
            "active_step": running.summary() if running is not None else None,
            "plan_progress": {
                "completed": [step.id for step in plan.steps if step.status == "completed"],
                "running": running.id if running is not None else None,
                "pending": [step.id for step in plan.steps if step.status == "pending"],
            },
        }

    @staticmethod
    def _task_plan_invariant_error(
        steps: list[TaskStep], *, allow_terminal: bool = False
    ) -> str | None:
        """校验计划只呈现 completed 前缀、单一 running、pending 后缀。"""
        if not steps:
            return "TaskPlan must contain at least one step."
        running_indices = [
            index for index, step in enumerate(steps) if step.status == "running"
        ]
        if not running_indices:
            if allow_terminal and all(step.status == "completed" for step in steps):
                return None
            return "TaskPlan must contain exactly one running step."
        if len(running_indices) != 1:
            return "TaskPlan must contain exactly one running step."
        running_index = running_indices[0]
        if any(step.status != "completed" for step in steps[:running_index]):
            return "Every step before the running step must be completed."
        if any(step.status != "pending" for step in steps[running_index + 1 :]):
            return "Every step after the running step must be pending."
        return None

    def _plan_rejection(self, reason: str, *, tool_name: str) -> dict[str, object]:
        """拒绝非法计划变更，同时返回可供下一轮纠正的确定性上下文。"""
        plan = self._task_plan
        valid_next_step_ids: list[str] = []
        valid_reopen_step_ids: list[str] = []
        if plan is not None:
            running = next(
                (step for step in plan.steps if step.status == "running"), None
            )
            if running is not None:
                running_index = plan.steps.index(running)
                pending = next(
                    (
                        step
                        for step in plan.steps[running_index + 1 :]
                        if step.status == "pending"
                    ),
                    None,
                )
                if pending is not None:
                    valid_next_step_ids.append(pending.id)
                valid_reopen_step_ids.extend(
                    step.id
                    for step in plan.steps[:running_index]
                    if step.status == "completed"
                )
        return {
            "status": "rejected",
            "reason": reason,
            "task_plan": plan.summary() if plan is not None else None,
            **self._plan_context(),
            "valid_next_step_ids": valid_next_step_ids,
            "valid_reopen_step_ids": valid_reopen_step_ids,
            "retry_instruction": (
                f"Re-read the returned TaskPlan and call {tool_name} again with corrected "
                "arguments. Do not repeat unchanged invalid arguments."
            ),
        }

    @staticmethod
    def _running_task_step(plan: TaskPlan) -> TaskStep:
        running = [step for step in plan.steps if step.status == "running"]
        if len(running) != 1:
            raise RuntimeError("TaskPlan must contain exactly one running step")
        return running[0]

    async def _prepare_next_turn(
        self,
        context: PrepareNextTurnContext[object],
        cancel_event: asyncio.Event,
    ) -> AgentLoopTurnUpdate[object] | None:
        """在物理工具后插入无工具状态反思，并在完成后恢复正常决策上下文。"""
        if cancel_event.is_set():
            return None
        if self._reflection_source is not None:
            reflection = self._record_pending_reflection(context.message)
            if self._pre_reflection_context is not None:
                decision_messages = [
                    *self._pre_reflection_context.messages,
                    context.message,
                ]
            else:
                decision_messages = list(context.context.messages)
            self._pre_reflection_context = None
            self._pending_decision_prompt = (
                "Latest Grounded State Reflection:\n"
                f"{reflection}\n\n"
                "Decide the next action based on the current plan, episode history, "
                "latest physical evidence, and this reflection. Use one normal tool."
            )
            return AgentLoopTurnUpdate(
                context=AgentContext(
                    system_prompt=self._decision_system_prompt,
                    messages=decision_messages,
                    tools=list(self._decision_tools),
                )
            )

        result = next(
            (
                item
                for item in context.tool_results
                if item.tool_name in _REFLECTION_TOOL_NAMES and not item.is_error
            ),
            None,
        )
        if result is None:
            return None
        self._reflection_source = (result.tool_call_id, result.tool_name)
        reflection_messages = list(context.context.messages)
        decision_messages = list(context.context.messages)
        for index, message in enumerate(reflection_messages):
            if not isinstance(message, ToolResultMessage):
                continue
            content = sanitize_reflection_content(message.content)
            if message.tool_call_id == result.tool_call_id:
                content = build_reflection_visual_evidence(content)
                decision_messages[index] = message.model_copy(
                    update={
                        "content": build_decision_visual_evidence(
                            message.content,
                            content,
                        )
                    }
                )
            reflection_messages[index] = message.model_copy(
                update={"content": content}
            )
        self._pre_reflection_context = AgentContext(
            system_prompt=context.context.system_prompt,
            messages=decision_messages,
            tools=list(context.context.tools),
        )
        return AgentLoopTurnUpdate(
            context=AgentContext(
                system_prompt=STATE_REFLECTION_PROMPT,
                messages=reflection_messages,
                tools=[],
            )
        )

    def _record_pending_reflection(self, message: AssistantMessage) -> str:
        """把独立反思响应归属到上一次物理调用，并返回其文本。"""
        if self._reflection_source is None:
            return ""
        reflection = "\n".join(
            block.text.strip()
            for block in message.content
            if isinstance(block, TextContent) and block.text.strip()
        )
        if not reflection:
            reflection = "Grounded State Reflection: No textual state reflection was produced."
        call_id, tool = self._reflection_source
        self._state_reflections.append(
            {"after_call_id": call_id, "tool": tool, "text": reflection}
        )
        self._reflection_source = None
        return reflection

    async def _should_stop_after_turn(
        self, context: ShouldStopAfterTurnContext[object]
    ) -> bool:
        self._turns += 1
        if self._pending_decision_prompt is not None:
            prompt = self._pending_decision_prompt
            self._pending_decision_prompt = None
            self._agent.follow_up(UserMessage(content=prompt))
            return False
        if self._finish_status is not None or self._turns >= self._options.max_turns:
            return True
        if context.tool_results:
            return False

        if self._task_plan is None:
            instruction = (
                "Your text-only response cannot change state or end the episode. Call "
                "create_task_plan now with the complete plan and exactly one running step."
            )
        else:
            running = next(
                (step for step in self._task_plan.steps if step.status == "running"),
                None,
            )
            if running is None:
                instruction = (
                    "Your text-only response cannot end the episode. The plan has no "
                    "running step; call finish using success=true only if task_success=true."
                )
            elif not self._current_step_has_executed_action:
                instruction = (
                    "Your text-only response cannot change state or end the episode. Call "
                    "execute_vla now with the running step_id: "
                    f"{running.id}"
                )
            else:
                instruction = (
                    "Your text-only response cannot end the episode. Use the latest "
                    "physical evidence to choose one available tool. Repeat execute_vla "
                    "with the running step_id when appropriate, call advance_task_plan "
                    "only if the returned visual evidence supports completion, call "
                    "repair_task_plan if evidence shows an earlier completed prerequisite "
                    "was lost, or "
                    "call finish if the episode is ready to terminate. Running step_id: "
                    f"{running.id}"
                )
        self._agent.follow_up(UserMessage(content=instruction))
        return False

    def _result_from_agent(self) -> EpisodeResult:
        if self._finish_status is not None:
            return EpisodeResult(
                status=cast(EpisodeStatus, self._finish_status),
                summary=self._finish_summary,
                turns=self._turns,
                tool_calls=self._tool_calls,
                details=self._result_details(),
            )
        if self._agent.state.error_message:
            return EpisodeResult(
                status="error",
                summary=self._agent.state.error_message,
                turns=self._turns,
                tool_calls=self._tool_calls,
                details=self._result_details(),
            )
        if (
            self._agent.state.messages
            and isinstance(self._agent.state.messages[-1], AssistantMessage)
            and self._agent.state.messages[-1].stop_reason == "aborted"
        ):
            return EpisodeResult(
                status="cancelled",
                summary="Agent run was aborted.",
                turns=self._turns,
                tool_calls=self._tool_calls,
                details=self._result_details(),
            )
        status: EpisodeStatus = (
            "max_turns" if self._turns >= self._options.max_turns else "incomplete"
        )
        return EpisodeResult(
            status=status,
            summary="Agent stopped without calling finish.",
            turns=self._turns,
            tool_calls=self._tool_calls,
            details=self._result_details(),
        )
