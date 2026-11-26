"""把控制代和交接边界绑定到冻结 π0.5 与 RoboAgent 的机器人端口。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import dist
from time import monotonic

from benchmark.libero._vendor.openpi_client import msgpack_numpy
from benchmark.robomemarena.client.observation import build_openpi_input
from benchmark.robomemarena.preprocessing import validate_action
from websockets.asyncio.client import ClientConnection

from roboagent.observer.types import ObservationEvent
from roboagent.ports import RobotPort
from roboagent.types import ActionChunk, Observation
from roboagent.utils.ee_pose import (
    EePoseAdjustmentConfig,
    build_adjustment_target,
    build_osc_action,
    pose_from_observation,
    workspace_violation_distance,
)

from .memory import grasp_preserved
from .schema import ActiveTask

PI05 = "pi05_robomemarena"
HOLD_POSITION_STEPS = 20


class OpenPiPolicy:
    """沿用 OpenPI 的二进制协议与输入适配，异步等待动作块。"""

    def __init__(self, socket: ClientConnection, chunk_size: int, timeout: float) -> None:
        self.socket = socket
        self.packer = msgpack_numpy.Packer()
        self.chunk_size = chunk_size
        self.timeout = timeout

    async def infer(self, instruction: str, observation: Observation) -> list[list[float]]:
        """提交一次原生输入，只返回本次接口允许执行的动作前缀。"""
        payload = build_openpi_input(instruction, observation)
        async with asyncio.timeout(self.timeout):
            await self.socket.send(self.packer.pack(payload))
            raw = await self.socket.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"OpenPI inference failed: {raw}")
        values = msgpack_numpy.unpackb(raw)["actions"]
        actions = [validate_action(row) for row in values[:self.chunk_size]]
        if not actions:
            raise ValueError("OpenPI returned an empty action chunk")
        return actions


@dataclass(frozen=True, slots=True)
class HandoffRequest:
    """锁定旧控制代和下一候选的版本。"""

    previous_epoch: int
    next_task: ActiveTask
    requested_at: float


@dataclass(frozen=True, slots=True)
class HandoffAck:
    """执行器在已授权动作边界回执。"""

    request: HandoffRequest
    last_step: int
    at: float


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """动作执行回执与实际物理步的台账。"""

    task: ActiveTask
    status: str
    infer_started: float
    infer_finished: float
    first_action: float | None
    finished: float
    start_step: int
    end_step: int


class Executor:
    """持有唯一控制代，只派发当前授权下的物理动作。"""

    def __init__(
        self,
        robot: RobotPort,
        policy: OpenPiPolicy,
        first: ObservationEvent,
        workspace: EePoseAdjustmentConfig,
        control_hz: float,
    ) -> None:
        self.robot = robot
        self.policy = policy
        self.latest = first
        self.workspace = workspace
        self.period = 1 / control_hz
        self.active: ActiveTask | None = None
        self.pending: HandoffRequest | None = None
        self.busy = False
        self.cancel_event = asyncio.Event()

    def inside_workspace(self) -> bool:
        """检查当前实测末端位置。"""
        pose = pose_from_observation(self.latest.observation)
        return workspace_violation_distance(pose, self.workspace) == 0

    def request_handoff(self, candidate: ActiveTask) -> HandoffRequest:
        """禁止旧执行实例领取下一块，保留正在执行的前缀。"""
        assert self.active is not None
        self.pending = HandoffRequest(self.active.epoch, candidate, monotonic())
        return self.pending

    def acknowledge(self, request: HandoffRequest) -> HandoffAck | None:
        """只在旧执行器完成边界且请求仍匹配时返回 ACK。"""
        if self.busy or self.pending != request or self.active is None:
            return None
        if self.active.epoch != request.previous_epoch:
            return None
        return HandoffAck(request, self.latest.sequence, monotonic())

    def commit(self, candidate: ActiveTask, ack: HandoffAck | None = None) -> None:
        """用已复核的交接令牌发布新控制代。"""
        if self.busy or candidate.start_step != self.latest.sequence:
            raise ValueError("Commit requires the current action boundary")
        if self.active is not None and (
            ack is None or self.pending is None or ack.request != self.pending
            or replace(candidate, start_step=-1) != ack.request.next_task
            or ack.last_step != self.latest.sequence
            or self.active.epoch != ack.request.previous_epoch
            or candidate.epoch != self.active.epoch + 1
        ):
            raise ValueError("Handoff ACK does not match the active epoch and candidate")
        self.active = candidate
        self.pending = None

    async def execute(
        self, task: ActiveTask, remaining: int, deadline: float,
        publish: Callable[[ObservationEvent], None],
    ) -> ChunkResult:
        """核对授权，执行一次原生动作块或一次受限的恢复操作。"""
        if self.busy or self.active != task or self.pending is not None:
            raise ValueError("Execution requires the current epoch and an unclaimed boundary")
        start_step = self.latest.sequence
        began = monotonic()
        first_action = None
        self.busy = True
        try:
            if task.mode == "return_to_home":
                outside_workspace = not self.inside_workspace()
                grip = float(self.latest.observation.robot_state["gripper_command"])
                async with asyncio.timeout_at(deadline):
                    feedback = await self.robot.return_to_home(max(-1.0, min(grip, 1.0)), self.cancel_event)
                if feedback.observation is None or not 0 <= feedback.executed_steps <= remaining:
                    raise RuntimeError("Invalid return-to-home feedback")
                if feedback.executed_steps:
                    self.latest = ObservationEvent(start_step + feedback.executed_steps,
                                                   monotonic(), feedback.observation)
                    publish(self.latest)
                status = feedback.status
                if status == "running":
                    home_reached = feedback.details.get("home_pose_reached") is True
                    if not self.inside_workspace() or (outside_workspace and not home_reached):
                        status = "safety_stop"
                    else:
                        status = "home_reached" if home_reached else "home_incomplete"
                return ChunkResult(task, status, began, began, began, monotonic(), start_step,
                                   self.latest.sequence)
            target = None
            if task.mode == "policy":
                actions = await self.policy.infer(task.goal.instruction, self.latest.observation)
            elif task.mode == "hold_position":
                grip = float(self.latest.observation.robot_state["gripper_command"])
                actions = [[0.0] * 6 + [max(-1.0, min(grip, 1.0))]] * HOLD_POSITION_STEPS
            elif task.mode in {"open_gripper", "close_gripper", "release_gripper"}:
                grip = 1.0 if task.mode == "close_gripper" else -1.0
                action_count = 10 if task.mode == "close_gripper" and task.goal.is_drawer_motion else 5
                actions = [[0.0] * 6 + [grip]] * action_count
            else:
                pose = pose_from_observation(self.latest.observation)
                if task.mode in {"restage_vla", "pull_drawer"}:
                    target = build_adjustment_target(pose, "camera_z_retract", 0.05, self.workspace)
                else:
                    target = build_adjustment_target(pose, "world_z_up", 0.025, self.workspace)
                if workspace_violation_distance(target, self.workspace) > 0:
                    return ChunkResult(task, "restage_unavailable", began, monotonic(), None,
                                       monotonic(), start_step, start_step)
                if task.mode == "pull_drawer" and abs(target.position[2] - pose.position[2]) > 0.015:
                    return ChunkResult(task, "pull_direction_unavailable", began, monotonic(), None,
                                       monotonic(), start_step, start_step)
                actions = [None] * self.workspace.max_control_steps
            inferred = monotonic()
            status = "running"
            next_action = inferred
            for action in actions[:remaining]:
                await asyncio.sleep(max(0, next_action - monotonic()))
                if self.active != task or self.cancel_event.is_set():
                    status = "cancelled"
                    break
                if monotonic() >= deadline:
                    status = "timeout"
                    break
                if not self.inside_workspace():
                    status = "safety_stop"
                    break
                if task.mode == "pull_drawer" and not grasp_preserved((self.latest,), self.latest.sequence):
                    status = "drawer_handle_released"
                    break
                if action is None:
                    assert target is not None
                    obs = self.latest.observation
                    pose = pose_from_observation(obs)
                    if dist(pose.position, target.position) <= self.workspace.translation_tolerance_m:
                        status = "retreated"
                        break
                    grip = float(obs.robot_state.get("gripper_command", -1.0))
                    action = build_osc_action(pose, target, grip, self.workspace)
                dispatched = monotonic()
                if first_action is None:
                    first_action = dispatched
                feedback = await self.robot.execute_action_chunk(ActionChunk([action]), self.cancel_event)
                if feedback.executed_steps != 1 or feedback.observation is None:
                    raise RuntimeError("The simulator did not acknowledge the dispatched action")
                self.latest = ObservationEvent(
                    self.latest.sequence + 1, monotonic(), feedback.observation,
                )
                publish(self.latest)
                status = feedback.status
                if status == "running" and not self.inside_workspace():
                    status = "safety_stop"
                if status != "running":
                    break
                next_action = dispatched + self.period
            if target is not None and status == "running":
                delta = dist(pose_from_observation(self.latest.observation).position, target.position)
                status = "retreated" if delta <= self.workspace.translation_tolerance_m else "restage_incomplete"
            return ChunkResult(task, status, began, inferred, first_action, monotonic(), start_step,
                               self.latest.sequence)
        finally:
            self.busy = False
