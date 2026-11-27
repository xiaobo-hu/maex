"""维护工作窗口、事件账本和当前执行器的程序性子树。"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboagent.observer.sampler import detect_gripper_keyframes
from roboagent.observer.types import ObservationEvent
from roboagent.types import Observation, summarize_robot_state

from .schema import (
    LOOKAHEAD_PROCEDURES,
    LOOKAHEAD_STEPS,
    ActiveTask,
    EvidenceWindow,
    ExecutionMode,
    Facet,
    Fact,
    FrontierNode,
    Goal,
    Judgment,
    Outcome,
    Plan,
    PlanRequest,
    Predicate,
    ProcedureReference,
    TaskStepReference,
    TreePatch,
)

if TYPE_CHECKING:
    from .procedural import ProcedureSnapshot, RecoveryCondition, RecoveryProcedure

CLOSED_GRIP_M = 0.006
OPEN_GRIP_M = 0.077
GRIPPER_KEYFRAME_DELTA_M = 0.008


def gripper_width(observation: Observation) -> float | None:
    """读取与 RoboAgent 一致的实测双指开度。"""
    width = summarize_robot_state(observation.robot_state).get("gripper_width_m")
    return float(width) if isinstance(width, (int, float)) else None


def grasp_preserved(frames: Iterable[ObservationEvent], since: int) -> bool:
    """用实际夹爪开度与开爪命令核对视觉抓持是否仍可用于交接。"""
    observed = False
    for frame in frames:
        if frame.sequence < since:
            continue
        state = frame.observation.robot_state
        width = gripper_width(frame.observation)
        if width is None or not CLOSED_GRIP_M < width < OPEN_GRIP_M or state.get("gripper_command", 0) < -0.1:
            return False
        observed = True
    return observed


def grasp_settled(frames: Iterable[ObservationEvent], step: int) -> bool:
    """排除夹爪空闭合途中短暂经过物体宽度的情况。"""
    recent = tuple(frame for frame in frames if step - 4 <= frame.sequence <= step)
    if len(recent) != 5 or not grasp_preserved(recent, step - 4):
        return False
    widths = [sum(abs(value) for value in frame.observation.robot_state["gripper_qpos"])
              for frame in recent]
    return max(widths) - min(widths) <= 0.004


class ProcedureTemplate(BaseModel):
    """定义核验条件及可以实例化的恢复分支。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: tuple[Facet, ...]
    recoveries: tuple[Literal["retry", "retreat"], ...]
    focus: str = Field(default="", max_length=240)


class ProcedureLibrary(BaseModel):
    """从固定版本的程序库实例化受限子树。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    executor: str
    templates: dict[Predicate, ProcedureTemplate]

    @classmethod
    def load(cls, path: Path) -> ProcedureLibrary:
        """读取当前 episode 使用的程序库。"""
        library = cls.model_validate_json(path.read_text())
        if set(library.templates) != set(Predicate):
            raise ValueError("The procedure library must cover every registered predicate")
        for predicate, template in library.templates.items():
            if predicate == Predicate.PLACED:
                required = {Facet.IDENTITY, Facet.RELEASE, Facet.DESTINATION}
            else:
                required = {Facet.IDENTITY, Facet.EFFECT}
            if predicate == Predicate.OPENED:
                required.add(Facet.APERTURE)
            if predicate == Predicate.POURED:
                required.add(Facet.UPRIGHT)
            if not required <= set(template.evidence) or "retry" not in template.recoveries:
                raise ValueError(f"{predicate} template omits a required check or retry branch")
        return library


@dataclass(frozen=True, slots=True)
class Branch:
    """围绕一步物理目标准备成功、失败和未决路径。"""

    successor: int | None
    evidence: tuple[Facet, ...]
    recoveries: tuple[Literal["retry", "retreat"], ...]
    recovery: Literal["retry", "retreat"] = "retry"
    unresolved: Literal["inspect"] = "inspect"
    focus: str = ""
    revision: int = 0
    request_id: int | None = None
    recovery_procedure: str | None = None
    bound_goal: Goal | None = None
    inspect_focus: str = ""
    procedure_refs: tuple[ProcedureReference, ...] = ()
    preferred_procedures: tuple[str, ...] = ()
    task_example: TaskStepReference | None = None


class IntentionTree:
    """组合有序的任务约束和有限深度的结果子树。"""

    def __init__(
        self, episode_id: str, plan: Plan, library: ProcedureLibrary, procedures: ProcedureSnapshot | None = None,
    ) -> None:
        self.id = f"{episode_id}:tree"
        self.episode_id = episode_id
        self.plan = plan
        self.library = library
        self.procedures = procedures
        self.recovery_details = {item.id: item for item in procedures.recoveries} if procedures is not None else {}
        self.consulted: set[tuple[int, Goal, str]] = set()
        self.branches = tuple(self.new_branch(step) for step in range(len(plan.goals)))

    def new_branch(self, step: int, goal: Goal | None = None, revision: int = 0) -> Branch:
        """实例化单步核验、恢复、补证和指向下一层的成功路径。"""
        template = self.library.templates[self.plan.goals[step].predicate]
        refs = tuple(
            item.reference() for item in self.recovery_details.values() if item.applies_to(step, goal)
        ) if goal is not None else ()
        return Branch(
            successor=step + 1 if step + 1 < len(self.plan.goals) else None,
            evidence=template.evidence,
            recoveries=template.recoveries,
            focus=template.focus,
            revision=revision,
            bound_goal=goal,
            procedure_refs=refs,
            task_example=self.task_example_for(step) if goal is not None else None,
        )

    def task_example_for(self, step: int) -> TaskStepReference | None:
        """在有序目标及槽位相容时挂载历史计划的对应层。"""
        exemplar = self.procedures.exemplar if self.procedures is not None else None
        if exemplar is None or exemplar.plan is None or step >= len(exemplar.plan.goals):
            return None
        goal = self.plan.goals[step]
        prior = exemplar.plan.goals[step]
        if prior.model_copy(update={"instruction": goal.instruction}) == goal and (
            "$drawer" not in (goal.target, goal.destination) or exemplar.plan.drawer_selection == self.plan.drawer_selection
        ):
            return TaskStepReference(exemplar.episode_id, exemplar.task.seed, step)
        return None

    def replace_branch(self, step: int, branch: Branch) -> None:
        """发布当前步骤的结构更新。"""
        self.branches = (*self.branches[:step], branch, *self.branches[step + 1:])

    def bind_branch(self, step: int, goal: Goal) -> Branch:
        """绑定改变时重建该层目录并撤回依赖旧角色的补丁。"""
        branch = self.branches[step]
        if branch.bound_goal != goal:
            branch = self.new_branch(step, goal, branch.revision + int(branch.bound_goal is not None))
            self.replace_branch(step, branch)
        return branch

    def branch_for(self, step: int, memory: EpisodicMemory) -> Branch:
        """读取该层最新有效的结构，保留尚待观测绑定的角色槽位。"""
        return self.bind_branch(step, self.goal(step, memory) or self.plan.goals[step])

    def prepare_recovery(self, active: ActiveTask, condition: RecoveryCondition) -> RecoveryProcedure | None:
        """按进入条件从该层目录加载详细规则，每个绑定只审查一次。"""
        branch = self.bind_branch(active.step, active.goal)
        preferences = {reference: rank for rank, reference in enumerate(branch.preferred_procedures)}
        candidates = sorted(branch.procedure_refs, key=lambda ref: (
            ref.scope != "task", preferences.get(ref.id, len(preferences)),
        ))
        for reference in candidates:
            procedure = self.recovery_details[reference.id]
            key = active.step, active.goal, procedure.id
            if key in self.consulted or not procedure.matches(active.step, active.goal, condition):
                continue
            self.consulted.add(key)
            self.replace_branch(active.step, replace(branch, recovery_procedure=procedure.id))
            return procedure
        return None

    def goal(self, step: int, memory: EpisodicMemory) -> Goal | None:
        """用已核查的放置与抽屉内容绑定延后的角色槽位。"""
        goal = self.plan.goals[step]
        if goal.destination_from is not None:
            source = memory.facts.get(goal.destination_from)
            if source is None:
                return None
            destination = source.goal.destination
            instruction = goal.instruction
            if "$destination" not in instruction:
                instruction = f"place {goal.target} at $destination"
            goal = goal.model_copy(update={
                "destination": destination, "destination_from": None,
                "instruction": instruction.replace("$destination", destination),
            })
        if "$drawer" not in (goal.target, goal.destination):
            return goal
        selection = self.plan.drawer_selection
        assert selection is not None
        if any(candidate not in memory.drawer_contents for candidate in selection.candidates):
            return None
        drawer = next(
            (
                candidate for candidate in selection.candidates
                if memory.drawer_contents[candidate].state == selection.condition
            ),
            None,
        )
        if drawer is None:
            return None
        return goal.model_copy(update={
            "target": goal.target.replace("$drawer", drawer),
            "destination": goal.destination.replace("$drawer", drawer),
            "instruction": goal.instruction.replace("$drawer", drawer),
        })

    def inspecting_drawer(self, step: int) -> str | None:
        """返回当前检查阶段需要读取内容的那一层抽屉。"""
        selection = self.plan.drawer_selection
        goal = self.plan.goals[step]
        if selection is not None and step < 2 * len(selection.candidates) and goal.predicate == Predicate.OPENED:
            return goal.target
        return None

    def request(
        self, request_id: int, active: ActiveTask, memory: EpisodicMemory,
        attempted: set[tuple[int, Goal, int]] | None = None,
    ) -> PlanRequest:
        """从当前步骤沿成功路径展开三层，只请求仍缺的结果分支。"""
        dependencies = []
        frontier = []
        for step in range(active.step, min(active.step + LOOKAHEAD_STEPS, len(self.plan.goals))):
            resolved = self.goal(step, memory)
            goal = resolved or self.plan.goals[step]
            branch = self.bind_branch(step, goal)
            frontier.append(FrontierNode(
                step, goal, step - 1 if step > active.step else None, branch.successor, resolved is None,
                branch.recoveries, branch.procedure_refs[:LOOKAHEAD_PROCEDURES], branch.task_example,
            ))
            dependency = step, goal, branch.revision
            if step not in memory.facts and branch.request_id is None and (attempted is None or dependency not in attempted):
                dependencies.append(dependency)
        return PlanRequest(
            request_id, self.episode_id, self.id, active.step, active.epoch, memory.version,
            self.library.version, tuple(dependencies), tuple(frontier),
            self.procedures.revision if self.procedures is not None else None,
        )

    def reconcile(self, request: PlanRequest, patch: TreePatch, memory: EpisodicMemory) -> dict[int, str]:
        """逐节点接续相容的分支，略过已确认的前缀。"""
        if (
            request.episode_id != self.episode_id
            or request.tree_id != self.id
            or request.procedures != self.library.version
            or request.experience_revision != (self.procedures.revision if self.procedures is not None else None)
        ):
            return {proposal.step: "foreign_request" for proposal in patch.branches}
        dependencies = {step: (goal, revision) for step, goal, revision in request.dependencies}
        advertised = {node.step: {ref.id for ref in node.procedure_refs} for node in request.frontier}
        decisions: dict[int, str] = {}
        for proposal in patch.branches:
            step = proposal.step
            if step not in dependencies:
                decisions[step] = "outside_anchor"
                continue
            if step in memory.facts:
                decisions[step] = "completed_prefix" if step < memory.progress else "already_verified"
                continue
            goal, revision = dependencies[step]
            branch = self.branch_for(step, memory)
            if goal != branch.bound_goal or revision != branch.revision or proposal.recovery not in branch.recoveries:
                decisions[step] = "incompatible_dependency"
                continue
            available = advertised.get(step, set())
            preferences = tuple(dict.fromkeys(ref for ref in proposal.preferred_procedures if ref in available))
            self.replace_branch(step, replace(
                branch, focus=proposal.focus, recovery=proposal.recovery,
                revision=revision + 1, request_id=request.request_id,
                inspect_focus=proposal.inspect_focus, preferred_procedures=preferences,
            ))
            decisions[step] = ("attached" if len(preferences) == len(proposal.preferred_procedures)
                               else "attached_with_filtered_procedures")
        return decisions


class WorkingMemory:
    """保留最近的双视角、动作关键帧和当前尝试的起点。"""

    def __init__(self, first: ObservationEvent, timesteps: int, capacity: int = 160) -> None:
        self.frames: deque[ObservationEvent] = deque((first,), maxlen=capacity)
        self.anchor = first
        self.timesteps = timesteps

    @property
    def latest(self) -> ObservationEvent:
        """返回最近一次实际机器人反馈。"""
        return self.frames[-1]

    def drawer_pull(self, since: int) -> int | None:
        """找出连续持握期间约 10 cm 的平面抽拉，实际打开哪层由视觉核验。"""
        origin = None
        previous = since - 1
        for frame in self.frames:
            if frame.sequence < since:
                continue
            if frame.sequence != previous + 1:
                origin = None
            previous = frame.sequence
            if not grasp_preserved((frame,), frame.sequence):
                origin = None
                continue
            position = frame.observation.robot_state["eef_position"]
            if origin is None:
                origin = position
            if abs(position[2] - origin[2]) <= 0.03 and dist(origin[:2], position[:2]) >= 0.09:
                return frame.sequence
        return None

    def sample(
        self, active: ActiveTask, *, inspect: bool = False, include_pour: bool = False,
        appearance_reference: bool = False,
    ) -> tuple[ObservationEvent, ...]:
        """选择起点、末帧及与持物、松爪或倾倒有关的关键帧。"""
        timesteps = max(2, self.timesteps - 1) if appearance_reference else self.timesteps
        pouring = include_pour or active.goal.predicate == Predicate.POURED
        horizon = 115 if inspect or pouring else 60
        recent = tuple(frame for frame in self.frames if frame.sequence >= max(
            active.start_step, self.latest.sequence - horizon,
        ))
        before_grasp = None
        if active.goal.predicate == Predicate.HELD and grasp_settled(self.frames, self.latest.sequence):
            begin = self.latest.sequence + 1
            for frame in reversed(recent):
                if frame.sequence != begin - 1 or frame.sequence <= active.start_step:
                    break
                if not grasp_preserved((frame,), frame.sequence):
                    before_grasp = frame
                    break
                begin = frame.sequence
            settled = tuple(
                frame for frame in recent
                if frame.sequence >= begin and grasp_settled(self.frames, frame.sequence)
            )
            if before_grasp is not None or len(settled) > 1 or not pouring:
                recent = settled
        if len(recent) + int(before_grasp is not None) <= timesteps:
            return recent if before_grasp is None else (before_grasp, *recent)
        picks = {recent[-1].sequence: recent[-1]}
        if before_grasp is not None:
            picks[before_grasp.sequence] = before_grasp
        if active.goal.predicate == Predicate.PLACED and (gripper_width(recent[-1].observation) or 0) >= OPEN_GRIP_M:
            for before, after in zip(reversed(recent[:-1]), reversed(recent[1:]), strict=True):
                width = gripper_width(before.observation)
                if width is None or before.sequence + 1 != after.sequence:
                    break
                if width < OPEN_GRIP_M:
                    for event in (before, after):
                        if len(picks) < timesteps:
                            picks[event.sequence] = event
                    break
        if len(picks) < timesteps:
            picks[recent[0].sequence] = recent[0]
        if (timesteps >= 3 and active.goal.predicate in {Predicate.OPENED, Predicate.CLOSED}
                and self.anchor.sequence < recent[0].sequence):
            picks[self.anchor.sequence] = self.anchor
        if pouring and len(recent) > 2 and len(picks) < timesteps:
            start = recent[0].observation.robot_state["eef_axis_angle"]
            orientation_peak = max(recent[1:-1], key=lambda frame: sum(
                (current - initial) ** 2 for current, initial in zip(
                    frame.observation.robot_state["eef_axis_angle"], start, strict=True,
                )
            ))
            picks[orientation_peak.sequence] = orientation_peak
        if before_grasp is not None and len(picks) < timesteps:
            lift_peak = max(recent, key=lambda frame: frame.observation.robot_state["eef_position"][2])
            picks[lift_peak.sequence] = lift_peak
        keyframes = detect_gripper_keyframes(None, recent, threshold=GRIPPER_KEYFRAME_DELTA_M)
        if keyframes and len(picks) < timesteps:
            sequence = keyframes[-1].sequence
            picks[sequence] = next(frame for frame in recent if frame.sequence == sequence)
        for fraction in ((0.70, 0.34, 0.5) if inspect else (0.5, 0.75, 0.25)):
            if len(picks) >= timesteps:
                break
            frame = recent[round((len(recent) - 1) * fraction)]
            picks[frame.sequence] = frame
        return tuple(picks[step] for step in sorted(picks))


@dataclass(frozen=True, slots=True)
class DrawerObservation:
    """某层抽屉在关闭之前有图像引用的内容观察。"""

    state: Literal["empty", "occupied"]
    step: int
    request_id: int
    refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttemptReport:
    """记录一次实际尝试的位移与夹爪反馈，供后续恢复区分重复失败。"""

    epoch: int
    mode: ExecutionMode
    start_step: int
    end_step: int
    outcome: str
    displacement_m: float
    gripper_before_m: float | None
    gripper_after_m: float | None


@dataclass(frozen=True, slots=True)
class Check:
    """Evidence Gate 的判定与只对该窗口有效的附加证据。"""

    outcome: Outcome
    reason: str
    fact: Fact | None = None
    prior_grasp: int | None = None
    hand_clear: bool = False
    retreat_clear: bool = False
    drawer_contents: DrawerObservation | None = None


class EvidenceGate:
    """核对窗口归属、视觉证据覆盖和事件时间。"""

    @staticmethod
    def check(
        window: EvidenceWindow,
        judgment: Judgment,
        required: tuple[Facet, ...],
    ) -> Check:
        """对一条观察应用已注册谓词的证据要求。"""
        goals = dict(window.goals)
        if judgment.step not in goals or window.frames[0].sequence < window.active.start_step:
            return Check(Outcome.UNKNOWN, "wrong_execution_interval")
        goal = goals[judgment.step]
        refs = window.frame_refs()
        support: dict[Facet, set[int]] = {}
        for item in judgment.support:
            if any(ref not in refs for ref in item.refs):
                return Check(Outcome.UNKNOWN, "unknown_evidence_reference")
            support.setdefault(item.facet, set()).update(refs[ref] for ref in item.refs)
        end = window.end
        current = {facet for facet, steps in support.items() if end in steps}
        retreat = Facet.RETREAT in current
        rechecking_placement = goal.predicate == Predicate.PLACED and judgment.step == window.recheck_step
        hand_clear = (goal.predicate == Predicate.PLACED and window.clearance == goal.destination and any(
            item.facet == Facet.CLEAR and f"front_{end}" in item.refs for item in judgment.support
        ))
        if judgment.outcome == Outcome.FALSE and Facet.CONTRADICTION in current:
            return Check(Outcome.FALSE, "visible_deviation", retreat_clear=retreat)
        if judgment.outcome != Outcome.TRUE or Facet.CONTRADICTION in support:
            clearance_only = (rechecking_placement and hand_clear and judgment.outcome == Outcome.UNKNOWN
                              and Facet.CONTRADICTION not in support)
            return Check(Outcome.UNKNOWN, "ambiguous_or_conflicting", hand_clear=clearance_only)
        if rechecking_placement:
            if not {Facet.IDENTITY, Facet.DESTINATION} <= current:
                return Check(Outcome.UNKNOWN, "current_placement_unverified", hand_clear=hand_clear)
            required = tuple(facet for facet in required if facet != Facet.RELEASE)
        if any(facet not in support for facet in required):
            return Check(Outcome.UNKNOWN, "incomplete_evidence")
        last = end
        if rechecking_placement:
            first = end
        elif goal.predicate == Predicate.PLACED:
            release = min((step for step in support[Facet.RELEASE]
                           if step > window.active.start_step), default=None)
            last = max(support[Facet.DESTINATION])
            if release is None or release > last:
                return Check(Outcome.UNKNOWN, "placement_and_release_not_aligned")
            if not any(
                release <= frame.sequence <= last
                and gripper_width(frame.observation) is not None
                and not grasp_preserved((frame,), frame.sequence)
                for frame in window.frames
            ):
                return Check(Outcome.UNKNOWN, "release_conflicts_with_gripper_state")
            first = last
        else:
            first = min(support[Facet.EFFECT])
        if goal.predicate == Predicate.POURED:
            last = min((upright for upright in support[Facet.UPRIGHT] if upright > first), default=-1)
            if last < 0:
                return Check(Outcome.UNKNOWN, "pour_cycle_not_observed")
            if first <= window.active.start_step:
                return Check(Outcome.UNKNOWN, "pour_cycle_outside_attempt")
        elif goal.predicate == Predicate.HELD:
            first = max(support[Facet.EFFECT])
            if first <= window.active.start_step:
                return Check(Outcome.UNKNOWN, "grasp_outside_attempt")
            if not grasp_preserved(window.frames, first):
                return Check(Outcome.UNKNOWN, "gripper_state_conflict")
        elif goal.predicate != Predicate.PLACED and Facet.EFFECT not in current:
            return Check(Outcome.UNKNOWN, "effect_missing_at_window_end")
        if goal.predicate == Predicate.OPENED and (
            Facet.APERTURE not in current or not any(
                item.facet == Facet.EFFECT and f"front_{end}" in item.refs
                for item in judgment.support
            )
        ):
            return Check(Outcome.UNKNOWN, "door_clear_opening_unverified")
        contents = None
        if goal.predicate == Predicate.OPENED and window.inspect_drawer == goal.target:
            visible = support[Facet.EFFECT] & support[Facet.APERTURE]
            empty = visible & support.get(Facet.EMPTY, set())
            occupied = visible & support.get(Facet.OCCUPIED, set())
            if bool(empty) == bool(occupied):
                return Check(Outcome.UNKNOWN, "drawer_contents_unverified")
            facet = Facet.EMPTY if empty else Facet.OCCUPIED
            observed = max(empty or occupied)
            if observed <= window.active.start_step:
                return Check(Outcome.UNKNOWN, "drawer_contents_unverified")
            contents = DrawerObservation(
                "empty" if empty else "occupied", observed, window.request_id,
                tuple(dict.fromkeys(ref for item in judgment.support if item.facet in {Facet.APERTURE, facet}
                                    for ref in item.refs if refs[ref] == observed)),
            )
        prior_grasp = max((
            frame.sequence for frame in window.frames if frame.sequence < end
            and frame.sequence in support.get(Facet.PRIOR_GRASP, ())
            and grasp_preserved((frame,), frame.sequence)
        ), default=None)
        fact = Fact(
            judgment.step, goal, window.active.epoch, first, last, window.request_id,
            tuple(dict.fromkeys(ref for item in judgment.support for ref in item.refs)),
        )
        return Check(Outcome.TRUE, "verified", fact, prior_grasp, hand_clear and last == end, retreat, contents)


class EpisodicMemory:
    """按步骤登记唯一事件，独立维护可被后续观察更新的关系。"""

    def __init__(self) -> None:
        self.facts: dict[int, Fact] = {}
        self.latest_checks: dict[int, tuple[int, Check]] = {}
        self.drawer_contents: dict[str, DrawerObservation] = {}
        self.drawer_pulls: dict[int, int] = {}
        self.attempts: dict[int, deque[AttemptReport]] = {}
        self.last_grasp_break: dict[str, int] = {}
        self.grasp_references: dict[int, ObservationEvent] = {}
        self.corrections: list[dict[str, object]] = []
        self.version = 0

    @property
    def progress(self) -> int:
        """从有效账本重新计算有序完成前缀。"""
        step = 0
        while step in self.facts:
            step += 1
        return step

    def pickup_for(self, step: int, target: str) -> Fact | None:
        """从当前步骤之前的有效事件查找同一物体的最近抓取。"""
        for index in range(step - 1, -1, -1):
            fact = self.facts.get(index)
            if fact is not None and fact.goal.predicate == Predicate.HELD and fact.goal.target == target:
                return fact
        return None

    def record_attempt(self, step: int, report: AttemptReport) -> None:
        """只保留同一步的四个最近尝试。"""
        self.attempts.setdefault(step, deque(maxlen=4)).append(report)

    def reopen_grasp(self, step: int, observed_at: int) -> None:
        """失去持物关系后重新打开抓取前提，历史核验留在日志中。"""
        if self.facts[step].goal.predicate != Predicate.HELD:
            raise ValueError("Only a lost pickup prerequisite can be reopened")
        self.reopen_effect(step, observed_at, "grasp_lost")

    def reopen_effect(self, step: int, observed_at: int, reason: str) -> None:
        """撤回已变化的状态，保留之前已经发生的倾倒事件。"""
        fact = self.facts[step]
        if fact.goal.predicate == Predicate.POURED:
            raise ValueError("A completed pour is an event and cannot be reopened")
        del self.facts[step]
        self.latest_checks.pop(step, None)
        self.drawer_contents.pop(fact.goal.target, None)
        self.corrections.append({
            "step": step, "supersedes_request_id": fact.request_id,
            "reason": reason, "observed_at_step": observed_at,
        })
        self.version += 1

    def record(self, step: int, end: int, check: Check) -> bool:
        """接纳事件，或在同一物理时刻证据冲突时撤回该事件。"""
        old = self.latest_checks.get(step)
        prior = self.facts.get(step)
        clearance_only = (check.hand_clear and prior is not None and prior.goal.predicate == Predicate.PLACED
                          and not (old is not None and old[0] == end and old[1].outcome == Outcome.FALSE))
        if (check.outcome != Outcome.UNKNOWN or clearance_only) and (old is None or end >= old[0]):
            self.latest_checks[step] = (end, check)
            self.version += 1
        if check.outcome == Outcome.FALSE and prior is not None and prior.last == end:
            self.corrections.append({"step": step, "supersedes_request_id": prior.request_id})
            del self.facts[step]
            self.drawer_contents.pop(prior.goal.target, None)
            return False
        fact = check.fact
        if fact is None:
            return False
        if fact.goal.predicate == Predicate.OPENED and check.drawer_contents is not None:
            old_reading = self.drawer_contents.get(fact.goal.target)
            if old_reading is None or old_reading.step <= check.drawer_contents.step:
                self.drawer_contents[fact.goal.target] = check.drawer_contents
        if prior is not None:
            return False
        if fact.goal.predicate == Predicate.POURED and any(
            event.goal.predicate == Predicate.POURED and event.goal.target == fact.goal.target
            and event.goal.destination == fact.goal.destination and event.last >= fact.first
            for event in self.facts.values()
        ):
            return False
        self.facts[step] = fact
        return True
