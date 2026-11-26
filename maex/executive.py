"""在同一意图树上协调异步验证、前瞻和物理控制。"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, replace
from math import dist, isfinite
from time import monotonic
from uuid import uuid4

from pydantic import ValidationError

from roboagent.observer.types import ObservationEvent
from roboagent.types import EpisodeResult, EpisodeStatus

from .executor import PI05, Executor
from .memory import (
    CLOSED_GRIP_M,
    GRIPPER_KEYFRAME_DELTA_M,
    OPEN_GRIP_M,
    AttemptReport,
    Check,
    EpisodicMemory,
    EvidenceGate,
    IntentionTree,
    ProcedureLibrary,
    WorkingMemory,
    grasp_preserved,
    grasp_settled,
    gripper_width,
)
from .models import Observer, Planner
from .schema import (
    ActiveTask,
    Assessment,
    EvidenceWindow,
    ExecutionMode,
    Facet,
    Fact,
    Judgment,
    Outcome,
    PlanRequest,
    Predicate,
    RecoveryDecision,
)


@dataclass(frozen=True, slots=True)
class Limits:
    """每个 episode 和每次物理尝试使用固定预算。"""

    max_steps: int = 2500
    wall_seconds: float = 900
    planner_calls: int = 20
    observer_calls: int = 100
    verify_stride: int = 45
    unverified_steps: int = 30
    attempt_chunks: int = 20
    timesteps: int = 5
    completion_wait_steps: int = 200


class IntentGate:
    """核查已绑定候选的当前进入条件和安全边界。"""

    @staticmethod
    def blocked(
        candidate: ActiveTask, executor: Executor, memory: EpisodicMemory, tree: IntentionTree,
        working: WorkingMemory,
    ) -> str | None:
        """返回未知或不满足的进入条件。"""
        if candidate.executor != PI05 or (candidate.mode != "return_to_home" and not executor.inside_workspace()):
            return "executor_or_workspace"
        if candidate.goal != tree.goal(candidate.step, memory):
            return "changed_binding"
        now = executor.latest.sequence
        if candidate.mode == "retreat":
            end, check = memory.latest_checks.get(candidate.step, (-1, Check(Outcome.UNKNOWN, "")))
            return None if now == end and check.retreat_clear else "retreat_clearance_unknown"
        if candidate.mode in {"return_to_home", "hold_position"}:
            active_step = executor.active.step if executor.active is not None else memory.progress
            final_hold = (candidate.mode == "hold_position"
                          and candidate.step + 1 == memory.progress == len(tree.plan.goals))
            if candidate.step != active_step and not final_hold:
                return "recovery_requires_current_intent"
            command = executor.latest.observation.robot_state.get("gripper_command")
            return None if isinstance(command, (float, int)) and isfinite(command) else "gripper_command_unknown"
        if candidate.mode != "policy":
            if (executor.active is None or candidate.step != executor.active.step
                    or (candidate.step != memory.progress and candidate.mode != "restage_vla")):
                return "recovery_requires_current_intent"
            if candidate.mode in {"open_gripper", "close_gripper"}:
                closing_on_handle = candidate.mode == "close_gripper" and candidate.goal.is_drawer_motion
                if candidate.goal.predicate != Predicate.HELD and not closing_on_handle:
                    return "gripper_recovery_requires_pickup"
                width = gripper_width(executor.latest.observation)
                if width is None:
                    return "gripper_width_unknown"
                if candidate.mode == "close_gripper" and width <= CLOSED_GRIP_M:
                    return "gripper_already_closed"
                if candidate.mode == "open_gripper" and width >= OPEN_GRIP_M:
                    return "gripper_already_open"
                if closing_on_handle and width < OPEN_GRIP_M:
                    return "drawer_handle_recovery_requires_open_gripper"
            if candidate.mode == "release_gripper" and (
                candidate.goal.predicate != Predicate.PLACED or not grasp_preserved((executor.latest,), now)
            ):
                return "release_requires_held_placement"
            if candidate.mode == "pull_drawer" and (
                not candidate.goal.is_drawer_motion or candidate.goal.predicate != Predicate.OPENED
                or not grasp_settled(working.frames, now)
            ):
                return "drawer_pull_requires_current_handle_grasp"
            return None
        if candidate.step != memory.progress:
            return "ordered_prefix"
        if candidate.step == 0:
            return None
        last = memory.facts[candidate.step - 1]
        inspection = tree.inspecting_drawer(last.step)
        if inspection is not None and inspection not in memory.drawer_contents:
            return "drawer_contents_unknown"
        if (candidate.goal.predicate in {Predicate.PLACED, Predicate.POURED}
                and last.goal.predicate in {Predicate.HELD, Predicate.POURED}
                and last.goal.target == candidate.goal.target):
            proof = memory.latest_checks[last.step][1].fact
            if last.goal.predicate == Predicate.POURED:
                pickup = memory.pickup_for(candidate.step, candidate.goal.target)
                if pickup is not None:
                    refreshed = memory.latest_checks[pickup.step][1].fact
                    if refreshed is not None and refreshed.first > last.last:
                        proof = refreshed
            if proof is None:
                return "grasp_not_current"
            since = proof.first if proof.goal.predicate == Predicate.HELD else proof.last
            if since <= memory.last_grasp_break.get(candidate.goal.target, -1) or not grasp_preserved(working.frames, since):
                return "grasp_not_current"
        if last.goal.predicate == Predicate.PLACED and candidate.goal.closes(last.goal.destination):
            end, check = memory.latest_checks[last.step]
            if end != now or check.outcome == Outcome.FALSE or not check.hand_clear:
                return "placement_or_door_clearance_unknown"
        return None


class Executive:
    """唯一接纳事实、修订候选和提交控制权的协调者。"""

    def __init__(
        self, planner: Planner, observer: Observer, executor: Executor, library: ProcedureLibrary,
        limits: Limits,
    ) -> None:
        if library.executor != PI05:
            raise ValueError(f"Procedures require an incompatible executor: {library.executor}")
        self.planner = planner
        self.observer = observer
        self.executor = executor
        self.library = library
        self.limits = limits
        self.episode_id = uuid4().hex
        self.memory = EpisodicMemory()
        self.working = WorkingMemory(executor.latest, limits.timesteps)
        self.journal: list[dict[str, object]] = []
        self.jobs: dict[str, asyncio.Task] = {}
        self.wakeup = asyncio.Event()
        self.window: EvidenceWindow | None = None
        self.expansion: PlanRequest | None = None
        self.recovery_snapshot: tuple[ActiveTask, int, str, str] | None = None
        self.recent_judgments: tuple[Judgment, ...] = ()
        self.started = monotonic()
        self.deadline = self.started + limits.wall_seconds
        self.planner_count = 0
        self.observer_count = 0
        self.vla_count = 0
        self.completed_chunks = 0
        self.goal_anchor = executor.latest
        self.attempts: dict[int, int] = {}
        self.expanded: set[int] = set()
        self.retreated: set[int] = set()
        self.boundary_recoveries: set[tuple[int, int, ExecutionMode]] = set()
        self.last_recovery_at: tuple[int, int, int] | None = None
        self.last_verified_step = -1
        self.verification_stride_anchor = -1
        self.same_boundary_queries = 0
        self.queried_windows: set[tuple[object, ...]] = set()
        self.last_pickup_review: tuple[int, int] | None = None
        self.completion_wait_started: int | None = None
        self.result: tuple[EpisodeStatus, str] | None = None
        self.tree: IntentionTree | None = None

    async def run(self, task: str) -> EpisodeResult:
        """运行一条完整 episode，终止时回收在途的语义与动作请求。"""
        try:
            async with asyncio.timeout_at(self.deadline):
                await self.initialize_tree(task)
                while self.result is None:
                    self.wakeup.clear()
                    self.receive()
                    if self.result is not None:
                        break
                    if monotonic() >= self.deadline:
                        self.stop("timeout", "Episode wall-clock budget reached")
                        break
                    self.decide(task)
                    if self.result is None:
                        await self.wakeup.wait()
        except TimeoutError:
            reason = ("Episode wall-clock budget reached" if monotonic() >= self.deadline
                      else "Planning or control request timed out")
            self.stop("timeout", reason)
        except Exception as error:
            self.stop("error", f"{type(error).__name__}: {error}")
        finally:
            self.executor.cancel_event.set()
            for name, job in self.jobs.items():
                if not job.done():
                    self.log("request_cancelled", worker=name)
                    job.cancel()
            await asyncio.gather(*self.jobs.values(), return_exceptions=True)
        assert self.result is not None
        status, summary = self.result
        return EpisodeResult(status, summary, self.planner_count + self.observer_count, self.vla_count, {
            "agent": "maex",
            "episode_id": self.episode_id,
            "step": self.working.latest.sequence,
            "procedure_version": self.library.version,
            "confirmed_prefix": self.memory.progress,
            "facts": [asdict(fact) for fact in self.memory.facts.values()],
            "corrections": self.memory.corrections,
            "drawer_contents": {name: asdict(reading) for name, reading in self.memory.drawer_contents.items()},
            "drawer_pulls": self.memory.drawer_pulls,
            "last_grasp_break": self.memory.last_grasp_break,
            "branches": [asdict(branch) for branch in self.tree.branches] if self.tree else [],
            "model_calls": {"planner": self.planner_count, "observer": self.observer_count,
                            "vla": self.vla_count},
            "journal": self.journal,
        })

    async def initialize_tree(self, task: str) -> None:
        """用 Planner 与墙钟预算继续修正无效的初始计划。"""
        feedback = ""
        for attempt in range(self.limits.planner_calls):
            self.planner_count += 1
            self.log("global_plan_requested", attempt=attempt + 1)
            try:
                plan, usage = await self.planner.plan(task, self.working.latest.observation, feedback)
            except ValidationError as error:
                feedback = "\n".join(
                    f"{item['loc']}: {item['msg']}" for item in error.errors(include_input=False)
                )
                self.log("global_plan_invalid", reason=feedback)
                continue
            except (RuntimeError, TimeoutError) as error:
                self.log("global_plan_failed", error=type(error).__name__, reason=str(error))
                await asyncio.sleep(0.25)
                continue
            self.tree = IntentionTree(self.episode_id, plan, self.library)
            self.log("global_plan_prepared", plan=plan, usage=usage)
            return
        self.stop("failure", f"Planner budget reached without a valid initial plan. {feedback}".rstrip())

    def receive(self) -> None:
        """优先读取动作反馈，再核验证据，最后接续候选补丁。"""
        motion = self.jobs.get("motion")
        if motion is not None and motion.done():
            del self.jobs["motion"]
            chunk = motion.result()
            self.completed_chunks += 1
            self.log("chunk_finished", **asdict(chunk))
            if chunk.start_step < self.last_verified_step <= chunk.end_step:
                self.verification_stride_anchor = chunk.end_step
            goal = chunk.task.goal
            if goal.predicate in {Predicate.PLACED, Predicate.POURED}:
                gap = next((
                    frame.sequence for frame in reversed(self.working.frames)
                    if frame.sequence > chunk.start_step and not grasp_preserved((frame,), frame.sequence)
                ), None)
                if gap is not None:
                    self.memory.last_grasp_break[goal.target] = gap
            self.record_drawer_pull(chunk.task)
            if chunk.task.mode != "policy":
                self.record_attempt(chunk.task, "executed" if chunk.status == "running" else chunk.status)
            if chunk.status == "safety_stop" and chunk.task.mode == "return_to_home":
                self.stop("safety_stop", "Guarded return-to-home could not restore the workspace")
                return
            if chunk.status in {"success", "failure", "timeout", "cancelled"}:
                self.stop(chunk.status, f"Physical execution ended with {chunk.status}")
                return
        observation = self.jobs.get("verify")
        if observation is not None and observation.done():
            del self.jobs["verify"]
            assert self.window is not None
            try:
                assessment, usage = observation.result()
            except Exception as error:
                self.log("verification_failed", id=self.window.request_id, error=type(error).__name__)
            else:
                self.admit(self.window, assessment, usage)
            self.window = None
        planning = self.jobs.get("expand")
        if planning is not None and planning.done():
            del self.jobs["expand"]
            assert self.expansion is not None and self.tree is not None
            try:
                patch, usage = planning.result()
            except Exception as error:
                self.log("expansion_failed", id=self.expansion.request_id, error=type(error).__name__)
            else:
                decisions = self.tree.reconcile(self.expansion, patch, self.memory)
                self.log("patch_returned", id=self.expansion.request_id, decisions=decisions,
                         patch=patch, usage=usage, version=self.memory.version)
            self.expansion = None
        recovery = self.jobs.get("recover")
        if recovery is not None and recovery.done() and self.result is None:
            del self.jobs["recover"]
            assert self.recovery_snapshot is not None
            active, step, task, feedback = self.recovery_snapshot
            self.recovery_snapshot = None
            try:
                decision, usage = recovery.result()
            except Exception as error:
                validation = ""
                if isinstance(error, ValidationError):
                    validation = ", ".join(
                        f"{item['loc']}: {item['type']}" for item in error.errors(include_input=False)
                    )
                self.log("recovery_planning_failed", error=type(error).__name__, validation=validation or None)
                if validation and not feedback and self.planner_count < self.limits.planner_calls:
                    self.review_execution(task, active, validation_feedback=validation)
                    return
                decision, usage = RecoveryDecision(action="retry", refs=(), reason="Recovery request failed"), {}
            self.apply_recovery(active, step, decision, usage)

    def admit(self, window: EvidenceWindow, assessment: Assessment, usage: dict[str, int]) -> None:
        """只把经过谓词核验的事件写入账本。"""
        assert self.tree is not None
        active = self.executor.active
        if active != window.active or window.episode_id != self.episode_id:
            self.log("verification_discarded", id=window.request_id, reason="inactive_execution")
            return
        judgments = {item.step: item for item in assessment.judgments}
        self.log("verification_returned", id=window.request_id, judgments=assessment, usage=usage)
        grasp = None
        for step, goal in window.goals:
            judgment = judgments.get(step)
            if judgment is None:
                continue
            if window.at_boundary and goal.predicate == Predicate.POURED and step in self.memory.facts:
                continue
            check = EvidenceGate.check(window, judgment, self.tree.branches[step].evidence)
            if step == window.pickup_review:
                self.apply_pickup_review(window, active, judgment, check)
                return
            if check.fact is not None and goal.is_drawer_motion and goal.predicate == Predicate.OPENED:
                pull = self.memory.drawer_pulls.get(step)
                if pull is None or pull > check.fact.last:
                    check = Check(Outcome.UNKNOWN, "drawer_pull_unverified")
            if step == active.step:
                self.recent_judgments = (*self.recent_judgments[-1:], judgment)
                if active.goal.predicate == Predicate.HELD and check.fact is not None:
                    if grasp_settled(self.working.frames, check.fact.first):
                        grasp = check.fact
                        if not grasp_preserved(self.working.frames, grasp.first):
                            check = Check(Outcome.UNKNOWN, "grasp_changed_after_capture")
                    else:
                        check = Check(Outcome.UNKNOWN, "grasp_not_settled")
            if step != active.step and step != window.recheck_step:
                if goal.target != active.goal.target or check.fact is None:
                    continue
                if active.goal.predicate == Predicate.HELD and active.step not in self.memory.facts:
                    if grasp is not None and grasp.first < check.fact.first:
                        fact = replace(grasp, last=grasp.first)
                    elif check.prior_grasp is not None and active.start_step < check.prior_grasp < check.fact.first:
                        fact = Fact(active.step, active.goal, active.epoch, check.prior_grasp,
                                    check.prior_grasp, window.request_id, check.fact.refs)
                    else:
                        continue
                    self.record_fact(active.step, fact.last, Check(Outcome.TRUE, "prior_grasp", fact), window)
                if active.step not in self.memory.facts:
                    continue
            previous = self.memory.latest_checks.get(step)
            if (window.recheck_step == step and check.outcome == Outcome.FALSE
                    and window.end == self.working.latest.sequence and step in self.memory.facts
                    and previous is not None and previous[0] == window.end and previous[1].outcome == Outcome.FALSE):
                self.memory.reopen_effect(step, window.end, "effect_contradicted")
                self.log("effect_reopened", goal_step=step, request_id=window.request_id)
            self.record_fact(step, window.end, check, window)

    def apply_pickup_review(
        self, window: EvidenceWindow, active: ActiveTask, judgment: Judgment, check: Check,
    ) -> None:
        """在静止边界复核抓取前提，保留已经确认的后续事件。"""
        step = judgment.step
        if window.end != self.working.latest.sequence or "motion" in self.jobs or step not in self.memory.facts:
            return
        fact = check.fact
        if (fact is not None and grasp_settled(self.working.frames, fact.first)
                and grasp_preserved(self.working.frames, fact.first)):
            self.record_fact(step, window.end, check, window)
            self.log("pickup_refreshed", goal_step=step, interrupted_step=active.step, fact=asdict(fact))
            assert self.tree is not None
            if self.memory.progress < len(self.tree.plan.goals) and self.transition(self.memory.progress):
                self.dispatch()
        elif (check.outcome == Outcome.FALSE and any(
            item.facet == Facet.CONTRADICTION and f"wrist_{window.end}" in item.refs
            for item in judgment.support
        )):
            self.memory.reopen_grasp(step, window.end)
            self.log("pickup_reopened", goal_step=step, interrupted_step=active.step, reason="grasp_contradicted",
                     request_id=window.request_id)

    def record_fact(self, step: int, end: int, check: Check, window: EvidenceWindow) -> None:
        """记录证据接纳、冲突修正或仍需验证的判断。"""
        admitted = self.memory.record(step, end, check)
        if admitted and check.fact is not None and check.fact.goal.predicate == Predicate.HELD:
            self.memory.grasp_references[step] = next(
                frame for frame in window.frames if frame.sequence == check.fact.first
            )
        self.log("evidence_gate", goal_step=step, observed_at_step=end, outcome=check.outcome,
                 reason=check.reason, admitted=admitted, hand_clear=check.hand_clear,
                 drawer_contents=asdict(check.drawer_contents) if check.drawer_contents else None,
                 fact=asdict(check.fact) if admitted and check.fact else None)

    def record_drawer_pull(self, active: ActiveTask) -> None:
        """在发布语义窗口前登记已经采集到的抽拉反馈。"""
        goal = active.goal
        if not goal.is_drawer_motion or goal.predicate != Predicate.OPENED or active.step in self.memory.drawer_pulls:
            return
        pull = self.working.drawer_pull(self.goal_anchor.sequence)
        if pull is not None:
            self.memory.drawer_pulls[active.step] = pull
            self.log("drawer_pull_observed", goal_step=active.step, observed_at_step=pull)

    def decide(self, task: str) -> None:
        """以本轮已到达的信息选择保持、继续、恢复或提交。"""
        assert self.tree is not None
        active = self.executor.active
        progress = self.memory.progress
        moving = "motion" in self.jobs
        step = self.working.latest.sequence
        if not moving and step >= self.limits.max_steps:
            self.stop("failure", "Action budget reached")
            return
        if "recover" in self.jobs:
            return
        if not moving and not self.executor.inside_workspace():
            if self.transition(active.step if active is not None else progress, mode="return_to_home"):
                self.log("workspace_recovery_requested")
                self.dispatch()
            else:
                self.stop("safety_stop", "Guarded return-to-home could not be started")
            return
        if active is None:
            if not self.transition(progress):
                self.stop("error", "The validated initial intent could not be committed")
            else:
                self.dispatch_after_commit(task)
            return
        complete = progress == len(self.tree.plan.goals)
        if not complete:
            self.completion_wait_started = None
        elif self.wait_for_completion(active, progress, moving):
            return
        if active.mode != "policy":
            if moving or "verify" in self.jobs:
                return
            if self.completed_chunks == 0:
                self.dispatch()
                return
            if (self.last_verified_step != step or self.same_boundary_queries == 0) and self.verify(inspect=True):
                return
        if progress > active.step:
            if not complete and self.transition(progress):
                self.dispatch_after_commit(task)
            elif not moving and "verify" not in self.jobs and not self.verify(inspect=True):
                if not complete:
                    self.reopen_pickup_at_handoff(active, progress)
                self.review_execution(task, active)
            return
        if progress < active.step or active.mode != "policy":
            if moving or "verify" in self.jobs:
                return
            if (active.mode == "close_gripper" and active.goal.is_drawer_motion
                    and active.goal.predicate == Predicate.OPENED and grasp_settled(self.working.frames, step)):
                self.review_execution(task, active)
                return
            if self.transition(progress):
                self.dispatch_after_commit(task)
            else:
                self.review_execution(task, active)
            return
        self.expand(task)
        if moving:
            if self.verification_due(active):
                self.verify()
            return
        at_limit = self.completed_chunks >= self.limits.attempt_chunks
        end, check = self.memory.latest_checks.get(active.step, (-1, Check(Outcome.UNKNOWN, "")))
        deviation = check.outcome == Outcome.FALSE and end == step and self.completed_chunks >= 8
        width = gripper_width(self.working.latest.observation)
        empty_after_grasp = (width is not None and width <= CLOSED_GRIP_M
                             and self.lost_pickup(active) is not None)
        if at_limit or deviation or empty_after_grasp:
            if "verify" in self.jobs:
                return
            if (at_limit or empty_after_grasp) and self.last_verified_step != step and self.verify(inspect=True):
                return
            self.review_execution(task, active)
            return
        if self.verification_due(active) and "verify" not in self.jobs and not self.verify():
            self.review_execution(task, active)
            return
        # 经过未验证动作上限时保持，直到当前窗口返回。
        holding_candidate = active.goal.predicate == Predicate.HELD and grasp_settled(self.working.frames, step)
        remaining_overlap = 0 if holding_candidate or active.goal.predicate == Predicate.POURED else self.limits.unverified_steps
        if self.window is None:
            self.dispatch()
        else:
            boundary = max(self.window.end, self.verification_stride_anchor)
            remaining = boundary + remaining_overlap - step
            if remaining > 0:
                self.dispatch(steps=remaining)
            else:
                self.log("hold", reason="waiting_for_observer")

    def wait_for_completion(self, active: ActiveTask, progress: int, moving: bool) -> bool:
        """保持已完成任务的现场，让官方终态检查使用同一物理步预算。"""
        if moving or "verify" in self.jobs:
            return True
        success = self.working.latest.observation.task_success
        if success is None or success:
            self.stop("success", "All intention effects verified" if success is None else "Benchmark confirmed completion")
            return True
        step = self.working.latest.sequence
        if self.completion_wait_started is None:
            self.completion_wait_started = step
            self.log("terminal_verification_requested", window_steps=self.limits.completion_wait_steps)
        if step - self.completion_wait_started >= self.limits.completion_wait_steps:
            return False
        already_holding = active.mode == "hold_position" and active.step == progress - 1
        if already_holding or self.transition(progress - 1, mode="hold_position"):
            self.dispatch(steps=self.completion_wait_started + self.limits.completion_wait_steps - step)
            return True
        return False

    def reopen_pickup_at_handoff(self, active: ActiveTask, successor_step: int) -> bool:
        """转步补证后，撤回动作块执行期间已经丢失的抓取。"""
        assert self.tree is not None
        if active.goal.predicate != Predicate.HELD:
            return False
        successor = self.tree.goal(successor_step, self.memory)
        if (successor is None or successor.target != active.goal.target
                or successor.predicate not in {Predicate.PLACED, Predicate.POURED}):
            return False
        current = self.working.latest
        if (gripper_width(current.observation) is None
                or grasp_preserved((current,), current.sequence)):
            return False
        self.memory.reopen_grasp(active.step, current.sequence)
        self.log("pickup_reopened", goal_step=active.step, interrupted_step=successor_step,
                 reason="grasp_lost_before_handoff")
        return True

    def review_execution(self, task: str, active: ActiveTask, *, validation_feedback: str = "") -> None:
        """在动作边界复查未决或停滞，沿用整个 episode 的剩余预算。"""
        assert self.tree is not None
        branch = self.tree.branches[active.step]
        current = self.working.latest
        end, check = self.memory.latest_checks.get(active.step, (-1, Check(Outcome.UNKNOWN, "")))
        if ((branch.recovery == "retreat" or active.step < self.memory.progress)
                and end == current.sequence and check.retreat_clear
                and active.step not in self.retreated and self.transition(active.step, mode="retreat")):
            self.dispatch()
            return
        progress = self.memory.progress
        pending = self.candidate_for(progress) if progress < len(self.tree.plan.goals) else None
        blocker = (IntentGate.blocked(pending, self.executor, self.memory, self.tree, self.working)
                   if pending is not None else None)
        if (blocker == "grasp_not_current" and pending is not None and not validation_feedback
                and grasp_settled(self.working.frames, current.sequence)):
            pickup = self.memory.pickup_for(progress, pending.goal.target)
            if pickup is not None and self.verify(inspect=True, pickup=pickup):
                return
        boundary = active.step, current.sequence, self.memory.progress
        if self.planner_count >= self.limits.planner_calls or (self.last_recovery_at == boundary and not validation_feedback):
            self.resume_at_boundary(active)
            return
        self.last_recovery_at = boundary
        self.recovery_snapshot = active, current.sequence, task, validation_feedback
        self.planner_count += 1
        lost_pickup = self.lost_pickup(active)
        recovery_subject = active
        if pending is not None and active.step < progress and (lost_pickup is not None or blocker == "grasp_not_current"):
            recovery_subject = replace(active, goal=pending.goal)
        if active.mode == "policy" and not validation_feedback:
            self.record_attempt(active, "goal_unverified")
        attempts = [asdict(report) for report in self.memory.attempts.get(active.step, ())]
        options = self.recovery_options(active, current) if lost_pickup is None else None
        self.log("recovery_requested", goal_step=active.step, epoch=active.epoch,
                 attempt=active.attempt, robot_state=current.observation.robot_state,
                 lost_pickup=lost_pickup, blocked_by=blocker, available_actions=options)
        focus = branch.focus
        if active.step < self.memory.progress:
            focus = (
                f"The current effect was verified. Seek a clearer view with safe arm repositioning "
                f"while preserving this effect and the gripper command. {focus}"
            )
        if blocker == "grasp_not_current" and pending is not None and lost_pickup is None:
            focus = (
                f"Fresh visual evidence of the held {pending.goal.target} is required to resume policy. "
                "A grip interruption invalidated the previous authorization. Retrying the same policy is blocked. "
                "If both current views show a clear wrist retraction, choose restage_vla to reveal the held "
                f"object while preserving the gripper command. {focus}"
            )
        inspection = self.tree.inspecting_drawer(active.step)
        if inspection is not None:
            focus = (
                f"Inspect the inside of the {inspection} before closing it. If it is already open but the arm "
                f"blocks its interior, look for a clear retraction path to expose it. {focus}"
            )
        if active.goal.is_drawer_motion and active.goal.predicate == Predicate.OPENED:
            pull = self.memory.drawer_pulls.get(active.step)
            focus = f"Measured planar pull: {'observed' if pull is not None else 'not yet observed'}. {focus}"
        if validation_feedback:
            focus = (
                f"Retry the same observation using exactly the supplied tool schema. "
                f"Correct the invalid fields: {validation_feedback}. {focus}"
            )
        if options is not None:
            focus = f"Available actions at the current boundary: {', '.join(options)}. {focus}"
        self.start_job("recover", self.planner.recover(
            task, recovery_subject, current.sequence, current.observation,
            self.recent_judgments, focus, lost_pickup, attempts,
            before=self.working.anchor.observation if active.mode == "policy" and active.attempt > 1 else None,
        ))

    @staticmethod
    def required_recovery_refs(active: ActiveTask, mode: str, step: int) -> set[str]:
        """给出当前恢复动作必须引用的相机帧。"""
        if mode in {"release_gripper", "pull_drawer", "restage_vla"} or (
            mode == "close_gripper" and active.goal.is_drawer_motion
        ):
            return {f"front_{step}", f"wrist_{step}"}
        return {f"wrist_{step}"} if mode == "close_gripper" else set()

    def recovery_options(self, active: ActiveTask, current: ObservationEvent) -> tuple[str, ...]:
        """预检查当前状态与相机覆盖，给 Planner 提供可申请的恢复操作。"""
        assert self.tree is not None
        available = {f"{view}_{current.sequence}" for view, image in (
            ("front", current.observation.rgb), ("wrist", current.observation.wrist_rgb),
        ) if image is not None}
        options = ["retry"]
        if not available:
            return tuple(options)
        modes: tuple[ExecutionMode, ...] = (
            "open_gripper", "close_gripper", "release_gripper", "pull_drawer", "restage_vla", "return_to_home",
        )
        for mode in modes:
            if not self.required_recovery_refs(active, mode, current.sequence) <= available:
                continue
            if mode == "return_to_home" and (active.step, active.attempt, mode) in self.boundary_recoveries:
                continue
            candidate = self.candidate_for(active.step, mode)
            if candidate is not None and IntentGate.blocked(candidate, self.executor, self.memory, self.tree, self.working) is None:
                options.append(mode)
        return tuple(options)

    def record_attempt(self, active: ActiveTask, outcome: str) -> None:
        """用实际起止反馈记录一次恢复或未核验的策略尝试。"""
        start, end = self.working.anchor, self.working.latest
        report = AttemptReport(
            active.epoch, active.mode, start.sequence, end.sequence, outcome,
            round(dist(start.observation.robot_state["eef_position"], end.observation.robot_state["eef_position"]), 3),
            gripper_width(start.observation), gripper_width(end.observation),
        )
        self.memory.record_attempt(active.step, report)
        self.log("attempt_reviewed", goal_step=active.step, **asdict(report))

    def lost_pickup(self, active: ActiveTask) -> int | None:
        """按待执行目标查找丢失的抓取，覆盖连续倾倒间的交接。"""
        assert self.tree is not None
        progress = self.memory.progress
        if not active.step <= progress < len(self.tree.plan.goals):
            return None
        goal = self.tree.goal(progress, self.memory)
        if goal is None or goal.target != active.goal.target or goal.predicate not in {Predicate.PLACED, Predicate.POURED}:
            return None
        current = self.working.latest
        if grasp_preserved((current,), current.sequence):
            return None
        pickup = self.memory.pickup_for(progress, goal.target)
        return pickup.step if pickup is not None else None

    def apply_recovery(
        self, active: ActiveTask, step: int, decision: RecoveryDecision, usage: dict[str, int],
    ) -> None:
        """恢复只触发当前意图的动作，效果仍需 Evidence Gate 核验。"""
        if self.executor.active != active or self.working.latest.sequence != step:
            self.log("recovery_discarded", epoch=active.epoch, reason="stale_snapshot")
            return
        observation = self.working.latest.observation
        available = {f"{view}_{step}" for view, frame in (
            ("front", observation.rgb), ("wrist", observation.wrist_rgb),
        ) if frame is not None}
        self.log("recovery_prepared", epoch=active.epoch, decision=decision, usage=usage)
        fresh = bool(decision.refs) and set(decision.refs) <= available
        if decision.action == "regrasp":
            prior = self.lost_pickup(active)
            if fresh and prior is not None:
                self.memory.reopen_grasp(prior, step)
                self.log("pickup_reopened", goal_step=prior, interrupted_step=active.step, refs=decision.refs)
                self.resume_at_boundary(active)
                return
            self.log("recovery_rejected", action=decision.action, reason="lost_pickup_not_established")
        elif decision.action != "retry":
            mode = decision.action
            fresh = fresh and self.required_recovery_refs(active, mode, step) <= set(decision.refs)
            if mode == "close_gripper":
                width = gripper_width(observation)
                if (active.goal.predicate == Predicate.HELD and fresh
                        and width is not None and width <= CLOSED_GRIP_M):
                    mode = "open_gripper"
                    self.log("recovery_adjusted", action=mode, reason="gripper_already_closed")
            if fresh:
                if mode == "return_to_home" and self.move_at_boundary(active, mode):
                    return
                if mode != "return_to_home" and self.transition(active.step, mode=mode):
                    self.dispatch()
                    return
            self.log("recovery_rejected", action=decision.action, reason="missing_current_authorization")
        self.resume_at_boundary(active)

    def resume_at_boundary(self, active: ActiveTask) -> None:
        """继续安全候选，补证抓持，或改变观察条件后再复查。"""
        assert self.tree is not None
        progress = self.memory.progress
        current = self.working.latest
        goal = self.tree.goal(progress, self.memory) if progress < len(self.tree.plan.goals) else None
        if goal is not None and self.transition(progress):
            self.dispatch()
            return
        if (goal is not None and goal.predicate in {Predicate.PLACED, Predicate.POURED}
                and grasp_settled(self.working.frames, current.sequence)):
            pickup = self.memory.pickup_for(progress, goal.target)
            if pickup is not None and self.verify(inspect=True, pickup=pickup):
                return
        if self.move_at_boundary(active, "return_to_home") or self.move_at_boundary(active, "hold_position"):
            return
        if not self.verify(inspect=True):
            reason = ("Observer budget reached without current evidence for another recovery"
                      if self.observer_count >= self.limits.observer_calls
                      else "No safe continuation after verification and recovery at the current boundary")
            self.stop("failure", reason)

    def move_at_boundary(self, active: ActiveTask, mode: ExecutionMode) -> bool:
        """每次策略尝试后至多进行一次同类兜底动作。"""
        key = active.step, active.attempt, mode
        if key in self.boundary_recoveries or not self.transition(active.step, mode=mode):
            return False
        self.boundary_recoveries.add(key)
        self.log("boundary_recovery_committed", mode=mode, goal_step=active.step)
        self.dispatch()
        return True

    def dispatch_after_commit(self, task: str) -> None:
        """提交后让局部 Planner 与第一块动作并发。"""
        self.expand(task)
        self.dispatch()

    def candidate_for(self, step: int, mode: ExecutionMode = "policy") -> ActiveTask | None:
        """将意图和当前物理状态绑定成待检查的候选。"""
        assert self.tree is not None
        goal = self.tree.goal(step, self.memory)
        if goal is None:
            return None
        active = self.executor.active
        return ActiveTask(
            step, goal, self.attempts.get(step, 0) + int(mode == "policy"),
            active.epoch + 1 if active else 1, self.working.latest.sequence,
            self.tree.branches[step].revision, PI05, mode,
        )

    def transition(self, step: int, mode: ExecutionMode = "policy") -> bool:
        """匹配候选版本、合法边界与 ACK，并在发布前复核实际状态。"""
        assert self.tree is not None
        candidate = self.candidate_for(step, mode)
        if candidate is None:
            return False
        active = self.executor.active
        reason = IntentGate.blocked(candidate, self.executor, self.memory, self.tree, self.working)
        if reason is not None:
            self.executor.pending = None
            self.log("hold", reason=reason, candidate_step=step)
            return False
        ack = None
        if active is not None:
            identity = replace(candidate, start_step=-1)
            pending = self.executor.pending
            if pending is None or pending.next_task != identity:
                pending = self.executor.request_handoff(identity)
                self.log("handoff_requested", **asdict(pending))
            ack = self.executor.acknowledge(pending)
            if ack is None:
                return False
            if IntentGate.blocked(candidate, self.executor, self.memory, self.tree, self.working):
                self.executor.pending = None
                return False
            self.log("handoff_ack", **asdict(ack))
        self.executor.commit(candidate, ack)
        self.working.anchor = self.working.latest
        if active is None or active.step != step or active.goal != candidate.goal:
            self.goal_anchor = self.working.latest
        self.last_verified_step = candidate.start_step
        self.verification_stride_anchor = candidate.start_step
        self.same_boundary_queries = 0
        self.completed_chunks = 0
        if mode == "retreat":
            self.retreated.add(step)
        elif mode == "policy":
            self.attempts[step] = candidate.attempt
        if active is not None and active.step != step:
            self.recent_judgments = ()
        self.log("intent_committed", **asdict(candidate))
        return True

    def dispatch(self, *, steps: int | None = None) -> None:
        """派发当前控制代的一块动作。"""
        active = self.executor.active
        assert active is not None
        if active.mode == "policy":
            self.vla_count += 1
        remaining = self.limits.max_steps - self.working.latest.sequence
        self.start_job("motion", self.executor.execute(
            active, min(remaining, steps) if steps is not None else remaining,
            self.deadline, self.publish_frame,
        ))

    def start_job(self, name: str, request: Coroutine[object, object, object]) -> None:
        """让语义或动作回执唤醒调度器。"""
        job = asyncio.create_task(request)
        self.jobs[name] = job
        job.add_done_callback(lambda _: self.wakeup.set())

    def publish_frame(self, frame: ObservationEvent) -> None:
        """常规动作块执行中，证据达到采样间隔即可唤醒调度。"""
        self.working.frames.append(frame)
        active = self.executor.active
        if (active is not None and active.mode == "policy" and "verify" not in self.jobs
                and self.observer_count < self.limits.observer_calls and self.verification_due(active)):
            self.wakeup.set()

    def verification_due(self, active: ActiveTask) -> bool:
        """在采样跨度满足且近期物理反馈可能支持效果时发起常规核验。"""
        end = self.working.latest.sequence
        if end - max(self.last_verified_step, self.verification_stride_anchor) < self.limits.verify_stride:
            return False
        predicate = active.goal.predicate
        if predicate not in {Predicate.HELD, Predicate.PLACED}:
            return True
        initial_width = gripper_width(self.working.anchor.observation)
        open_width = OPEN_GRIP_M
        if initial_width is not None and CLOSED_GRIP_M < initial_width < OPEN_GRIP_M:
            open_width = min(OPEN_GRIP_M, initial_width + GRIPPER_KEYFRAME_DELTA_M)
        since = max(active.start_step, self.last_verified_step)
        for frame in reversed(self.working.frames):
            if frame.sequence <= since:
                break
            width = gripper_width(frame.observation)
            if width is None:
                return True
            if (predicate == Predicate.HELD and CLOSED_GRIP_M < width < OPEN_GRIP_M
                    and grasp_settled(self.working.frames, frame.sequence)):
                return True
            if predicate == Predicate.PLACED and (width <= CLOSED_GRIP_M or width >= open_width):
                return True
        return False

    def expand(self, task: str) -> None:
        """从单个在途前沿异步补全尚缺的观察与恢复分支。"""
        assert self.tree is not None and self.executor.active is not None
        active = self.executor.active
        if active.step in self.expanded or "expand" in self.jobs or self.planner_count >= self.limits.planner_calls:
            return
        self.expanded.add(active.step)
        self.planner_count += 1
        self.expansion = self.tree.request(self.planner_count, active, self.memory)
        self.log("expansion_requested", **asdict(self.expansion))
        self.start_job("expand", self.planner.expand(
            task, self.expansion, tuple(self.memory.facts.values()), self.working.latest.observation,
        ))

    def verify(self, *, inspect: bool = False, pickup: Fact | None = None) -> bool:
        """复用近期帧检查当前条件，补证时采用更宽的时间窗口。"""
        assert self.tree is not None and self.executor.active is not None
        active = self.executor.active
        end = self.working.latest.sequence
        if end != self.last_verified_step:
            self.same_boundary_queries = 0
        if "verify" in self.jobs or self.observer_count >= self.limits.observer_calls:
            return False
        if (pickup is None and self.same_boundary_queries >= 2
                or pickup is not None and self.last_pickup_review == (end, pickup.step)):
            return False
        self.record_drawer_pull(active)
        goals = [(active.step, active.goal)]
        branch = self.tree.branches[active.step]
        next_step = branch.successor
        if active.goal.predicate in {Predicate.HELD, Predicate.POURED} and next_step is not None:
            successor = self.tree.goal(next_step, self.memory)
            if successor is not None and successor.target == active.goal.target and (
                active.goal.predicate == Predicate.HELD and successor.predicate in {Predicate.PLACED, Predicate.POURED}
                or active.goal.predicate == successor.predicate == Predicate.POURED
                and successor.destination == active.goal.destination
            ):
                goals.append((next_step, successor))
        clearance = None
        step, goal = goals[-1]
        next_step = self.tree.branches[step].successor
        if goal.predicate == Predicate.PLACED and next_step is not None:
            successor = self.tree.goal(next_step, self.memory)
            if successor is not None and successor.closes(goal.destination):
                clearance = goal.destination
        sample_task = active
        identity_reference = None
        if pickup is not None:
            goals = [(pickup.step, pickup.goal)]
            clearance = None
            sample_task = replace(active, goal=pickup.goal)
            identity_reference = self.memory.grasp_references.get(pickup.step)
        recheck_step = None
        confirmed_step = self.memory.progress - 1
        confirmed_goal = dict(goals).get(confirmed_step)
        if inspect and pickup is None and confirmed_goal is not None and confirmed_goal.predicate != Predicate.POURED:
            recheck_step = confirmed_step
            if confirmed_goal.predicate == Predicate.PLACED:
                sample_task = replace(active, goal=confirmed_goal)
                placed_at = self.memory.facts[confirmed_step].last
                identity_reference = next(
                    (frame for frame in reversed(self.working.frames) if frame.sequence == placed_at), None,
                )
        predicates = dict.fromkeys(goal.predicate for _, goal in goals)
        focus = " ".join(self.library.templates[predicate].focus for predicate in predicates).strip()
        if recheck_step is not None:
            previous_step, previous = self.memory.latest_checks[recheck_step]
            if previous_step == end and previous.outcome == Outcome.FALSE:
                focus = f"{focus} Verify the reported contradiction using the exact target's appearance and location.".strip()
        frames = self.working.sample(
            sample_task, inspect=inspect, include_pour=Predicate.POURED in predicates,
        )
        fixture_reference = None
        if active.goal.predicate in {Predicate.OPENED, Predicate.CLOSED}:
            if len(frames) > 3:
                frames = (frames[0], *frames[-2:])
            if self.goal_anchor.observation.rgb is not None and self.goal_anchor.sequence < frames[0].sequence:
                fixture_reference = self.goal_anchor
        window = EvidenceWindow(
            self.observer_count + 1, self.episode_id, active,
            frames, tuple(goals), focus, clearance, at_boundary=inspect,
            inspect_drawer=self.tree.inspecting_drawer(active.step),
            identity_reference=identity_reference,
            recheck_step=recheck_step,
            pickup_review=pickup.step if pickup is not None else None,
            fixture_reference=fixture_reference,
        )
        query = (
            active.start_step, window.goals, tuple(frame.sequence for frame in window.frames),
            window.at_boundary, window.focus, window.clearance, window.inspect_drawer, window.recheck_step,
            window.pickup_review, identity_reference.sequence if identity_reference is not None else None,
            fixture_reference.sequence if fixture_reference is not None else None,
        )
        if query in self.queried_windows:
            self.log("verification_reused", goal_steps=[step for step, _ in goals], epoch=active.epoch)
            return False
        self.queried_windows.add(query)
        if pickup is None:
            self.same_boundary_queries += 1
        else:
            self.last_pickup_review = end, pickup.step
        self.last_verified_step = end
        self.observer_count += 1
        self.window = window
        self.log("verification_requested", id=self.window.request_id, epoch=active.epoch,
                 goal_steps=[step for step, _ in goals], frames=self.window.frame_refs(),
                 captures=[(frame.sequence, frame.captured_at) for frame in self.window.frames],
                 gripper_qpos={
                     frame.sequence: frame.observation.robot_state.get("gripper_qpos")
                     for frame in self.window.frames
                 },
                 mode=branch.unresolved if inspect else "verify", focus=focus, clearance=clearance,
                 inspect_drawer=self.window.inspect_drawer, pickup_review=pickup.step if pickup is not None else None,
                 recheck_step=recheck_step,
                 fixture_reference=fixture_reference.sequence if fixture_reference is not None else None,
                 identity_reference=self.window.identity_reference.sequence if self.window.identity_reference else None)
        self.start_job("verify", self.observer.verify(window))
        return True

    def stop(self, status: EpisodeStatus, reason: str) -> None:
        """记录唯一终止决定，不再发出动作。"""
        self.result = status, reason
        self.log("terminate", status=status, reason=reason)

    def log(self, event: str, **values: object) -> None:
        """记录实际时间和物理步，供离线复核。"""
        self.journal.append({"event": event, "at": monotonic(), "physical_step": self.working.latest.sequence,
                             **values})
