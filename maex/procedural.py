"""持久化已执行的程序案例，并在 episode 启动时冻结可复用的子树。"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from roboagent.types import Observation

from .memory import CLOSED_GRIP_M, OPEN_GRIP_M, gripper_width
from .schema import ExecutionMode, Goal, Outcome, Plan, Predicate, ProcedureReference

ReusableAction = Literal[
    "open_gripper", "close_gripper", "release_gripper", "pull_drawer", "restage_vla",
]
REUSABLE_ACTIONS = frozenset(get_args(ReusableAction))
Scope = Literal["task", "global"]
CaseOutcome = Literal["verified", "contradicted", "unresolved", "interrupted"]


class TaskContext(BaseModel):
    """以任务身份和指令限定任务经验，用 seed 区分来源回合。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: int
    instruction: str = Field(min_length=1)
    seed: int

    @property
    def key(self) -> tuple[int, str]:
        """跨 seed 匹配同一任务，同时防止编号对应的任务文字已变更。"""
        return self.task_id, " ".join(self.instruction.casefold().split())


class RecoveryCondition(BaseModel):
    """可在新回合重测的恢复前提。中间开度只描述夹爪几何。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predicate: Predicate
    drawer: bool
    mode: ExecutionMode
    jaws: Literal["open", "empty_closed", "intermediate", "unknown"]
    outcome: Outcome
    lost_pickup: bool
    blocked_by: str | None

    @classmethod
    def observe(
        cls, goal: Goal, mode: ExecutionMode, observation: Observation, outcome: Outcome,
        lost_pickup: bool, blocked_by: str | None,
    ) -> RecoveryCondition:
        """将实测反馈转换为不依赖具体物体坐标或 seed 的进入条件。"""
        width = gripper_width(observation)
        if width is None:
            jaws = "unknown"
        elif width <= CLOSED_GRIP_M:
            jaws = "empty_closed"
        elif width >= OPEN_GRIP_M:
            jaws = "open"
        else:
            jaws = "intermediate"
        return cls(
            predicate=goal.predicate, drawer=uses_drawer(goal),
            mode=mode, jaws=jaws, outcome=outcome, lost_pickup=lost_pickup, blocked_by=blocked_by,
        )


class RecoveryCase(BaseModel):
    """实际动作及其验证回执。旧帧引用只供历史复核。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    goal: Goal
    condition: RecoveryCondition
    action: ReusableAction
    start: int
    end: int
    outcome: CaseOutcome
    result_step: int | None = None
    evidence_request: int | None = None
    evidence_refs: tuple[str, ...] = ()


class EpisodeExperience(BaseModel):
    """一份可重放的归档，为库提供计划以及已尝试的条件分支。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    task: TaskContext
    executor: str
    library_version: str
    trace: str
    plan: Plan | None
    recoveries: tuple[RecoveryCase, ...]


@dataclass(frozen=True, slots=True)
class RecoveryProcedure:
    """通过回放汇总的有条件恢复子树，复用时仍需核验当前目标。"""

    id: str
    scope: Scope
    condition: RecoveryCondition
    action: ReusableAction
    support: int
    contradictions: int
    source_episodes: tuple[str, ...]
    task_step: int | None
    task_goal: Goal | None

    def applies_to(self, step: int, goal: Goal) -> bool:
        """只按静态角色过滤目录，保留尚待绑定的槽位和当前条件。"""
        if self.condition.predicate != goal.predicate:
            return False
        if goal.destination_from is None and self.condition.drawer != uses_drawer(goal):
            return False
        if self.task_goal is None:
            return True
        return (
            self.task_step == step
            and self.task_goal.occurrence == goal.occurrence
            and (goal.target == "$drawer" or goal.target.casefold() == self.task_goal.target.casefold())
            and (goal.destination_from is not None or goal.destination == "$drawer"
                 or goal.destination.casefold() == self.task_goal.destination.casefold())
        )

    def reference(self) -> ProcedureReference:
        """为树节点提供简短目录项。"""
        return ProcedureReference(self.id, self.scope, self.action, self.condition.outcome, self.support)

    def matches(self, step: int, goal: Goal, condition: RecoveryCondition) -> bool:
        """在当前步骤和绑定已知时实例化可执行的恢复候选。"""
        return self.condition == condition and (
            self.task_goal is None or (self.task_step == step and goal_roles(self.task_goal) == goal_roles(goal))
        )


def uses_drawer(goal: Goal) -> bool:
    """识别抽屉目标或抽屉目的地，包括待绑定的抽屉槽位。"""
    return goal.is_drawer_motion or goal.destination.casefold().endswith("drawer")


def goal_roles(goal: Goal) -> tuple[Predicate, str, str, int]:
    """按绑定角色和次数匹配步骤，允许等效 VLA 指令措辞。"""
    return goal.predicate, goal.target.casefold(), goal.destination.casefold(), goal.occurrence


@dataclass(frozen=True, slots=True)
class ProcedureSnapshot:
    """整个 episode 共用一个版本，发布新条目只影响之后的 episode。"""

    revision: int
    exemplar: EpisodeExperience | None
    recoveries: tuple[RecoveryProcedure, ...]


def consolidate(task: TaskContext, revision: int, episodes: tuple[EpisodeExperience, ...]) -> ProcedureSnapshot:
    """任务计划跨 seed 检索，对通用恢复按条件和独立 episode 汇总。"""
    exemplar = next(
        (episode for episode in reversed(episodes)
         if episode.task.key == task.key and episode.task.seed != task.seed and episode.plan is not None),
        None,
    )
    groups: dict[
        tuple[Scope, RecoveryCondition, ReusableAction, int | None, tuple[Predicate, str, str, int] | None],
        dict[str, list[RecoveryCase]],
    ] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        task_specific = episode.task.key == task.key
        if task_specific and episode.task.seed == task.seed:
            continue
        for case in episode.recoveries:
            if case.condition.jaws == "unknown":
                continue
            key = (
                "task" if task_specific else "global", case.condition, case.action,
                case.step if task_specific else None, goal_roles(case.goal) if task_specific else None,
            )
            groups[key][episode.episode_id].append(case)

    procedures = []
    for (scope, condition, action, step, roles), samples in groups.items():
        positive = []
        contradictions = 0
        goal = None
        for episode_id, cases in samples.items():
            if any(case.outcome == "contradicted" for case in cases):
                contradictions += 1
            elif verified := next((case for case in cases if case.outcome == "verified"), None):
                positive.append(episode_id)
                if scope == "task":
                    goal = verified.goal
        if len(positive) < (1 if scope == "task" else 2) or len(positive) <= contradictions:
            continue
        identity = repr((scope, task.key if scope == "task" else None, condition.model_dump(mode="json"),
                         action, step, roles))
        procedures.append(RecoveryProcedure(
            sha256(identity.encode()).hexdigest()[:16], scope, condition, action,
            len(positive), contradictions, tuple(positive), step, goal,
        ))
    procedures.sort(key=lambda item: (
        item.scope != "task", -item.support + item.contradictions, -item.support, item.id,
    ))
    return ProcedureSnapshot(revision, exemplar, tuple(procedures))


class ProcedureStore:
    """以 SQLite 事务发布归档，让并发写入获得唯一可回放版本。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    revision INTEGER PRIMARY KEY,
                    episode_id TEXT NOT NULL UNIQUE,
                    executor TEXT NOT NULL,
                    library_version TEXT NOT NULL,
                    record TEXT NOT NULL
                )
            """)

    def connect(self) -> sqlite3.Connection:
        """为每次读取或发布使用独立连接。"""
        return sqlite3.connect(self.path, timeout=30)

    def publish(self, episode: EpisodeExperience) -> int:
        """重复导入同一 episode 返回原版本。"""
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """INSERT INTO episodes (episode_id, executor, library_version, record) VALUES (?, ?, ?, ?)
                   ON CONFLICT(episode_id) DO NOTHING""",
                (episode.episode_id, episode.executor, episode.library_version, episode.model_dump_json()),
            )
            return connection.execute(
                "SELECT revision FROM episodes WHERE episode_id = ?", (episode.episode_id,),
            ).fetchone()[0]

    def snapshot(
        self, task: TaskContext, executor: str, library_version: str, revision: int | None = None,
    ) -> ProcedureSnapshot:
        """按执行器及程序库版本读入兼容案例。"""
        with closing(self.connect()) as connection:
            latest = connection.execute("SELECT COALESCE(MAX(revision), 0) FROM episodes").fetchone()[0]
            revision = latest if revision is None else revision
            if not 0 <= revision <= latest:
                raise ValueError(f"Experience revision {revision} is outside 0..{latest}")
            episodes = tuple(EpisodeExperience.model_validate_json(record) for (record,) in connection.execute(
                """SELECT record FROM episodes
                   WHERE executor = ? AND library_version = ? AND revision <= ? ORDER BY revision""",
                (executor, library_version, revision),
            ))
        return consolidate(task, revision, episodes)
