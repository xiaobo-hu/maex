"""定义 MAEX 在 Planner、Observer 与 Executive 之间传递的记录。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from roboagent.observer.types import ObservationEvent


class Predicate(StrEnum):
    """首版执行器能够验证的目标谓词。"""

    HELD = "held"
    PLACED = "placed"
    OPENED = "opened"
    CLOSED = "closed"
    POURED = "poured"


class Outcome(StrEnum):
    """物理结果的三值判断。"""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Facet(StrEnum):
    """Observer 可以引用的视觉证据。"""

    IDENTITY = "identity"
    EFFECT = "effect"
    RELEASE = "release"
    DESTINATION = "destination"
    APERTURE = "aperture"
    PRIOR_GRASP = "prior_grasp"
    UPRIGHT = "upright"
    CLEAR = "hand_clear"
    RETREAT = "retreat_clear"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    CONTRADICTION = "contradiction"


class Goal(BaseModel):
    """一个冻结的语义目标及可直接传给 π0.5 的指令。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    predicate: Predicate
    target: str = Field(min_length=1, max_length=80)
    destination: str = Field(default="", max_length=80)
    destination_from: int | None = Field(default=None, ge=0)
    occurrence: int = Field(default=1, ge=1, le=8)
    instruction: str = Field(min_length=1, max_length=240)

    def closes(self, container: str) -> bool:
        """关门目标与容器共用同一个物理边界。"""
        return (
            self.predicate == Predicate.CLOSED
            and self.target.casefold().removesuffix(" door") == container.casefold().removesuffix(" door")
        )

    @property
    def is_drawer_motion(self) -> bool:
        """打开或关闭指定抽屉的意图。"""
        return self.predicate in {Predicate.OPENED, Predicate.CLOSED} and self.target.casefold().endswith("drawer")

    @model_validator(mode="after")
    def validate_destination(self) -> Goal:
        """放置和倾倒绑定目的地，其他目标通过目标对象定位。"""
        needs_destination = self.predicate in {Predicate.PLACED, Predicate.POURED}
        inherited = self.destination_from is not None
        if needs_destination != (bool(self.destination) or inherited):
            raise ValueError(f"{self.predicate} has an invalid destination")
        if inherited and self.predicate != Predicate.PLACED:
            raise ValueError("destination_from applies to placement steps")
        if not inherited and "$destination" in self.instruction:
            raise ValueError("An instruction slot requires destination_from")
        if not inherited and re.search(r"\b(where|there|same|previous)\b", self.destination, re.IGNORECASE):
            raise ValueError("Use a concrete destination or destination_from, not a relative phrase")
        return self


class DrawerSelection(BaseModel):
    """任务根据实际检查结果选择一个抽屉。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: Literal["empty", "occupied"]
    candidates: tuple[str, ...] = Field(min_length=2, max_length=3)


class Plan(BaseModel):
    """任务顺序与需要延后确认的语义角色。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goals: tuple[Goal, ...] = Field(min_length=1, max_length=16)
    drawer_selection: DrawerSelection | None = None

    @model_validator(mode="after")
    def validate_goals(self) -> Plan:
        """不同倾倒别名仍需由前后独立的物理周期区分。"""
        if self.drawer_selection is not None:
            inspection = tuple(
                (predicate, drawer) for drawer in self.drawer_selection.candidates
                for predicate in (Predicate.OPENED, Predicate.CLOSED)
            )
            if tuple((goal.predicate, goal.target) for goal in self.goals[:len(inspection)]) != inspection:
                raise ValueError("Inspect each named drawer by opening and closing it in the requested order")
            bound_predicates = {
                goal.predicate for goal in self.goals[len(inspection):]
                if "$drawer" in (goal.target, goal.destination)
            }
            if not {Predicate.OPENED, Predicate.PLACED, Predicate.CLOSED} <= bound_predicates:
                raise ValueError("Use $drawer for reopening, placing in, and closing the drawer selected by inspection")
            for goal in self.goals[len(inspection):]:
                if goal.predicate in {Predicate.OPENED, Predicate.CLOSED} and goal.target != "$drawer":
                    raise ValueError("A drawer selected by inspection must not be named before it is observed")
                if goal.predicate == Predicate.PLACED and goal.destination != "$drawer":
                    raise ValueError("Place the object in the selected $drawer")
        occurrences: dict[tuple[str, str], int] = {}
        for step, goal in enumerate(self.goals):
            parent = goal.destination_from
            if parent is not None and (parent >= step or self.goals[parent].predicate != Predicate.PLACED):
                raise ValueError("destination_from must refer to an earlier placement step")
            if goal.predicate == Predicate.POURED:
                key = (goal.target.casefold(), goal.destination.casefold())
                occurrences[key] = occurrences.get(key, 0) + 1
                if goal.occurrence != occurrences[key]:
                    raise ValueError("Pour occurrences must be consecutive, starting at 1")
            if "$drawer" in (goal.target, goal.destination) and (
                self.drawer_selection is None or "$drawer" not in goal.instruction
            ):
                raise ValueError("$drawer requires a selection and an instruction slot")
            if "$drawer" in goal.instruction and "$drawer" not in (goal.target, goal.destination):
                raise ValueError("Drawer instruction slot must be bound to a goal role")
        return self


class BranchProposal(BaseModel):
    """Planner 对候选节点的补证问题和失败路径的建议。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    focus: str = Field(min_length=1, max_length=240)
    recovery: Literal["retry", "retreat"]


class TreePatch(BaseModel):
    """一至两个步骤的局部补全。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    branches: tuple[BranchProposal, ...] = Field(max_length=2)


class VisualSupport(BaseModel):
    """从当前窗口引用一个或多个图像。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facet: Facet
    refs: tuple[str, ...] = Field(min_length=1, max_length=6)


class Judgment(BaseModel):
    """带证据的候选判断，尚无事实和控制权限。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    outcome: Outcome
    support: tuple[VisualSupport, ...] = Field(max_length=8)
    summary: str = Field(max_length=320)


class Assessment(BaseModel):
    """一次只读观察可核查活动步骤和一个相关后继。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judgments: tuple[Judgment, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_steps(self) -> Assessment:
        """每个步骤只接收一条判断，防止同一窗口内的候选互相覆盖。"""
        if len({judgment.step for judgment in self.judgments}) != len(self.judgments):
            raise ValueError("Each observed step must have exactly one judgment")
        return self


RecoveryAction = Literal[
    "retry", "regrasp", "open_gripper", "close_gripper", "release_gripper", "pull_drawer",
    "restage_vla", "return_to_home",
]
ExecutionMode = Literal[
    "policy", "retreat", "open_gripper", "close_gripper", "release_gripper", "pull_drawer",
    "restage_vla", "return_to_home", "hold_position",
]


class RecoveryDecision(BaseModel):
    """在当前意图内选择一次恢复操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: RecoveryAction
    refs: tuple[str, ...] = Field(max_length=2)
    reason: str = Field(min_length=1, max_length=320)


class PickupRepair(BaseModel):
    """核查失去持物后的同物体抓取前提。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_regrasp: bool
    clear_retraction: bool = False
    refs: tuple[str, ...] = Field(max_length=2)
    reason: str = Field(min_length=1, max_length=320)


@dataclass(frozen=True, slots=True)
class ActiveTask:
    """已提交的绑定、执行实例和控制代。"""

    step: int
    goal: Goal
    attempt: int
    epoch: int
    start_step: int
    revision: int
    executor: str
    mode: ExecutionMode = "policy"


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """Executive 创建的只读窗口，引用由实际采集的帧生成。"""

    request_id: int
    episode_id: str
    active: ActiveTask
    frames: tuple[ObservationEvent, ...]
    goals: tuple[tuple[int, Goal], ...]
    focus: str
    clearance: str | None
    at_boundary: bool = False
    inspect_drawer: str | None = None
    identity_reference: ObservationEvent | None = None
    recheck_step: int | None = None
    pickup_review: int | None = None
    fixture_reference: ObservationEvent | None = None

    @property
    def end(self) -> int:
        """返回窗口边界的物理步号。"""
        return self.frames[-1].sequence

    def frame_refs(self) -> dict[str, int]:
        """列出能够支撑语义判断的真实视觉引用。"""
        return {
            f"{view}_{frame.sequence}": frame.sequence
            for frame in self.frames
            for view, image in (
                ("front", frame.observation.rgb),
                ("wrist", frame.observation.wrist_rgb),
            )
            if image is not None
        }


@dataclass(frozen=True, slots=True)
class Fact:
    """经核验的任务事件，记录对应的真实观察区间。"""

    step: int
    goal: Goal
    execution_id: int
    first: int
    last: int
    request_id: int
    refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """绑定在途扩展的锚点与有权更新的后继。"""

    request_id: int
    episode_id: str
    tree_id: str
    anchor: int
    execution_id: int
    source_version: int
    procedures: str
    # 结构依赖包括原始语义目标和候选分支版本。
    dependencies: tuple[tuple[int, Goal, int], ...]
