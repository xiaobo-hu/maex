"""检查双门控、物理事件去重及迟到计划接续。"""

import asyncio
from dataclasses import replace
from math import pi
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
from maex.executive import Executive, IntentGate, Limits
from maex.executor import PI05, Executor
from maex.memory import (
    Check,
    EpisodicMemory,
    EvidenceGate,
    IntentionTree,
    ProcedureLibrary,
    WorkingMemory,
    grasp_settled,
)
from maex.models import Observer, Planner, use_checkpoint_pour_labels
from maex.procedural import ProcedureSnapshot, RecoveryCondition, RecoveryProcedure
from maex.schema import (
    ActiveTask,
    Assessment,
    BranchProposal,
    DrawerSelection,
    EvidenceWindow,
    Facet,
    Fact,
    Goal,
    Judgment,
    Outcome,
    PickupRepair,
    Plan,
    Predicate,
    RecoveryDecision,
    TreePatch,
    VisualSupport,
)
from pydantic import ValidationError

from roboagent.observer.types import ObservationEvent
from roboagent.types import ExecutionFeedback, ImageFrame, Observation
from roboagent.utils.ee_pose import EePoseAdjustmentConfig

PLACEMENT = Goal(predicate="placed", target="cookies", destination="microwave",
                 instruction="place cookies inside the microwave")
PICKUP = Goal(predicate="held", target="cookies", instruction="pick cookies")
CLOSURE = Goal(predicate="closed", target="microwave door", instruction="close the microwave door")
LIBRARY = ProcedureLibrary.load(Path(__file__).parents[1] / "procedures.json")


def frame(step: int, width: float = 0.08) -> ObservationEvent:
    return ObservationEvent(step, monotonic(), Observation(
        rgb=ImageFrame("frame"), wrist_rgb=ImageFrame("wrist frame"), robot_state={
            "eef_position": [0.0, 0.0, 0.65], "eef_axis_angle": [0.0, 0.0, 0.0],
            "gripper_qpos": [width / 2, -width / 2],
        },
    ))


def claim(step: int, end: int, facets: tuple[Facet, ...]) -> Judgment:
    return Judgment(step=step, outcome=Outcome.TRUE, summary="Visible direct evidence",
                    support=tuple(VisualSupport(facet=facet, refs=(f"front_{end}",)) for facet in facets))


def test_placement_requires_actual_separation_and_window_references() -> None:
    active = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(3, "ep", active, (frame(0), frame(10), frame(20)),
                            ((0, PLACEMENT),), "", "microwave")
    required = LIBRARY.templates[Predicate.PLACED].evidence
    assert EvidenceGate.check(window, claim(0, 20, required[:-1]), required).outcome == Outcome.UNKNOWN
    assert EvidenceGate.check(window, claim(0, 23, required), required).reason == "unknown_evidence_reference"
    assert EvidenceGate.check(window, claim(0, 0, required), required).reason == "placement_and_release_not_aligned"
    earlier = EvidenceGate.check(window, claim(0, 10, required + (Facet.CLEAR,)), required)
    assert earlier.fact is not None and earlier.fact.last == 10
    assert not earlier.hand_clear
    accepted = EvidenceGate.check(window, claim(0, 20, required), required)
    assert accepted.fact is not None
    assert not accepted.hand_clear


def test_rechecked_placement_requires_current_front_clearance_without_contradiction() -> None:
    active = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(3, "ep", active, (frame(0), frame(20)), ((0, PLACEMENT),),
                            "", "microwave", recheck_step=0)
    required = LIBRARY.templates[Predicate.PLACED].evidence
    ambiguous = Judgment(step=0, outcome=Outcome.UNKNOWN, summary="The recess hides the object.", support=(
        VisualSupport(facet=Facet.CLEAR, refs=("wrist_20",)),
    ))
    assert not EvidenceGate.check(window, ambiguous, required).hand_clear
    clear = ambiguous.model_copy(update={"support": (VisualSupport(facet=Facet.CLEAR, refs=("front_20",)),)})
    assert EvidenceGate.check(window, clear, required).hand_clear
    contradicted = clear.model_copy(update={"outcome": Outcome.FALSE, "support": (
        *clear.support, VisualSupport(facet=Facet.CONTRADICTION, refs=("front_20",)),
    )})
    assert not EvidenceGate.check(window, contradicted, required).hand_clear
    with pytest.raises(ValidationError, match="exactly one judgment"):
        Assessment(judgments=(clear, contradicted))


def test_placement_uses_release_then_destination_and_preserves_clearance() -> None:
    active = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(3, "ep", active, (frame(0, 0.042), frame(10), frame(20)),
                            ((0, PLACEMENT),), "", "microwave")
    required = LIBRARY.templates[Predicate.PLACED].evidence
    later_destination = Judgment(step=0, outcome=Outcome.TRUE, summary="The box is inside.",
                                support=(
                                    VisualSupport(facet=Facet.IDENTITY, refs=("wrist_0",)),
                                    VisualSupport(facet=Facet.RELEASE, refs=("wrist_10",)),
                                    VisualSupport(facet=Facet.DESTINATION, refs=("wrist_20",)),
                                    VisualSupport(facet=Facet.CLEAR, refs=("front_20",)),
                                ))
    completed = EvidenceGate.check(window, later_destination, required)
    assert completed.fact is not None and completed.fact.last == 20 and completed.hand_clear
    backwards = later_destination.model_copy(update={"support": (
        later_destination.support[0],
        VisualSupport(facet=Facet.RELEASE, refs=("wrist_20",)),
        VisualSupport(facet=Facet.DESTINATION, refs=("wrist_10",)),
    )})
    assert EvidenceGate.check(window, backwards, required).outcome == Outcome.UNKNOWN


def test_gripper_still_at_occupied_width_blocks_placement_claim() -> None:
    active = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(3, "ep", active, (frame(0, 0.062), frame(20, 0.062)),
                            ((0, PLACEMENT),), "", None)
    required = LIBRARY.templates[Predicate.PLACED].evidence
    placement = claim(0, 20, required)
    assert EvidenceGate.check(window, placement, required).reason == "release_conflicts_with_gripper_state"


def test_pour_cycle_needs_two_times_and_is_not_counted_twice() -> None:
    pour = Goal(predicate="poured", target="milk", destination="mug", instruction="pour milk into mug once")
    active = ActiveTask(0, pour, 1, 1, 0, 0, PI05)
    required = LIBRARY.templates[Predicate.POURED].evidence
    second = pour.model_copy(update={"occurrence": 2})
    window = EvidenceWindow(
        1, "ep", active, (frame(0), frame(12), frame(35), frame(50), frame(70)),
        ((0, pour), (1, second)), "", None,
    )
    at_end = claim(0, 35, required)
    assert EvidenceGate.check(window, at_end, required).reason == "pour_cycle_not_observed"
    observed = at_end.model_copy(update={"support": tuple(
        VisualSupport(facet=item.facet, refs=("front_12",)) if item.facet == Facet.EFFECT else item
        for item in at_end.support
    )})
    checked = EvidenceGate.check(window, observed, required)
    memory = EpisodicMemory()
    assert memory.record(0, 35, checked)
    assert checked.fact is not None
    duplicate = replace(checked.fact, step=1, goal=second)
    assert not memory.record(1, 35, replace(checked, fact=duplicate))
    assert memory.progress == 1
    later = claim(1, 70, required).model_copy(update={"support": (
        VisualSupport(facet=Facet.IDENTITY, refs=("front_50",)),
        VisualSupport(facet=Facet.EFFECT, refs=("front_50",)),
        VisualSupport(facet=Facet.UPRIGHT, refs=("front_70",)),
    )})
    assert memory.record(1, 70, EvidenceGate.check(window, later, required))
    assert memory.progress == 2


def test_regrasp_after_first_pour_keeps_the_count_and_can_start_second_pour() -> None:
    pickup = Goal(predicate="held", target="milk", instruction="pick milk")
    first_pour = Goal(predicate="poured", target="milk", destination="mug", instruction="pour milk into mug 1st")
    second_pour = first_pour.model_copy(update={"occurrence": 2, "instruction": "pour milk into mug 2nd"})
    memory = EpisodicMemory()
    tree = IntentionTree("ep", Plan(goals=(pickup, first_pour, second_pour)), LIBRARY)
    original_grasp = Fact(0, pickup, 1, 15, 15, 1, ("front_15",))
    first_cycle = Fact(1, first_pour, 2, 35, 45, 2, ("front_35", "front_45"))
    assert memory.record(0, 15, Check(Outcome.TRUE, "verified", original_grasp))
    assert memory.record(1, 45, Check(Outcome.TRUE, "verified", first_cycle))
    dropped = frame(72, 0.001)
    reacquired = frame(81, 0.042)
    working = WorkingMemory(frame(45, 0.04), 3)
    working.frames.extend((dropped, reacquired))
    executor = Executor(None, None, reacquired, EePoseAdjustmentConfig(), 20)
    candidate = ActiveTask(2, second_pour, 2, 5, reacquired.sequence, 0, PI05)
    assert IntentGate.blocked(candidate, executor, memory, tree, working) == "grasp_not_current"

    memory.reopen_grasp(0, dropped.sequence)
    new_grasp = Fact(0, pickup, 4, 81, 81, 3, ("front_81",))
    assert memory.record(0, 81, Check(Outcome.TRUE, "verified", new_grasp))
    assert memory.facts[1] == first_cycle and memory.progress == 2
    assert IntentGate.blocked(candidate, executor, memory, tree, working) is None


@pytest.mark.parametrize("correct_grasp", (True, False))
async def test_current_regrasp_is_verified_without_rewriting_pour_history(correct_grasp) -> None:
    pickup = Goal(predicate="held", target="milk", instruction="pick milk")
    pour = Goal(predicate="poured", target="milk", destination="mug", instruction="pour milk into mug 1st")
    second = pour.model_copy(update={"occurrence": 2, "instruction": "pour milk into mug 2nd"})
    first_pick = Fact(0, pickup, 1, 15, 15, 1, ("front_15",))
    first_cycle = Fact(1, pour, 2, 30, 40, 2, ("front_30", "front_40"))

    class TestObserver:
        async def verify(self, window):
            assert window.goals == ((0, pickup),)
            assert window.frames[0].sequence > 72
            assert window.identity_reference.sequence == first_pick.first
            assert "front_15" not in window.frame_refs()
            if correct_grasp:
                judgment = claim(0, window.end, LIBRARY.templates[Predicate.HELD].evidence)
            else:
                judgment = Judgment(
                    step=0, outcome=Outcome.FALSE, summary="The gripper holds the mug instead of the milk.",
                    support=(VisualSupport(facet=Facet.CONTRADICTION, refs=(f"wrist_{window.end}",)),),
                )
            return Assessment(judgments=(judgment,)), {}

    class TestPolicy:
        async def infer(self, instruction, observation):
            assert instruction == (second.instruction if correct_grasp else pickup.instruction)
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0, 0.04).observation, 1)

    executor = Executor(TestRobot(), TestPolicy(), frame(45, 0.04), EePoseAdjustmentConfig(), 100)
    executive = Executive(None, TestObserver(), executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(pickup, pour, second)), LIBRARY)
    assert executive.memory.record(0, 15, Check(Outcome.TRUE, "verified", first_pick))
    executive.memory.grasp_references[0] = frame(15, 0.04)
    assert executive.memory.record(1, 40, Check(Outcome.TRUE, "verified", first_cycle))
    assert executive.transition(2)
    active = executor.active
    executive.working.frames.extend(frame(step, 0.04) for step in range(80, 86))
    executor.latest = executive.working.latest
    executive.memory.last_grasp_break[pickup.target] = 72
    executive.apply_recovery(active, executor.latest.sequence, RecoveryDecision(
        action="retry", refs=(), reason="The same object is visibly held again.",
    ), {})

    await executive.jobs["verify"]
    executive.receive()
    if not correct_grasp:
        assert executive.memory.facts == {1: first_cycle}
        executive.expanded.add(0)
        executive.decide("Pour milk into the mug twice.")
    chunk = await executive.jobs["motion"]
    assert chunk.task.step == (2 if correct_grasp else 0) and chunk.task.epoch == active.epoch + 1
    if correct_grasp:
        assert executive.memory.facts == {0: first_pick, 1: first_cycle}
        assert executive.memory.latest_checks[0][1].fact.first == 85


async def test_second_pour_reopens_pickup_if_the_confirmed_first_pour_lost_its_grasp() -> None:
    first_pour = Goal(predicate="poured", target=PICKUP.target, destination="mug", instruction="first pour")
    second_pour = first_pour.model_copy(update={"occurrence": 2, "instruction": "second pour"})

    class TestPolicy:
        async def infer(self, instruction, observation):
            assert instruction == PICKUP.instruction
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0).observation, 1)

    executor = Executor(TestRobot(), TestPolicy(), frame(15, 0.04), EePoseAdjustmentConfig(), 100)
    executive = Executive(None, None, executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PICKUP, first_pour, second_pour)), LIBRARY)
    grasp = Fact(0, PICKUP, 1, 15, 15, 1, ("front_15",))
    assert executive.memory.record(0, 15, Check(Outcome.TRUE, "verified", grasp))
    assert executive.transition(1)
    completed = Fact(1, first_pour, executor.active.epoch, 31, 40, 2, ("front_31", "front_40"))
    assert executive.memory.record(1, 40, Check(Outcome.TRUE, "verified", completed))
    executor.latest = frame(45)
    executive.working.frames.append(executor.latest)

    executive.apply_recovery(executor.active, 45, RecoveryDecision(
        action="regrasp", refs=("wrist_45",), reason="The container is detached on the table.",
    ), {})
    assert executive.memory.facts == {1: completed}
    assert (await executive.jobs["motion"]).task.goal == PICKUP


@pytest.mark.parametrize("preposition", ("into", "over"))
def test_checkpoint_pour_labels_preserve_target_destination_and_count(preposition) -> None:
    first = Goal(predicate="poured", target="milk", destination="mug", instruction="pour milk into mug first")
    second = first.model_copy(update={"instruction": "pour milk into mug again", "occurrence": 2})
    different = first.model_copy(update={"destination": "frypan", "instruction": "pour milk over frypan"})
    plan = Plan(goals=(first, second, different))
    bound = use_checkpoint_pour_labels(plan, (
        f"1. pick milk\n2. pour milk {preposition} red coffee mug 1st\n"
        f"3. pour milk {preposition} red coffee mug 2nd\n4. place milk bowl drainer"
    ))
    assert [goal.instruction for goal in bound.goals] == [
        f"pour milk {preposition} red coffee mug 1st",
        f"pour milk {preposition} red coffee mug 2nd",
        "pour milk over frypan",
    ]
    assert [goal.occurrence for goal in bound.goals] == [1, 2, 1]


async def test_checkpoint_cleanup_is_optional_for_microwave_placement() -> None:
    class TestPlanner(Planner):
        async def ask(self, name, output, prompt, context, observation):
            return Plan(goals=(PLACEMENT, CLOSURE)), {}

    planner = TestPlanner(None, None, None, 1)
    task = "Put cream into the microwave and then put popcorn into the location where the cream was placed."
    plan, _ = await planner.plan(task, frame(0).observation)
    assert plan.goals == (PLACEMENT,)
    explicit, _ = await planner.plan(f"{task} Then close the microwave door.", frame(0).observation)
    assert explicit.goals == (PLACEMENT, CLOSURE)


@pytest.mark.parametrize(("mode", "refs", "action"), [
    ("release_gripper", ("front_20", "wrist_20"), "restage_vla"),
    ("release_gripper", ("front_19", "wrist_20"), "retry"),
    ("restage_vla", ("front_20", "wrist_20"), "retry"),
])
async def test_occluded_placement_can_retract_once_with_two_current_views(mode, refs, action) -> None:
    verified_pickup = frame(10, 0.04).observation

    class TestPlanner(Planner):
        async def ask(self, name, output, prompt, context, observation, *, before=None):
            assert before is verified_pickup
            return PickupRepair(needs_regrasp=False, clear_retraction=True, refs=refs,
                                reason="The object's support and retraction path are visible."), {}

    planner = TestPlanner(None, None, None, 1)
    active = ActiveTask(1, PLACEMENT, 1, 3, 15, 0, PI05, mode)
    decision, _ = await planner.recover("Put cookies in the microwave.", active, 20, frame(20).observation,
                                        (), "", 0, [], before=verified_pickup)
    assert decision.action == action


def test_opening_requires_visible_access_and_fixed_view() -> None:
    opened = Goal(predicate="opened", target="microwave", instruction="open the microwave door")
    active = ActiveTask(0, opened, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(1, "ep", active, (frame(0), frame(12), frame(20)), ((0, opened),), "", None)
    required = LIBRARY.templates[Predicate.OPENED].evidence
    wrist_only = claim(0, 20, required).model_copy(update={"support": tuple(
        VisualSupport(facet=facet, refs=("wrist_20",)) for facet in required
    )})
    assert EvidenceGate.check(window, wrist_only, required).reason == "door_clear_opening_unverified"
    mixed = claim(0, 20, required).model_copy(update={"support": (
        VisualSupport(facet=Facet.IDENTITY, refs=("front_0",)),
        VisualSupport(facet=Facet.EFFECT, refs=("front_20",)),
        VisualSupport(facet=Facet.APERTURE, refs=("wrist_12",)),
    )})
    assert EvidenceGate.check(window, mixed, required).outcome == Outcome.UNKNOWN
    single = claim(0, 20, required)
    assert EvidenceGate.check(window, single, required).outcome == Outcome.TRUE


def test_drawer_selection_uses_observed_contents_from_before_closure() -> None:
    drawers = ("top drawer", "middle drawer")
    inspection = tuple(
        Goal(predicate=predicate, target=target, instruction=f"{predicate} {target}")
        for target in drawers for predicate in (Predicate.OPENED, Predicate.CLOSED)
    )
    late = (
        Goal(predicate="opened", target="$drawer", instruction="open $drawer again"),
        PLACEMENT.model_copy(update={"destination": "$drawer", "instruction": "place cookies into $drawer"}),
        Goal(predicate="closed", target="$drawer", instruction="close $drawer"),
    )
    selection = DrawerSelection(condition="empty", candidates=drawers)
    with pytest.raises(ValidationError, match=r"Use \$drawer"):
        Plan(goals=inspection + late[1:], drawer_selection=selection)
    plan = Plan(goals=inspection + late, drawer_selection=selection)
    tree = IntentionTree("ep", plan, LIBRARY)
    memory = EpisodicMemory()
    assert tree.goal(5, memory) is None
    required = LIBRARY.templates[Predicate.OPENED].evidence
    for step, state in ((0, Facet.OCCUPIED), (2, Facet.EMPTY)):
        goal = plan.goals[step]
        active = ActiveTask(step, goal, 1, step + 1, 0, 0, PI05)
        window = EvidenceWindow(step + 1, "ep", active, (frame(0), frame(10), frame(20)),
                                ((step, goal),), "", None, inspect_drawer=goal.target)
        opened = claim(step, 20, required)
        assert EvidenceGate.check(window, opened, required).reason == "drawer_contents_unverified"
        contents = opened.model_copy(update={"support": (*opened.support,
            VisualSupport(facet=Facet.EFFECT, refs=("front_10",)),
            VisualSupport(facet=Facet.APERTURE, refs=("front_10",)),
            VisualSupport(facet=state, refs=("wrist_10",)),
        )})
        checked = EvidenceGate.check(window, contents, required)
        assert memory.record(step, 20, checked)
        closed = Fact(step + 1, plan.goals[step + 1], step + 2, 28, 28, step + 2, ("front_28",))
        assert memory.record(step + 1, 28, Check(Outcome.TRUE, "verified", closed))
    reading = memory.drawer_contents["middle drawer"]
    assert reading.state == "empty" and reading.step == 10 and "wrist_10" in reading.refs
    assert tree.goal(5, memory).destination == "middle drawer"


def test_grasp_can_outlive_its_image_only_with_continuous_gripper_state() -> None:
    held = Goal(predicate="held", target="tomato sauce", instruction="pick tomato sauce")
    active = ActiveTask(0, held, 1, 1, 0, 0, PI05)
    window = EvidenceWindow(
        1, "ep", active, (frame(0), frame(20, 0.039), frame(30, 0.039)),
        ((0, held),), "", None,
    )
    required = LIBRARY.templates[Predicate.HELD].evidence
    visible_grasp = claim(0, 20, required)
    assert EvidenceGate.check(window, visible_grasp, required).outcome == Outcome.TRUE
    opened = replace(window, frames=(*window.frames, frame(31)))
    assert EvidenceGate.check(opened, visible_grasp, required).reason == "gripper_state_conflict"
    new_pickup = replace(window, active=replace(active, start_step=20), frames=window.frames[1:])
    assert EvidenceGate.check(new_pickup, visible_grasp, required).reason == "grasp_outside_attempt"


def test_closing_through_object_width_is_not_a_settled_grasp() -> None:
    closing = [frame(step, width) for step, width in enumerate((0.071, 0.064, 0.057, 0.050, 0.042))]
    assert not grasp_settled(closing, 4)
    holding = [frame(step, width) for step, width in enumerate((0.043, 0.042, 0.042, 0.041, 0.042))]
    assert grasp_settled(holding, 4)


def test_grasp_window_discards_the_old_attempt_anchor() -> None:
    memory = WorkingMemory(frame(0), timesteps=5)
    memory.frames.extend(frame(step) for step in range(1, 101))
    active = ActiveTask(0, PICKUP, 1, 1, 0, 0, PI05)
    sample = memory.sample(active)
    assert sample[0].sequence == 40
    assert sample[-1].sequence == 100
    assert memory.anchor not in sample


def test_first_settled_grasp_window_keeps_context_for_a_pour_precondition() -> None:
    memory = WorkingMemory(frame(0), timesteps=5)
    memory.frames.extend(frame(step, 0.08 if step < 19 else 0.04) for step in range(1, 24))
    active = ActiveTask(0, PICKUP, 1, 1, 0, 0, PI05)
    sample = memory.sample(active, include_pour=True)
    assert sample[0].sequence < 19
    assert sample[-1].sequence == 23


def test_stable_grasp_window_also_preserves_successor_pour_pose() -> None:
    working = WorkingMemory(frame(0), 5)
    for step in range(1, 101):
        item = frame(step, 0.052 if step < 45 or step >= 80 else 0.08)
        if step == 92:
            item.observation.robot_state["eef_axis_angle"] = [0.0, 1.0, 0.0]
        working.frames.append(item)
    active = ActiveTask(0, PICKUP, 1, 1, 0, 0, PI05)
    sample = [item.sequence for item in working.sample(active, include_pour=True)]
    assert sample[0] == 79 and sample[-1] == 100
    assert 92 in sample


def test_stable_grasp_window_keeps_precontact_and_lift_frames_within_budget() -> None:
    working = WorkingMemory(frame(0), 5)
    for step in range(1, 41):
        item = frame(step, 0.08 if step < 16 else 0.052)
        item.observation.robot_state["eef_position"][2] = 0.81 if step == 30 else 0.65
        working.frames.append(item)
    active = ActiveTask(0, PICKUP, 1, 1, 0, 0, PI05)
    sample = [item.sequence for item in working.sample(active)]
    assert sample == sorted(sample) and len(sample) <= working.timesteps
    assert {15, 20, 30, 40} <= set(sample)


def test_drawer_pull_needs_continuous_planar_motion_with_grip() -> None:
    start = frame(10, 0.04)
    middle = frame(11, 0.04)
    finish = frame(12, 0.042)
    middle.observation.robot_state["eef_position"] = [0.05, 0.0, 0.65]
    finish.observation.robot_state["eef_position"] = [0.10, 0.0, 0.65]
    working = WorkingMemory(start, 3)
    working.frames.extend((middle, finish))
    assert working.drawer_pull(10) == 12
    working.frames[1] = frame(11, 0.08)
    assert working.drawer_pull(10) is None


async def test_observer_uses_goal_predicate_and_latest_images() -> None:
    sent = {}

    async def complete(**kwargs):
        sent.update(kwargs)
        result = Assessment(judgments=(Judgment(
            step=0, outcome=Outcome.UNKNOWN, support=(), summary="The grasp is unclear",
        ),))
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=result.model_dump_json()),
        )], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    active = ActiveTask(0, PICKUP, 1, 1, 0, 0, PI05)
    hint = "Is the box still lying flat on the table?"
    window = EvidenceWindow(
        1, "ep", active, (frame(10), frame(20)), ((0, PICKUP),), hint, None,
        identity_reference=frame(5, 0.04),
    )
    await Observer(client, "test").verify(window)
    text = [item["text"] for item in sent["messages"][1]["content"] if item["type"] == "text"]
    assert any(hint in section for section in text)
    assert any('"predicate": "held"' in section for section in text)
    assert any("LATEST step 20" in section for section in text)
    assert any("Verified appearance reference" in section for section in text)
    assert "wrist_5" not in window.frame_refs()
    assert all("Match the named panel" not in section for section in text)


async def test_fixture_verification_keeps_the_goal_reference_across_a_retry() -> None:
    sent = {}

    async def complete(**kwargs):
        sent.update(kwargs)
        result = Assessment(judgments=(Judgment(
            step=0, outcome=Outcome.UNKNOWN, support=(), summary="The drawer is occluded.",
        ),))
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=result.model_dump_json()),
        )], usage=None)

    drawer = Goal(predicate="opened", target="middle drawer", instruction="Open the middle drawer.")
    start = frame(0)
    robot = Executor(None, None, start, EePoseAdjustmentConfig(), 100)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    executive = Executive(None, Observer(client, "test"), robot, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(drawer,)), LIBRARY)
    assert executive.transition(0)
    executive.working.frames.extend(frame(step) for step in range(1, 81))
    robot.latest = executive.working.latest
    assert executive.transition(0)
    executive.working.frames.extend(frame(step) for step in range(81, 151))
    robot.latest = executive.working.latest
    assert executive.verify()
    window = executive.window
    assert window is not None and window.fixture_reference is start and window.end == 150
    assert len(window.frames) == 3
    assert all(item.sequence >= 80 for item in window.frames)
    assert "front_0" not in window.frame_refs()
    required = LIBRARY.templates[Predicate.OPENED].evidence
    assert EvidenceGate.check(window, claim(0, 0, required), required).reason == "unknown_evidence_reference"

    await executive.jobs["verify"]
    payload = sent["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in payload) == 7
    assert any("exact vertical level" in item.get("text", "") for item in payload)
    assert any("before this fixture goal at step 0" in item.get("text", "") for item in payload)


def test_late_patch_retains_only_compatible_suffix() -> None:
    memory = EpisodicMemory()
    tree = IntentionTree("ep", Plan(goals=(PLACEMENT, CLOSURE)), LIBRARY)
    active = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    request = tree.request(1, active, memory)
    fact = Fact(0, PLACEMENT, 1, 10, 10, 2, ("front_10",))
    assert memory.record(0, 10, Check(Outcome.TRUE, "verified", fact))
    patch = TreePatch(branches=(
        BranchProposal(step=0, focus="Look for cookies", recovery="retry"),
        BranchProposal(step=1, focus="Look at the door hinge", recovery="retry"),
    ))
    assert tree.reconcile(request, patch, memory) == {0: "completed_prefix", 1: "attached"}
    assert tree.branches[0].revision == 0 and tree.branches[1].revision == 1
    assert tree.reconcile(request, patch, memory)[1] == "incompatible_dependency"


def test_later_placement_inherits_the_verified_destination() -> None:
    successor = Goal(predicate="placed", target="popcorn", destination_from=0, destination="microwave",
                     instruction="place popcorn inside $destination")
    tree = IntentionTree("ep", Plan(goals=(PLACEMENT, successor, CLOSURE)), LIBRARY)
    memory = EpisodicMemory()
    assert tree.goal(1, memory) is None
    fact = Fact(0, PLACEMENT, 1, 10, 10, 2, ("front_10",))
    assert memory.record(0, 10, Check(Outcome.TRUE, "verified", fact))
    bound = tree.goal(1, memory)
    assert bound.destination == "microwave" and bound.destination_from is None
    assert bound.instruction == "place popcorn inside microwave"


def test_handoff_rejects_ack_from_an_older_action_boundary() -> None:
    executor = Executor(None, None, frame(0), EePoseAdjustmentConfig(), 20)
    old = ActiveTask(0, PLACEMENT, 1, 1, 0, 0, PI05)
    candidate = ActiveTask(1, CLOSURE, 1, 2, 0, 0, PI05)
    executor.commit(old)
    request = executor.request_handoff(replace(candidate, start_step=-1))
    ack = executor.acknowledge(request)
    executor.latest = frame(1)
    with pytest.raises(ValueError, match="ACK"):
        executor.commit(replace(candidate, start_step=1), ack)
    executor.commit(replace(candidate, start_step=1), executor.acknowledge(request))
    assert executor.active.epoch == 2


async def test_observer_starts_when_frames_are_ready_inside_an_action_chunk() -> None:
    verify_started = asyncio.Event()
    actions = []

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7] * 12

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            actions.append(chunk)
            if len(actions) == 9:
                await verify_started.wait()
            return ExecutionFeedback("success" if len(actions) == 12 else "running", frame(0).observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PLACEMENT,)), {}

    class TestObserver:
        async def verify(self, window):
            assert window.end == 8
            verify_started.set()
            return Assessment(judgments=(Judgment(
                step=0, outcome=Outcome.UNKNOWN, support=(), summary="Still observing.",
            ),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 100)
    episode = await Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        max_steps=12, verify_stride=2, planner_calls=1, wall_seconds=0.5,
    )).run("Place cookies into the microwave.")
    assert episode.status == "success"
    events = [event["event"] for event in episode.details["journal"]]
    assert events.index("verification_requested") < events.index("chunk_finished")


@pytest.mark.parametrize(("goal", "approach_width", "effect_width", "verification_step", "max_steps"), (
    (PICKUP, 0.08, 0.04, 11, 12), (PLACEMENT, 0.04, 0.08, 7, 8),
))
async def test_routine_verification_waits_for_possible_gripper_evidence(
    goal, approach_width, effect_width, verification_step, max_steps,
) -> None:
    actions = 0
    requests = []

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7] * 4

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            nonlocal actions
            actions += 1
            width = approach_width if actions < 7 else effect_width
            status = "success" if actions == max_steps else "running"
            observation = frame(0, width).observation
            if goal.predicate == Predicate.PLACED:
                observation.robot_state["gripper_command"] = -1.0
            return ExecutionFeedback(status, observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(goal,)), {}

    class TestObserver:
        async def verify(self, window):
            requests.append(window.end)
            return Assessment(judgments=(Judgment(
                step=0, outcome=Outcome.UNKNOWN, support=(), summary="Still observing.",
            ),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0, approach_width), EePoseAdjustmentConfig(), 100)
    episode = await Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        max_steps=max_steps, verify_stride=2, planner_calls=1, wall_seconds=1,
    )).run(goal.instruction)
    assert episode.status == "success"
    assert requests == [verification_step]


async def test_early_verification_preserves_the_action_boundary_cadence() -> None:
    actions = 0
    requests = []

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7] * 4

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            nonlocal actions
            actions += 1
            return ExecutionFeedback("success" if actions == 24 else "running", frame(0).observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PLACEMENT,)), {}

    class TestObserver:
        async def verify(self, window):
            requests.append(window.end)
            return Assessment(judgments=(Judgment(
                step=0, outcome=Outcome.UNKNOWN, support=(), summary="Still observing.",
            ),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 100)
    episode = await Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        max_steps=24, verify_stride=9, planner_calls=1, wall_seconds=1,
    )).run(PLACEMENT.instruction)
    assert episode.status == "success"
    assert requests == [9, 21]


async def test_static_boundary_uses_one_verification_for_the_same_input() -> None:
    class TestObserver:
        async def verify(self, window):
            return Assessment(judgments=(Judgment(
                step=0, outcome=Outcome.UNKNOWN, support=(), summary="The object is occluded.",
            ),)), {}

    executor = Executor(None, None, frame(0), EePoseAdjustmentConfig(), 100)
    executive = Executive(None, TestObserver(), executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PLACEMENT,)), LIBRARY)
    assert executive.transition(0)
    for step in (20, 21):
        executor.latest = frame(step)
        executive.working.frames.append(executor.latest)
        assert executive.verify(inspect=True)
        await executive.jobs["verify"]
        executive.receive()
        assert not executive.verify(inspect=True)
    assert executive.observer_count == 2


async def test_verification_uses_registered_focus_instead_of_speculative_descriptions() -> None:
    class TestObserver:
        async def verify(self, window):
            assert window.focus == "Check separation from the table."

    template = LIBRARY.templates[Predicate.HELD].model_copy(update={"focus": "Check separation from the table."})
    library = LIBRARY.model_copy(update={"templates": {**LIBRARY.templates, Predicate.HELD: template}})
    executor = Executor(None, None, frame(0), EePoseAdjustmentConfig(), 100)
    executive = Executive(None, TestObserver(), executor, library, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PICKUP,)), library)
    assert executive.transition(0)
    request = executive.tree.request(1, executor.active, executive.memory)
    patch = TreePatch(branches=(
        BranchProposal(step=0, focus="The small blue cookie box is still on the table.", recovery="retry"),
    ))
    assert executive.tree.reconcile(request, patch, executive.memory) == {0: "attached"}
    assert executive.verify()
    await executive.jobs["verify"]


async def test_stale_grasp_is_rechecked_and_explained_to_recovery_planner() -> None:
    class TestObserver:
        async def verify(self, window):
            assert window.pickup_review == 0
            return Assessment(judgments=(Judgment(
                step=0, outcome=Outcome.UNKNOWN, support=(), summary="The object is occluded.",
            ),)), {}

    class TestPlanner:
        async def recover(self, task, active, step, observation, judgments, hint, lost_pickup, history, *, before):
            assert "Fresh visual evidence of the held cookies is required" in hint
            return RecoveryDecision(action="restage_vla", refs=("front_20", "wrist_20"),
                                    reason="The wrist can retract while keeping its grip."), {}

    executor = Executor(None, None, frame(0, 0.04), EePoseAdjustmentConfig(), 100)
    executive = Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PICKUP, PLACEMENT)), LIBRARY)
    assert executive.transition(0)
    executor.latest = frame(5, 0.04)
    executive.working.frames.append(executor.latest)
    pickup = Fact(0, PICKUP, 1, 5, 5, 1, ("wrist_5",))
    assert executive.memory.record(0, 5, Check(Outcome.TRUE, "verified", pickup))
    assert executive.transition(1)
    executive.working.frames.extend(frame(step, 0.04) for step in range(16, 21))
    executor.latest = executive.working.latest
    executive.memory.last_grasp_break["cookies"] = 15

    executive.review_execution("Place the cookies.", executor.active)
    assert "recover" not in executive.jobs
    await executive.jobs["verify"]
    executive.receive()
    executive.review_execution("Place the cookies.", executor.active)
    decision, _ = await executive.jobs["recover"]
    assert decision.action == "restage_vla"


async def test_placement_is_admitted_before_clearance_and_delayed_planner_overlaps() -> None:
    closed_started = asyncio.Event()

    class TestPolicy:
        async def infer(self, instruction, observation):
            if instruction == CLOSURE.instruction:
                closed_started.set()
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0).observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PLACEMENT, CLOSURE)), {}

        async def expand(self, task, request, facts, observation):
            await closed_started.wait()
            return TreePatch(branches=tuple(
                BranchProposal(step=step, focus="Check geometry", recovery="retry")
                for step, _, _ in request.dependencies
            )), {}

    class TestObserver:
        placement_calls = 0

        async def verify(self, window):
            await asyncio.sleep(0.002)
            step = window.active.step
            required = LIBRARY.templates[window.active.goal.predicate].evidence
            if step == 0:
                self.placement_calls += 1
                if self.placement_calls > 1:
                    required += (Facet.CLEAR,)
            return Assessment(judgments=(claim(step, window.end, required),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 20)
    executive = Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        wall_seconds=3, verify_stride=2, unverified_steps=0,
    ))
    episode = await executive.run("Put cookies in the microwave and close the door.")
    assert episode.status == "success"
    assert episode.details["confirmed_prefix"] == 2
    journal = episode.details["journal"]
    holds = [event for event in journal if event["event"] == "hold"]
    assert any(event["reason"] == "placement_or_door_clearance_unknown" for event in holds)
    admitted_placement = next(event for event in journal if event["event"] == "evidence_gate"
                              and event["admitted"] and event["goal_step"] == 0)
    handoff = next(event for event in journal if event["event"] == "handoff_ack")
    assert not admitted_placement["hand_clear"] and admitted_placement["at"] < handoff["at"]
    patch = next(event for event in journal if event["event"] == "patch_returned")
    assert patch["decisions"][0] == "completed_prefix"


@pytest.mark.parametrize(("fresh", "width", "expected_command", "with_memory"), [
    (True, 0.08, 1.0, False),
    (False, 0.08, None, False),
    (True, 0.001, -1.0, False),
    (True, 0.08, 1.0, True),
    (False, 0.08, None, True),
])
async def test_uncertain_attempt_keeps_running_and_checks_gripper_recovery(
    fresh, width, expected_command, with_memory,
) -> None:
    gripper_commands = []
    received_procedures = []
    snapshot = None
    if with_memory:
        condition = RecoveryCondition.observe(PICKUP, "policy", frame(0, width).observation,
                                              Outcome.UNKNOWN, False, None)
        memory_rule = RecoveryProcedure("example", "global", condition, "close_gripper",
                                        2, 0, ("source-1", "source-2"), None, None)
        snapshot = ProcedureSnapshot(2, None, (memory_rule,))

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            command = chunk.actions[0][6]
            gripper_commands.append(command)
            measured = 0.001 if command > 0 else 0.08 if command < 0 else width
            return ExecutionFeedback("running", frame(0, measured).observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PICKUP, PLACEMENT)), {}

        async def expand(self, task, request, facts, observation):
            return TreePatch(branches=()), {}

        async def recover(self, task, active, step, observation, judgments, hint, lost_pickup, recovery_history,
                          *, before=None, procedure=None):
            if procedure is not None:
                received_procedures.append(procedure)
            assert recovery_history[-1]["mode"] == "policy"
            assert recovery_history[-1]["outcome"] == "goal_unverified"
            return RecoveryDecision(
                action="close_gripper", refs=(f"wrist_{step if fresh else step - 1}",),
                reason="The fingers straddle the intended object.",
            ), {}

    class TestObserver:
        async def verify(self, window):
            return Assessment(judgments=(Judgment(
                step=window.active.step, outcome=Outcome.UNKNOWN, support=(), summary="Not verified",
            ),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0, width), EePoseAdjustmentConfig(), 100)
    executive = Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        max_steps=9, wall_seconds=3, attempt_chunks=2,
    ), procedures=snapshot)
    episode = await executive.run("Put cookies into the microwave.")
    assert episode.details["step"] == 9
    assert episode.details["confirmed_prefix"] == 0
    committed = [event for event in episode.details["journal"] if event["event"] == "intent_committed"]
    assert all(event["step"] == 0 for event in committed)
    assert [command for command in gripper_commands if command] == (
        [expected_command] * 5 if expected_command is not None else []
    )
    recovery_feedback = [
        event for event in episode.details["journal"]
        if event["event"] == "attempt_reviewed" and event["mode"] != "policy"
    ]
    assert len(recovery_feedback) == int(expected_command is not None)
    if recovery_feedback:
        report = recovery_feedback[0]
        assert report["outcome"] == "executed" and report["end_step"] - report["start_step"] == 5
        assert report["gripper_after_m"] == (0.001 if expected_command > 0 else 0.08)
    assert committed[-1]["mode"] == "policy"
    assert received_procedures == ([memory_rule] if with_memory else [])
    assert sum(event["event"] == "procedure_committed" for event in episode.details["journal"]) == (
        int(with_memory and fresh)
    )


async def test_dropped_object_reopens_the_grasp_and_preserves_completed_work() -> None:
    class TestPolicy:
        async def infer(self, instruction, observation):
            assert instruction == PICKUP.instruction
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0).observation, 1)

    first = frame(20, width=0.001)
    executor = Executor(TestRobot(), TestPolicy(), first, EePoseAdjustmentConfig(), 20)
    executive = Executive(None, None, executor, LIBRARY, Limits())
    first_placement = PLACEMENT.model_copy(update={"target": "butter"})
    plan = Plan(goals=(first_placement, PICKUP, PLACEMENT))
    executive.tree = IntentionTree("ep", plan, LIBRARY)
    accepted = Fact(0, first_placement, 1, 4, 4, 1, ("front_4",))
    pickup = Fact(1, PICKUP, 2, 8, 8, 2, ("wrist_8",))
    assert executive.memory.record(0, 4, Check(Outcome.TRUE, "verified", accepted))
    assert executive.memory.record(1, 8, Check(Outcome.TRUE, "verified", pickup))
    placing = ActiveTask(2, PLACEMENT, 1, 3, first.sequence, 0, PI05)
    executor.commit(placing)
    executive.apply_recovery(placing, first.sequence, RecoveryDecision(
        action="regrasp", refs=("front_20",), reason="The cookies fell onto the table.",
    ), {})
    assert executive.memory.facts == {0: accepted}
    assert executive.memory.progress == 1
    assert executive.memory.corrections[-1]["supersedes_request_id"] == pickup.request_id
    chunk = await executive.jobs["motion"]
    assert chunk.task.step == 1 and chunk.task.epoch == 4


@pytest.mark.parametrize(("refs", "allowed"), [
    (("front_20", "wrist_20"), True),
    (("front_20", "wrist_15"), False),
    (("wrist_20",), False),
])
async def test_placement_release_requires_both_current_views(refs, allowed) -> None:
    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7]

    actions = []

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            actions.extend(chunk.actions)
            width = 0.08 if chunk.actions[0][6] < 0 else 0.04
            return ExecutionFeedback("running", frame(0, width=width).observation, 1)

    current = frame(20, width=0.04)
    executor = Executor(TestRobot(), TestPolicy(), current, EePoseAdjustmentConfig(), 100)
    executive = Executive(None, None, executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PICKUP, PLACEMENT)), LIBRARY)
    pickup = Fact(0, PICKUP, 1, 10, 10, 1, ("wrist_10",))
    assert executive.memory.record(0, 10, Check(Outcome.TRUE, "verified", pickup))
    placing = ActiveTask(1, PLACEMENT, 1, 2, current.sequence, 0, PI05)
    executor.commit(placing)
    executive.apply_recovery(placing, current.sequence, RecoveryDecision(
        action="release_gripper", refs=refs, reason="The object rests on the receptacle floor.",
    ), {})

    chunk = await executive.jobs["motion"]
    assert chunk.task.mode == ("release_gripper" if allowed else "policy")
    assert actions == ([[0.0] * 6 + [-1.0]] * 5 if allowed else [[0.0] * 7])
    assert executive.memory.progress == 1


async def test_grasp_lost_before_handoff_resumes_pickup() -> None:
    class TestPolicy:
        async def infer(self, instruction, observation):
            assert instruction == PICKUP.instruction
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0).observation, 1)

    class TestPlanner:
        async def recover(self, task, active, step, observation, judgments, hint, lost_pickup, recovery_history,
                          *, before=None):
            assert active.goal == PICKUP
            return RecoveryDecision(action="retry", refs=(), reason="Retry the same pickup."), {}

    current = frame(20, width=0.001)
    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 20)
    executive = Executive(TestPlanner(), None, executor, LIBRARY, Limits())
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(PICKUP, PLACEMENT)), LIBRARY)
    assert executive.transition(0)
    executor.latest = current
    executive.working.frames.extend((frame(15, width=0.04), current))
    pickup = Fact(0, PICKUP, 1, 15, 15, 1, ("wrist_15",))
    assert executive.memory.record(0, 15, Check(Outcome.TRUE, "verified", pickup))
    executive.last_verified_step = current.sequence
    executive.same_boundary_queries = 2

    executive.decide("Put cookies into the microwave.")
    await executive.jobs["recover"]
    executive.receive()
    chunk = await executive.jobs["motion"]
    assert executive.result is None
    assert executive.memory.progress == 0
    assert executive.memory.corrections[-1]["supersedes_request_id"] == pickup.request_id
    assert chunk.task.step == 0 and chunk.task.epoch == 2


async def test_recovery_validation_retries_once_at_the_same_action_boundary() -> None:
    requested_steps = []

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0).observation, 1)

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PICKUP,)), {}

        async def expand(self, task, request, facts, observation):
            return TreePatch(branches=()), {}

        async def recover(self, task, active, step, observation, judgments, hint, lost_pickup, recovery_history,
                          *, before=None):
            requested_steps.append(step)
            if len(requested_steps) == 1:
                return RecoveryDecision.model_validate({"action": "unsupported", "refs": [], "reason": "Test"}), {}
            assert "literal_error" in hint
            return RecoveryDecision(action="retry", refs=(), reason="Retry the same pickup."), {}

    class TestObserver:
        async def verify(self, window):
            return Assessment(judgments=(Judgment(
                step=window.active.step, outcome=Outcome.UNKNOWN, support=(), summary="Not verified",
            ),)), {}

    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 100)
    executive = Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        max_steps=4, attempt_chunks=2, wall_seconds=3,
    ))
    episode = await executive.run("Pick up the cookies.")
    assert requested_steps == [2, 2]
    assert episode.details["step"] == 4
    assert episode.details["model_calls"]["planner"] == 4


@pytest.mark.parametrize(("placement_visible", "placed_during_pickup"), (
    (True, False), (False, False), (False, True),
))
async def test_blocked_placement_recovers_a_view_and_keeps_working(placement_visible, placed_during_pickup) -> None:
    class TestPolicy:
        async def infer(self, instruction, observation):
            assert instruction == CLOSURE.instruction
            return [[0.0] * 7]

    class TestRobot:
        async def return_to_home(self, grip, cancel_event):
            assert grip == -1.0
            return ExecutionFeedback("running", current.observation, 3, {"home_pose_reached": True})

        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("success", current.observation, 1)

    class TestPlanner:
        async def recover(self, task, active, step, observation, judgments, hint, lost_pickup, history,
                          *, before=None):
            return RecoveryDecision(action="retry", refs=(), reason="The wrist blocks the view."), {}

    class TestObserver:
        async def verify(self, window):
            assert window.active.mode == "return_to_home"
            assert window.recheck_step == 1
            if placement_visible:
                judgment = claim(1, window.end, (Facet.IDENTITY, Facet.DESTINATION, Facet.CLEAR))
            else:
                judgment = Judgment(
                    step=1, outcome=Outcome.UNKNOWN, summary="The arm is clear. The drawer hides the cookies.",
                    support=(VisualSupport(facet=Facet.CLEAR, refs=(f"front_{window.end}",)),),
                )
            return Assessment(judgments=(judgment,)), {}

    current = frame(20)
    current.observation.robot_state["gripper_command"] = -1.0
    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 100)
    executive = Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits())
    executive.tree = IntentionTree(
        executive.episode_id, Plan(goals=(PICKUP, PLACEMENT, CLOSURE)), LIBRARY,
    )
    if placed_during_pickup:
        assert executive.transition(0)
    pickup = Fact(0, PICKUP, 1, 10, 10, 1, ("front_10",))
    assert executive.memory.record(0, 10, Check(Outcome.TRUE, "verified", pickup))
    executor.latest = frame(10, width=0.04)
    executive.working.frames.append(executor.latest)
    if not placed_during_pickup:
        assert executive.transition(1)
    executor.latest = current
    executive.working.frames.append(current)
    placed = Fact(1, PLACEMENT, 1, 20, 20, 2, ("front_20",))
    assert executive.memory.record(1, 20, Check(Outcome.TRUE, "verified", placed))
    executive.last_verified_step = 20
    executive.same_boundary_queries = 2
    executive.expanded.add(2)

    executive.decide("Place cookies into the microwave and close it.")
    await executive.jobs["recover"]
    executive.receive()
    assert (await executive.jobs["motion"]).task.mode == "return_to_home"
    executive.receive()
    executive.decide("Place cookies into the microwave and close it.")
    await executive.jobs["verify"]
    executive.receive()
    executive.decide("Place cookies into the microwave and close it.")
    assert (await executive.jobs["motion"]).task.goal == CLOSURE
    executive.receive()
    assert executive.result[0] == "success"
    assert executive.memory.facts[1] == placed


async def test_initial_plan_can_be_repaired_after_two_invalid_responses() -> None:
    feedbacks = []

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            feedbacks.append(feedback)
            if len(feedbacks) <= 2:
                Goal(predicate="placed", target="cookies", instruction="place cookies")
            return Plan(goals=(PLACEMENT,)), {}

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("success", frame(0).observation, 1)

    executor = Executor(TestRobot(), TestPolicy(), frame(0), EePoseAdjustmentConfig(), 100)
    episode = await Executive(TestPlanner(), None, executor, LIBRARY, Limits(
        planner_calls=3,
    )).run("Place cookies into the microwave.")
    assert episode.status == "success" and episode.details["step"] == 1
    assert feedbacks[0] == "" and all("invalid destination" in feedback for feedback in feedbacks[1:])


@pytest.mark.parametrize("home_reached", (True, False))
async def test_workspace_exit_only_allows_guarded_home_before_retrying(home_reached) -> None:
    actions = []
    outside = frame(1, 0.04)
    outside.observation.robot_state.update(eef_position=[0.41, 0.0, 0.65], gripper_command=0.6)
    inside = frame(2, 0.04)
    inside.observation.robot_state["gripper_command"] = 0.6

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PICKUP,)), {}

    class TestPolicy:
        async def infer(self, instruction, observation):
            assert observation is not outside.observation
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            actions.append("policy")
            if len(actions) == 1:
                return ExecutionFeedback("running", outside.observation, 1)
            return ExecutionFeedback("success", inside.observation, 1)

        async def return_to_home(self, grip, cancel_event):
            assert grip == 0.6
            actions.append("home")
            return ExecutionFeedback("running", inside.observation, 1, {"home_pose_reached": home_reached})

    class TestObserver:
        async def verify(self, window):
            return Assessment(judgments=(Judgment(
                step=window.active.step, outcome=Outcome.UNKNOWN,
                summary="No effect verified after homing.", support=(),
            ),)), {}

    workspace = EePoseAdjustmentConfig(
        workspace_min_m=(-0.45, -0.40, 0.30), workspace_max_m=(0.35, 0.65, 1.20),
    )
    executor = Executor(TestRobot(), TestPolicy(), frame(0), workspace, 100)
    episode = await Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        planner_calls=1, max_steps=3,
    )).run("Pick up the cookies.")
    assert actions == (["policy", "home", "policy"] if home_reached else ["policy", "home"])
    assert episode.status == ("success" if home_reached else "safety_stop")
    assert episode.details["step"] == (3 if home_reached else 2)


@pytest.mark.parametrize("needs_retry", (False, True))
async def test_pending_official_confirmation_can_recover_the_final_effect(needs_retry) -> None:
    commands = []
    current = frame(0, 0.04).observation
    current.task_success = False
    current.robot_state["gripper_command"] = -0.7
    released = frame(8).observation
    released.task_success = False
    released.robot_state["gripper_command"] = -0.7

    class TestPlanner:
        async def plan(self, task, observation, feedback):
            return Plan(goals=(PLACEMENT,)), {}

    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.1] + [0.0] * 5 + [-0.7]] * 8

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            commands.append(chunk.actions[0])
            success = len(commands) == (11 if needs_retry else 10)
            observation = current if len(commands) < 8 else released
            return ExecutionFeedback("success" if success else "running", observation, 1)

    class TestObserver:
        async def verify(self, window):
            if window.recheck_step == 0:
                assert needs_retry
                return Assessment(judgments=(Judgment(
                    step=0, outcome=Outcome.FALSE, summary="The object is beside the microwave.",
                    support=(VisualSupport(facet=Facet.CONTRADICTION, refs=(f"front_{window.end}",)),),
                ),)), {}
            return Assessment(judgments=(claim(0, window.end, LIBRARY.templates[Predicate.PLACED].evidence),)), {}

    executor = Executor(TestRobot(), TestPolicy(), ObservationEvent(0, monotonic(), current),
                        EePoseAdjustmentConfig(), 100)
    episode = await Executive(TestPlanner(), TestObserver(), executor, LIBRARY, Limits(
        verify_stride=1, unverified_steps=0, planner_calls=1, max_steps=11, completion_wait_steps=2,
    )).run("Place the cookies into the microwave.")
    assert episode.status == "success"
    assert commands[8:10] == [[0.0] * 6 + [-0.7]] * 2
    assert episode.details["step"] == len(commands)
    assert bool(episode.details["corrections"]) == needs_retry
    if needs_retry:
        assert commands[-1][0] == 0.1


@pytest.mark.parametrize(("refs", "width", "mode"), [
    (("front_20", "wrist_20"), 0.08, "close_gripper"),
    (("wrist_20",), 0.08, "policy"),
    (("front_20", "wrist_20"), 0.04, "policy"),
])
async def test_drawer_handle_closure_checks_alignment_reference_and_gap(refs, width, mode) -> None:
    class TestPolicy:
        async def infer(self, instruction, observation):
            return [[0.0] * 7]

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            return ExecutionFeedback("running", frame(0, width).observation, 1)

    current = frame(20, width)
    executor = Executor(TestRobot(), TestPolicy(), current, EePoseAdjustmentConfig(), 100)
    executive = Executive(None, None, executor, LIBRARY, Limits())
    goal = Goal(predicate="opened", target="top drawer", instruction="open top drawer")
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(goal,)), LIBRARY)
    assert executive.transition(0)
    executive.apply_recovery(executor.active, current.sequence, RecoveryDecision(
        action="close_gripper", refs=refs, reason="The jaws align with the named drawer handle.",
    ), {})
    chunk = await executive.jobs["motion"]
    assert chunk.task.mode == mode and executive.memory.progress == 0


@pytest.mark.parametrize(("axis_angle", "moves"), [
    ([0.0, pi / 2, 0.0], True),
    ([0.0, 0.0, 0.0], False),
])
async def test_drawer_recovery_pulls_horizontally_with_a_preserved_grip(axis_angle, moves) -> None:
    actions = []
    current = frame(20, 0.04)
    current.observation.robot_state.update(eef_axis_angle=axis_angle, gripper_command=1.0)

    class TestRobot:
        async def execute_action_chunk(self, chunk, cancel_event):
            actions.extend(chunk.actions)
            return ExecutionFeedback("running", current.observation, 1)

    executor = Executor(TestRobot(), None, current, EePoseAdjustmentConfig(), 100)
    executive = Executive(None, None, executor, LIBRARY, Limits())
    goal = Goal(predicate="opened", target="top drawer", instruction="open top drawer")
    executive.tree = IntentionTree(executive.episode_id, Plan(goals=(goal,)), LIBRARY)
    executive.working.frames.extendleft(frame(step, 0.04) for step in (19, 18, 17, 16))
    assert executive.transition(0)
    executive.apply_recovery(executor.active, 20, RecoveryDecision(
        action="pull_drawer", refs=("front_20", "wrist_20"), reason="The engaged handle has a clear track.",
    ), {})
    chunk = await executive.jobs["motion"]
    assert chunk.task.mode == "pull_drawer"
    assert bool(actions) == moves
    if moves:
        assert all(action[0] < 0 and action[6] == 1.0 for action in actions)
    else:
        assert chunk.status == "pull_direction_unavailable"
