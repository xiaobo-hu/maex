"""验证归档、发布、跨 seed 检索及跨任务恢复的生效边界。"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from maex.executor import PI05
from maex.memory import IntentionTree, ProcedureLibrary
from maex.models import Planner
from maex.procedural import ProcedureStore, TaskContext
from maex.reflector import publish_trace, recoveries_from_trace
from maex.schema import ActiveTask, Goal, Plan

from roboagent.types import Observation

LIBRARY = ProcedureLibrary.load(Path(__file__).parents[1] / "procedures.json")


def record(tmp_path, episode_id, task_id, seed, target, *, outcome="true", execute=True) -> Path:
    """生成包含实际准入事件和恢复派发关系的单步骤轨迹。"""
    goal = Goal(predicate="held", target=target, instruction=f"Pick up {target}.")
    plan = Plan(goals=(goal,))
    bound = goal.model_dump(mode="json")
    fact = {
        "step": 0, "goal": bound, "execution_id": 2, "first": 105, "last": 105,
        "request_id": 1, "refs": ["wrist_105"],
    }
    journal = [
        {"event": "global_plan_prepared", "plan": plan.model_dump(mode="json")},
        {"event": "intent_committed", "epoch": 1, "mode": "policy", "step": 0, "goal": bound},
        {"event": "recovery_requested", "physical_step": 100, "epoch": 1, "goal_step": 0,
         "robot_state": {"gripper_qpos": [0.04, -0.04]}, "lost_pickup": None, "blocked_by": None},
        {"event": "recovery_prepared", "physical_step": 100, "epoch": 1, "decision": {"action": "close_gripper"}},
        {"event": "intent_committed", "epoch": 2, "mode": "close_gripper", "step": 0, "goal": bound, "start_step": 100},
        {"event": "verification_requested", "id": 1, "epoch": 2, "goal_steps": [0], "frames": {"wrist_105": 105}},
        {"event": "verification_returned", "id": 1, "judgments": {"judgments": [{
            "step": 0, "outcome": outcome, "support": [
                {"facet": facet, "refs": ["wrist_105"]}
                for facet in (("identity", "effect") if outcome == "true" else ("contradiction",))
            ],
        }]}},
        {"event": "evidence_gate", "goal_step": 0, "observed_at_step": 105, "outcome": outcome,
         "admitted": outcome == "true", "fact": fact if outcome == "true" else None},
    ]
    if execute:
        journal.insert(5, {
            "event": "chunk_finished", "task": {"epoch": 2}, "status": "running", "start_step": 100, "end_step": 105,
        })
    path = tmp_path / episode_id / "trace.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "episode": {
            "status": "success" if outcome == "true" else "failure",
            "details": {
                "agent": "maex", "episode_id": episode_id, "procedure_version": LIBRARY.version,
                "facts": [fact] if outcome == "true" else [], "journal": journal,
            },
        },
        # 让测试同时覆盖档案中存在独立评测数据的情况。
        "evaluation": {"TSR": int(outcome != "true")},
    }))
    path.with_name("run_config.json").write_text(json.dumps({
        "task": task_id, "seed": seed, "task_description": goal.instruction,
        "executor": PI05, "procedures": {"version": LIBRARY.version},
    }))
    return path


def test_published_cases_are_retrieved_across_seeds_and_tasks(tmp_path) -> None:
    db = ProcedureStore(tmp_path / "memory.sqlite3")
    first = record(tmp_path, "cookies-a", 1, 50, "cookies")
    assert publish_trace(db, first) == 1
    goal = Goal(predicate="held", target="cookies", instruction="Pick up cookies.")
    task = TaskContext(task_id=1, instruction="  PICK UP cookies.  ", seed=100)
    task_snapshot = db.snapshot(task, PI05, LIBRARY.version)
    assert task_snapshot.exemplar.task.seed == 50
    assert task_snapshot.exemplar.plan.goals == (goal,)
    task_rule, = task_snapshot.recoveries
    assert task_rule.scope == "task" and task_rule.matches(0, goal, task_rule.condition)
    assert not task_rule.matches(1, goal, task_rule.condition)
    assert db.snapshot(task.model_copy(update={"seed": 50}), PI05, LIBRARY.version).exemplar is None
    assert db.snapshot(task.model_copy(update={"instruction": "Pick up butter."}), PI05, LIBRARY.version).exemplar is None
    assert db.snapshot(task, PI05, "incompatible").exemplar is None

    other = TaskContext(task_id=3, instruction="Pick up popcorn.", seed=50)
    before = db.snapshot(other, PI05, LIBRARY.version)
    assert before.exemplar is None and before.recoveries == ()
    second = record(tmp_path, "butter-a", 2, 50, "butter")
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert all(revision == 2 for revision in workers.map(lambda _: publish_trace(db, second), range(2)))
    assert db.snapshot(other, PI05, LIBRARY.version, before.revision) == before
    after = db.snapshot(other, PI05, LIBRARY.version)
    assert after.exemplar is None
    global_rule, = after.recoveries
    assert global_rule.scope == "global"
    assert global_rule.source_episodes == ("cookies-a", "butter-a")

    popcorn = Goal(predicate="held", target="popcorn", instruction=other.instruction)
    tree = IntentionTree("current", Plan(goals=(popcorn,)), LIBRARY, after)
    active = ActiveTask(0, popcorn, 1, 1, 0, 0, PI05)
    mismatched = global_rule.condition.model_copy(update={"jaws": "empty_closed"})
    assert tree.prepare_recovery(active, mismatched) is None
    assert tree.prepare_recovery(active, global_rule.condition) == global_rule
    assert tree.branches[0].recovery_procedure == global_rule.id
    assert tree.prepare_recovery(active, global_rule.condition) is None
    assert tree.prepare_recovery(replace(active, epoch=2), global_rule.condition) is None


def test_unexecuted_or_contradicted_cases_do_not_supply_support(tmp_path) -> None:
    db = ProcedureStore(tmp_path / "memory.sqlite3")
    phantom = record(tmp_path, "phantom", 1, 50, "cookies", execute=False)
    assert publish_trace(db, phantom) is None
    traces = (
        record(tmp_path, "cookies-a", 1, 50, "cookies"),
        record(tmp_path, "butter-a", 2, 50, "butter"),
        record(tmp_path, "pudding-a", 4, 50, "pudding", outcome="false"),
        record(tmp_path, "cream-a", 5, 50, "cream", outcome="false"),
    )
    for trace in traces:
        publish_trace(db, trace)
    target = TaskContext(task_id=3, instruction="Pick up popcorn.", seed=50)
    assert db.snapshot(target, PI05, LIBRARY.version, 2).recoveries[0].support == 2
    assert db.snapshot(target, PI05, LIBRARY.version).recoveries == ()
    malformed = json.loads(traces[0].read_text())
    reply = next(event for event in malformed["episode"]["details"]["journal"]
                 if event["event"] == "verification_returned")
    reply["judgments"]["judgments"][0]["support"] = []
    traces[0].write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="Cannot replay MAEX fact"):
        publish_trace(db, traces[0])


def test_timeout_followed_by_cleanup_remains_interrupted(tmp_path) -> None:
    path = record(tmp_path, "timeout", 1, 50, "cookies", outcome="unknown")
    journal = json.loads(path.read_text())["episode"]["details"]["journal"]
    journal += [{"event": "terminate", "status": "timeout"}, {"event": "request_cancelled", "worker": "verify"}]
    case, = recoveries_from_trace(journal, {})
    assert case.outcome == "interrupted"
    assert case.evidence_refs == ()


async def test_task_example_only_supplies_plan_context(tmp_path) -> None:
    db = ProcedureStore(tmp_path / "memory.sqlite3")
    publish_trace(db, record(tmp_path, "cookies-a", 1, 50, "cookies"))
    task = TaskContext(task_id=1, instruction="Pick up cookies.", seed=100)
    example = db.snapshot(task, PI05, LIBRARY.version).exemplar.plan
    current_goal = example.goals[0].model_copy(update={"instruction": "Lift the cookies from the table"})
    current = Plan(goals=(current_goal,))

    class TestPlanner(Planner):
        async def ask(self, name, output, prompt, context, observation):
            assert name == "create_intention_tree"
            assert example.model_dump_json() in context
            return current, {}

    planner = TestPlanner(None, None, None, 1, task_example=example)
    plan, _ = await planner.plan(task.instruction, Observation())
    assert plan == current
