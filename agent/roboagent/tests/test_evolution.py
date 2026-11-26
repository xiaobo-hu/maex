"""验证 RoboAgent-Evolve V3 的事件、Wiki、Skill 与数据隔离契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai import SimpleStreamOptions, create_models
from ai.providers.faux import faux_assistant_message, faux_provider
from roboagent.evolution.compiler import (
    compile_batch,
    compile_paired_comparisons,
    compile_reviewed_events,
    compile_run,
)
from roboagent.evolution.gate import compare_rollouts, promote_candidate
from roboagent.evolution.replay import create_model_decider, replay_tools
from roboagent.evolution.schema import AllowedAction, ReplayCase
from roboagent.evolution.workspace import initialize_workspace, validate_workspace
from roboagent.skills import apply_task_execution_skill, load_task_execution_skill


def test_skill_loader_checks_hash_and_applies_only_when_configured(tmp_path: Path) -> None:
    skill_dir = _write_skill_file(tmp_path / "v1", "## Routing\nUse evidence.")
    skill = load_task_execution_skill(skill_dir / "SKILL.md")
    assert apply_task_execution_skill("base", None) == "base"
    assert skill.content in apply_task_execution_skill("base", skill)
    (skill_dir / "SKILL.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_task_execution_skill(skill_dir / "SKILL.md")


def test_replay_exposes_recovery_tools() -> None:
    assert {tool.name for tool in replay_tools()} == {
        "create_task_plan",
        "execute_vla",
        "restage_vla",
        "adjust_ee_pose",
        "close_gripper",
        "open_gripper",
        "return_to_home",
        "advance_task_plan",
        "repair_task_plan",
        "finish",
    }


@pytest.mark.asyncio
async def test_replay_records_text_only_response_as_incorrect_decision() -> None:
    models = create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses([faux_assistant_message("I would preserve the grasp.")])
    models.set_provider(faux.provider)
    decide = create_model_decider(
        model, models.stream_simple, SimpleStreamOptions(tool_choice="required")
    )
    case = ReplayCase(
        case_id="text-only",
        task="pick object",
        task_plan={"steps": []},
        episode_memory=[],
        latest_reflection="Grounded State Reflection: grasp identity is uncertain.",
        allowed_actions=(AllowedAction("adjust_ee_pose", {"action": "world_z_up"}),),
        source_experiences=("exp-1",),
    )
    assert (await decide(case, None)).tool == "__no_tool_call__"


def test_compiler_emits_only_confirmed_stage_event(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    run = _write_run(tmp_path / "run", stage_event=True)
    first = compile_run(run, workspace)
    assert first == compile_run(run, workspace)
    assert first["experience_units"] == 1
    unit = _read_units(workspace)[0]
    assert unit["schema_version"] == 3
    assert unit["event_type"] == "successful_state_change"
    assert unit["call_span"]["call_ids"] == ["physical"]
    assert unit["evidence"]["official_stage_events"] == [
        {"stage": "01_Open_Drawer", "environment_step": 3}
    ]
    assert unit["learning"]["skill_eligible"] is True


def test_compiler_keeps_episode_without_low_value_units(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    result = compile_run(_write_run(tmp_path / "run", stage_event=False), workspace)
    assert result["experience_units"] == 0
    assert (workspace / f"experiences/episodes/{result['episode_id']}.json").is_file()
    assert _read_units(workspace) == []
    assert validate_workspace(workspace) == []


def test_compiler_requires_v3_provenance(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    run = _write_run(tmp_path / "run", stage_event=False)
    config = json.loads((run / "run_config.json").read_text())
    config.pop("runtime_revision")
    _write_json(run / "run_config.json", config)
    with pytest.raises(ValueError, match="runtime_revision"):
        compile_run(run, workspace)


def test_explicit_failure_is_quarantined(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    compile_run(
        _write_run(
            tmp_path / "run", stage_event=False, physical_status="controller_failed"
        ),
        workspace,
    )
    unit = _read_units(workspace)[0]
    assert unit["event_type"] == "failure_recovery"
    assert unit["interpretation"]["failure_domain"] == "unknown"
    assert unit["learning"]["learning_status"] == "quarantined"
    assert unit["learning"]["skill_eligible"] is False


def test_reviewed_multicall_window_is_one_experience(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    compiled = compile_run(_write_run(tmp_path / "run", stage_event=False), workspace)
    manifest = tmp_path / "events.jsonl"
    _write_jsonl(
        manifest,
        {
            "episode_id": compiled["episode_id"],
            "event_type": "failure_recovery",
            "start_call_id": "physical",
            "end_call_id": "advance",
            "interpretation": {
                "failure_domain": "agent_decision",
                "confidence": "high",
                "maintainer_summary": "The decision contradicted the available state.",
            },
            "learning": {
                "learning_status": "active",
                "skill_eligible": True,
                "exclusion_reasons": [],
            },
        },
    )
    result = compile_reviewed_events(manifest, workspace)
    assert result["reviewed_events"] == 1
    unit = _read_units(workspace)[0]
    assert unit["event_type"] == "failure_recovery"
    assert unit["call_span"]["call_ids"] == [
        "physical",
        "advance",
    ]


def test_batch_allows_episode_with_no_experience_units(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    run = _write_run(tmp_path / "run", stage_event=False)
    manifest = tmp_path / "batch.jsonl"
    _write_jsonl(
        manifest,
        {
            "batch_id": "collection-seed50",
            "task_id": 11,
            "seed": 50,
            "run_dir": str(run),
            "exit_code": 0,
            "artifact_complete": True,
        },
    )
    result = compile_batch(manifest, workspace, policy_role="baseline")
    cohort = json.loads(Path(result["cohort_manifest"]).read_text())
    assert result["experience_units"] == 0
    assert cohort["runs"][0]["experience_ids"] == []
    assert validate_workspace(workspace) == []


def test_paired_comparison_is_derived_and_not_independent(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    episode_ids = []
    for marker, role, skill in (
        ("base", "baseline", None),
        ("candidate", "candidate", "v1"),
    ):
        run = _write_run(
            tmp_path / marker,
            stage_event=True,
            skill_version=skill,
            skill_hash="a" * 64 if skill else None,
        )
        trace = json.loads((run / "trace.json").read_text())
        trace["marker"] = marker
        _write_json(run / "trace.json", trace)
        manifest = tmp_path / f"{marker}.jsonl"
        _write_jsonl(
            manifest,
            {
                "batch_id": marker,
                "task_id": 11,
                "seed": 50,
                "run_dir": str(run),
                "exit_code": 0,
                "artifact_complete": True,
            },
        )
        result = compile_batch(manifest, workspace, policy_role=role)
        episode_ids.append(
            json.loads(Path(result["cohort_manifest"]).read_text())["runs"][0][
                "episode_id"
            ]
        )
    pair_manifest = tmp_path / "pairs.jsonl"
    _write_jsonl(
        pair_manifest,
        {
            "pair_id": "base-vs-v1-task11-seed50",
            "source_episode_ids": episode_ids,
            "transfer_direction": "positive",
            "baseline_action": {"call_id": "physical", "tool": "execute_vla"},
            "candidate_action": {"call_id": "physical", "tool": "execute_vla"},
            "divergence_reason": "The outcomes differed.",
            "immediate_effect": "One action changed the state.",
            "later_consequence": "The scores differed.",
            "skill_attribution": {
                "verdict": "supported",
                "reason": "The policy differed.",
            },
            "distilled_execution_trace": [{"call_id": "physical"}],
        },
    )
    result = compile_paired_comparisons(pair_manifest, workspace)
    unit = json.loads(
        (workspace / f"experiences/units/{result['experience_ids'][0]}.json").read_text()
    )
    assert unit["derived"] is True
    assert unit["independent_evidence"] is False
    assert unit["source_episode_ids"] == episode_ids


def test_wiki_maturity_counts_unique_eligible_episodes_and_tasks(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [
        _write_source_event(workspace, "exp-1", "episode-1", 1, 50),
        _write_source_event(workspace, "exp-2", "episode-2", 1, 50),
    ]
    _write_pattern(workspace, "pattern", "probable", evidence)
    assert validate_workspace(workspace) == []
    _write_pattern(workspace, "pattern", "verified", evidence)
    assert any("two distinct tasks" in error for error in validate_workspace(workspace))


def test_skill_must_use_wiki_and_change_one_section(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [_write_source_event(workspace, "exp-1", "episode-1", 1, 50)]
    pattern = _write_pattern(workspace, "pattern", "single-shot", evidence)
    parent = _write_candidate(
        workspace,
        "v1",
        "## Routing\nOld.\n\n## Boundary\nKeep.\n",
        pattern,
    )
    candidate = _write_candidate(
        workspace,
        "v2",
        "## Routing\nNew.\n\n## Boundary\nChanged too.\n",
        pattern,
        parent_version="v1",
        changed_section="Routing",
    )
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest["source_experiences"] = ["exp-1"]
    _write_json(candidate / "manifest.json", manifest)
    errors = validate_workspace(workspace)
    assert parent.is_dir()
    assert any("must not reference Experience" in error for error in errors)
    assert any("change only section" in error for error in errors)


def test_single_shot_candidate_can_replay_but_not_formal_gate(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [_write_source_event(workspace, "exp-1", "episode-1", 1, 50)]
    pattern = _write_pattern(workspace, "pattern", "single-shot", evidence)
    candidate = _write_candidate(
        workspace, "v1", "## Routing\nUse evidence.\n", pattern
    )
    _approve_replay(workspace, candidate)
    skill = load_task_execution_skill(candidate / "SKILL.md")
    base = _write_gate_run(
        tmp_path / "base", 2, 51, None, 0, 0.0, "candidate_gate"
    )
    trial = _write_gate_run(
        tmp_path / "trial",
        2,
        51,
        skill.content_sha256,
        1,
        100.0,
        "candidate_gate",
    )
    manifest = tmp_path / "gate.json"
    _write_json(
        manifest,
        {"pairs": [{"baseline_run": str(base), "candidate_run": str(trial)}]},
    )
    with pytest.raises(ValueError, match="probable or verified"):
        compare_rollouts(workspace, candidate, manifest)


def test_gate_allows_repeated_seed_but_rejects_source_run_reuse(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [
        _write_source_event(workspace, "exp-1", "episode-1", 1, 50),
        _write_source_event(workspace, "exp-2", "episode-2", 2, 50),
    ]
    pattern = _write_pattern(workspace, "pattern", "verified", evidence)
    candidate = _write_candidate(
        workspace, "v1", "## Routing\nUse evidence.\n", pattern
    )
    _approve_replay(workspace, candidate)
    skill = load_task_execution_skill(candidate / "SKILL.md")
    base = _write_gate_run(
        tmp_path / "base", 3, 50, None, 0, 50.0, "candidate_gate"
    )
    trial = _write_gate_run(
        tmp_path / "trial",
        3,
        50,
        skill.content_sha256,
        0,
        60.0,
        "candidate_gate",
    )
    manifest = tmp_path / "gate.json"
    _write_json(
        manifest,
        {"pairs": [{"baseline_run": str(base), "candidate_run": str(trial)}]},
    )
    assert compare_rollouts(workspace, candidate, manifest)["passed"] is True
    leaked_base = _write_gate_run(tmp_path / "leaked-base", 1, 50, None, 0, 0.0, "candidate_gate")
    leaked_trial = _write_gate_run(
        tmp_path / "leaked-trial",
        1,
        50,
        skill.content_sha256,
        1,
        100.0,
        "candidate_gate",
    )
    source_unit = json.loads((workspace / "experiences/units/exp-1.json").read_text())
    source_unit["evidence"]["source_trace"] = str(leaked_base / "trace.json")
    _write_json(workspace / "experiences/units/exp-1.json", source_unit)
    source_episode = json.loads(
        (workspace / "experiences/episodes/episode-1.json").read_text()
    )
    source_episode["source_trace"] = str(leaked_base / "trace.json")
    _write_json(workspace / "experiences/episodes/episode-1.json", source_episode)
    pattern = _write_pattern(
        workspace,
        "pattern",
        "verified",
        [
            ("exp-1", str(leaked_base / "trace.json"), "call-exp-1"),
            evidence[1],
        ],
    )
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    candidate_manifest["source_patterns"][0]["content_sha256"] = hashlib.sha256(
        pattern.read_bytes()
    ).hexdigest()
    _write_json(candidate / "manifest.json", candidate_manifest)
    _write_json(manifest, {"pairs": [{"baseline_run": str(leaked_base), "candidate_run": str(leaked_trial)}]})
    with pytest.raises(ValueError, match="source Episode run"):
        compare_rollouts(workspace, candidate, manifest)


def test_seed51_gate_rejects_parent_success_regression(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [
        _write_source_event(workspace, "exp-1", "episode-1", 1, 50),
        _write_source_event(workspace, "exp-2", "episode-2", 2, 50),
    ]
    pattern = _write_pattern(workspace, "pattern", "verified", evidence)
    candidate = _write_candidate(
        workspace, "v1", "## Routing\nUse evidence.\n", pattern
    )
    _approve_replay(workspace, candidate)
    skill_hash = load_task_execution_skill(candidate / "SKILL.md").content_sha256
    pairs = []
    for task, base_tsr, candidate_tsr in ((3, 0, 1), (4, 0, 1), (5, 1, 0)):
        baseline = _write_gate_run(
            tmp_path / f"base-{task}",
            task,
            51,
            None,
            base_tsr,
            float(base_tsr * 100),
            "candidate_gate",
        )
        trial = _write_gate_run(
            tmp_path / f"trial-{task}",
            task,
            51,
            skill_hash,
            candidate_tsr,
            float(candidate_tsr * 100),
            "candidate_gate",
        )
        pairs.append({"baseline_run": str(baseline), "candidate_run": str(trial)})
    manifest = tmp_path / "gate.json"
    _write_json(manifest, {"pairs": pairs})

    report = compare_rollouts(workspace, candidate, manifest)

    assert report["candidate_score"]["sum_TSR"] > report["baseline_score"]["sum_TSR"]
    assert report["regressions"] == ["task5-seed51-pair3"]
    assert report["passed"] is False


def test_promotion_requires_two_candidate_gate_rollouts(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [
        _write_source_event(workspace, "exp-1", "episode-1", 1, 50),
        _write_source_event(workspace, "exp-2", "episode-2", 2, 50),
    ]
    pattern = _write_pattern(workspace, "pattern", "verified", evidence)
    candidate = _write_candidate(
        workspace, "v1", "## Routing\nUse evidence.\n", pattern
    )
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest["status"] = "gate_approved"
    _write_json(candidate / "manifest.json", manifest)
    reports = workspace / "runs/v1"
    _write_json(reports / "replay.json", {"passed": True})
    _write_json(
        reports / "candidate-gate.json",
        {
            "passed": True,
            "pairs": [{"candidate_run": "/runs/candidate-1"}],
        },
    )
    with pytest.raises(ValueError, match="two distinct candidate rollouts"):
        promote_candidate(workspace, candidate)
    _write_json(
        reports / "candidate-gate.json",
        {
            "passed": True,
            "pairs": [
                {"candidate_run": "/runs/candidate-1"},
                {"candidate_run": "/runs/candidate-2"},
            ],
        },
    )
    assert promote_candidate(workspace, candidate)["version"] == "v1"
    assert json.loads((candidate / "manifest.json").read_text())["status"] == "gate_approved"
    assert json.loads((workspace / "skills/active/manifest.json").read_text())["status"] == "active"
    assert validate_workspace(workspace) == []


def test_final_evaluation_requires_new_runs_and_matching_revisions(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    evidence = [
        _write_source_event(workspace, "exp-1", "episode-1", 1, 50),
        _write_source_event(workspace, "exp-2", "episode-2", 2, 50),
    ]
    pattern = _write_pattern(workspace, "pattern", "verified", evidence)
    candidate = _write_candidate(
        workspace, "v1", "## Routing\nUse evidence.\n", pattern
    )
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    candidate_manifest["status"] = "gate_approved"
    _write_json(candidate / "manifest.json", candidate_manifest)
    _write_json(workspace / "runs/v1/replay.json", {"passed": True})
    _write_json(workspace / "runs/v1/candidate-gate.json", {"passed": True})
    skill = load_task_execution_skill(candidate / "SKILL.md")
    base = _write_gate_run(
        tmp_path / "base", 3, 50, None, 1, 100.0, "final_evaluation"
    )
    trial = _write_gate_run(
        tmp_path / "trial",
        3,
        50,
        skill.content_sha256,
        1,
        100.0,
        "final_evaluation",
    )
    manifest = tmp_path / "final.json"
    _write_json(
        manifest,
        {"pairs": [{"baseline_run": str(base), "candidate_run": str(trial)}]},
    )
    assert compare_rollouts(
        workspace, candidate, manifest, phase="final-evaluation"
    )["passed"] is True

    trial_config = json.loads((trial / "run_config.json").read_text())
    trial_config["runtime_revision"] = "different-runtime"
    _write_json(trial / "run_config.json", trial_config)
    with pytest.raises(ValueError, match="config mismatch"):
        compare_rollouts(workspace, candidate, manifest, phase="final-evaluation")

    trial_config["runtime_revision"] = "runtime-v3"
    _write_json(trial / "run_config.json", trial_config)
    _write_json(
        workspace / "runs/v1/candidate-gate.json",
        {
            "passed": True,
            "pairs": [
                {"baseline_run": str(base), "candidate_run": str(trial)}
            ],
        },
    )
    with pytest.raises(ValueError, match="reuses a Candidate Gate run"):
        compare_rollouts(workspace, candidate, manifest, phase="final-evaluation")


def test_initialize_does_not_convert_legacy_workspace(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    episode = legacy / "experiences/episodes/episode-old.json"
    _write_json(episode, {"schema_version": 2})
    before = episode.read_bytes()
    before_paths = sorted(path.relative_to(legacy) for path in legacy.rglob("*"))
    with pytest.raises(ValueError, match="legacy/non-V3"):
        initialize_workspace(legacy)
    assert episode.read_bytes() == before
    assert sorted(path.relative_to(legacy) for path in legacy.rglob("*")) == before_paths
    assert not (legacy / "workspace.json").exists()


def _write_run(
    path: Path,
    *,
    stage_event: bool,
    task: int = 11,
    seed: int = 50,
    physical_status: str = "running",
    skill_version: str | None = None,
    skill_hash: str | None = None,
) -> Path:
    path.mkdir(parents=True)
    evaluation: dict[str, object] = {
        "task_id": task,
        "TSR": 0,
        "CSR": 20.0,
        "stage_done": {"01_Open_Drawer": stage_event},
        "stage_events": (
            [{"stage": "01_Open_Drawer", "environment_step": 3}]
            if stage_event
            else []
        ),
    }
    trace = {
        "episode": {
            "status": "failure",
            "details": {
                "tool_trace": [
                    {
                        "call_id": "plan",
                        "name": "create_task_plan",
                        "args": {"steps": []},
                        "result": {"status": "created", "task_plan": {"steps": []}},
                    },
                    {
                        "call_id": "physical",
                        "name": "execute_vla",
                        "args": {"step_id": "open"},
                        "result": {"status": physical_status, "executed_steps": 5},
                    },
                    {
                        "call_id": "advance",
                        "name": "advance_task_plan",
                        "args": {"completed_step_id": "open", "next_step_id": None},
                        "result": {"status": "updated", "task_plan": {"steps": []}},
                    },
                ],
                "episode_memory": [],
                "state_reflections": [
                    {
                        "after_call_id": "physical",
                        "tool": "execute_vla",
                        "text": "Grounded State Reflection: drawer opened.",
                    }
                ],
            },
        },
        "evaluation": evaluation,
    }
    _write_json(path / "trace.json", trace)
    _write_json(
        path / "run_config.json",
        {
            "task": task,
            "seed": seed,
            "planner": "qwen",
            "resolved_model": "qwen",
            "observer_model": "observer",
            "task_description": "open the drawer",
            "runtime_revision": "runtime-v3",
            "evaluator_revision": "evaluator-v3",
            "data_role": "experience_collection",
            "task_execution_skill": "candidate/SKILL.md" if skill_version else None,
            "task_execution_skill_version": skill_version,
            "task_execution_skill_sha256": skill_hash,
        },
    )
    return path


def _write_source_event(
    workspace: Path,
    experience_id: str,
    episode_id: str,
    task: int,
    seed: int,
) -> tuple[str, str, str]:
    trace = f"/trace/{episode_id}.json"
    call_id = f"call-{experience_id}"
    _write_json(
        workspace / f"experiences/episodes/{episode_id}.json",
        {
            "schema_version": 3,
            "episode_id": episode_id,
            "task_id": task,
            "seed": seed,
            "source_trace": trace,
            "provenance": {
                "runtime_revision": "runtime-v3",
                "evaluator_revision": "evaluator-v3",
                "data_role": "experience_collection",
                "planner_model": "planner",
                "observer_model": "observer",
                "seed": seed,
                "task_id": task,
            },
        },
    )
    _write_json(
        workspace / f"experiences/units/{experience_id}.json",
        {
            "schema_version": 3,
            "unit_kind": "event",
            "event_type": "decision_contradiction",
            "experience_id": experience_id,
            "episode_id": episode_id,
            "provenance": {
                "runtime_revision": "runtime-v3",
                "evaluator_revision": "evaluator-v3",
                "data_role": "experience_collection",
                "planner_model": "planner",
                "observer_model": "observer",
                "seed": seed,
                "task_id": task,
            },
            "call_span": {
                "start_call_id": call_id,
                "end_call_id": call_id,
                "call_ids": [call_id],
            },
            "evidence": {"source_trace": trace},
            "interpretation": {
                "failure_domain": "agent_decision",
                "confidence": "high",
            },
            "learning": {
                "learning_status": "active",
                "skill_eligible": True,
                "exclusion_reasons": [],
            },
        },
    )
    return experience_id, trace, call_id


def _write_pattern(
    workspace: Path,
    pattern_id: str,
    maturity: str,
    evidence: list[tuple[str, str, str]],
) -> Path:
    lines = "\n".join(
        f'- experience="{experience}" trace="{trace}" call_id="{call_id}"'
        for experience, trace, call_id in evidence
    )
    path = workspace / f"wiki/patterns/{pattern_id}.md"
    path.write_text(
        f"""---
pattern_id: {pattern_id}
category: plan_transition
maturity: {maturity}
scope: global
---
# Pattern
## Applicability
Applies to reviewed decisions.
## Observation
The evidence is consistent.
## Candidate guidance
Use the reviewed rule.
## Counterexamples
Do not generalize beyond evidence.
## Evidence
{lines}
""",
        encoding="utf-8",
    )
    return path


def _write_candidate(
    workspace: Path,
    version: str,
    content: str,
    pattern: Path,
    *,
    parent_version: str | None = None,
    changed_section: str = "Routing",
) -> Path:
    path = _write_skill_file(workspace / f"skills/candidates/{version}", content)
    pattern_text = pattern.read_text(encoding="utf-8")
    maturity = next(
        line.split(":", 1)[1].strip()
        for line in pattern_text.splitlines()
        if line.startswith("maturity:")
    )
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(
        {
            "parent_version": parent_version,
            "change_summary": "One logical rule change.",
            "changed_section": changed_section,
            "source_patterns": [
                {
                    "pattern_id": pattern.stem,
                    "maturity": maturity,
                    "content_sha256": hashlib.sha256(pattern_text.encode()).hexdigest(),
                }
            ],
        }
    )
    _write_json(path / "manifest.json", manifest)
    return path


def _write_skill_file(path: Path, content: str) -> Path:
    path.mkdir(parents=True)
    normalized = content.strip()
    (path / "SKILL.md").write_text(normalized + "\n", encoding="utf-8")
    (path / "PURPOSE.md").write_text("One atomic change.\n", encoding="utf-8")
    _write_json(
        path / "manifest.json",
        {
            "skill_id": "embodied-task-execution",
            "version": path.name,
            "status": "candidate",
            "content_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        },
    )
    return path


def _write_gate_run(
    path: Path,
    task: int,
    seed: int,
    skill_hash: str | None,
    tsr: int,
    csr: float,
    data_role: str,
) -> Path:
    path.mkdir(parents=True)
    fields = (
        "warmup_steps",
        "max_steps",
        "action_chunk_size",
        "max_vla_chunks_per_call",
        "max_agent_turns",
        "planner_retries",
        "planner_retry_delay_ms",
        "planner",
        "model",
        "resolved_model",
        "policy_host",
        "policy_port",
        "env_host",
        "env_port",
        "observer_base_url",
        "observer_model",
        "observer_timesteps",
        "observer_min_new_observations",
        "observer_gripper_threshold",
    )
    config = {field: None for field in fields}
    config.update(
        {
            "task": task,
            "seed": seed,
            "task_execution_skill_sha256": skill_hash,
            "runtime_revision": "runtime-v3",
            "evaluator_revision": "evaluator-v3",
            "data_role": data_role,
        }
    )
    _write_json(path / "run_config.json", config)
    _write_json(path / "evaluation.json", {"TSR": tsr, "CSR": csr})
    return path


def _approve_replay(workspace: Path, candidate: Path) -> None:
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest["status"] = "replay_approved"
    _write_json(candidate / "manifest.json", manifest)
    _write_json(workspace / f"runs/{manifest['version']}/replay.json", {"passed": True})


def _read_units(workspace: Path) -> list[dict[str, object]]:
    path = workspace / "experiences/units.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def _write_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
