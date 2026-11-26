"""把 RoboMemArena trace 编译为 Episode 和高价值 Experience 事件。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from .trace_adapter import RoboMemArenaTrace

_PHYSICAL_TOOLS = {
    "execute_vla",
    "restage_vla",
    "adjust_ee_pose",
    "close_gripper",
    "open_gripper",
    "return_to_home",
}
_RESULT_FIELDS = (
    "status",
    "stop_reason",
    "step_id",
    "chunks_used",
    "executed_steps",
    "action",
    "requested_magnitude",
    "requested_distance_m",
    "target_pose",
    "applied_translation_m",
    "applied_rotation_deg",
    "command",
    "reason",
    "consecutive_vla_attempts",
    "home_pose_reached",
)
_REJECTION_RESULT_FIELDS = (
    "task_plan",
    "active_step",
    "plan_progress",
    "valid_next_step_ids",
    "valid_reopen_step_ids",
    "retry_instruction",
)
_EXPLICIT_FAILURE_STATUSES = {"failure", "controller_failed", "rejected"}
_EVENT_TYPES = {
    "successful_state_change",
    "failure_recovery",
    "collateral_effect",
    "decision_contradiction",
}
_FAILURE_DOMAINS = {
    "runtime_safety",
    "tool_capability",
    "vla_execution",
    "observation_reflection",
    "agent_decision",
    "skill_related",
    "unknown",
}
_LEARNING_STATUSES = {"active", "quarantined", "obsolete_runtime_issue"}
_CONFIDENCE_LEVELS = {"low", "medium", "high"}


def compile_run(
    run_dir: str | Path,
    workspace: str | Path,
    *,
    policy_context: dict[str, object] | None = None,
    run_status: str | None = None,
) -> dict[str, object]:
    """保存一个不可变 Episode，并只提取确定性的高价值事件。"""
    if run_status not in {None, "complete_run", "incomplete_run"}:
        raise ValueError("run_status must be complete_run or incomplete_run")
    root = Path(workspace).expanduser().resolve()
    _require_v3_workspace(root)
    source = RoboMemArenaTrace.load(run_dir)
    trace_path = source.trace_path
    trace = source.trace
    run_config = source.run_config
    episode = source.episode
    evaluation = source.evaluation
    tool_trace = source.tool_trace
    reflections = source.state_reflections

    trace_hash = _hash_json(trace)
    episode_id = f"episode-{trace_hash[:16]}"
    task_id = _integer(run_config.get("task", evaluation.get("task_id")))
    seed = _integer(run_config.get("seed"))
    task = str(run_config.get("task_description", "")).strip()
    stage_events_present = "stage_events" in evaluation
    stage_events = _stage_events(evaluation.get("stage_events", []))
    action_ranges = _physical_action_ranges(tool_trace)
    reflection_by_call = {
        str(item.get("after_call_id")): str(item.get("text", "")).strip()
        for item in reflections
        if item.get("after_call_id")
    }
    tool_by_call = {
        str(item.get("call_id")): item for item in tool_trace if item.get("call_id")
    }

    provenance = _provenance(
        run_config,
        task_id=task_id,
        seed=seed,
        policy_context=policy_context,
    )
    normalized_episode = {
        "schema_version": 3,
        "episode_id": episode_id,
        "source_trace": str(trace_path),
        "source_trace_sha256": trace_hash,
        "task_id": task_id,
        "seed": seed,
        "task": task,
        "provenance": deepcopy(provenance),
        "episode_status": episode.get("status"),
        "evaluation": evaluation,
        "hindsight_attribution": "exact" if stage_events_present else "episode_only",
    }
    if policy_context is not None:
        normalized_episode["cohort_id"] = policy_context["cohort_id"]
    if run_status is not None:
        normalized_episode["run_status"] = run_status
        normalized_episode["official_score_eligible"] = run_status == "complete_run"
    _write_immutable(root / "experiences/episodes" / f"{episode_id}.json", normalized_episode)

    units: list[dict[str, object]] = []
    latest_physical_call_id: str | None = None
    for entry in tool_trace:
        call_id = str(entry.get("call_id", "")).strip()
        tool = str(entry.get("name", "")).strip()
        if not call_id or not tool:
            continue
        consequence_events = _events_for_call(stage_events, action_ranges, call_id)
        result = entry.get("result")
        status = str(result.get("status", "")) if isinstance(result, dict) else ""
        if consequence_events:
            units.append(
                _event_payload(
                    episode_id=episode_id,
                    trace_path=trace_path,
                    trace_hash=trace_hash,
                    task=task,
                    provenance=provenance,
                    entries=[entry],
                    event_type="successful_state_change",
                    official_stage_events=consequence_events,
                    reflection_by_call=reflection_by_call,
                    run_status=run_status,
                    interpretation={
                        "failure_domain": None,
                        "confidence": "high",
                        "maintainer_summary": None,
                    },
                    learning={
                        "learning_status": "active",
                        "skill_eligible": True,
                        "exclusion_reasons": [],
                    },
                )
            )
        if status in _EXPLICIT_FAILURE_STATUSES:
            units.append(
                _event_payload(
                    episode_id=episode_id,
                    trace_path=trace_path,
                    trace_hash=trace_hash,
                    task=task,
                    provenance=provenance,
                    entries=[entry],
                    event_type="failure_recovery",
                    official_stage_events=consequence_events,
                    reflection_by_call=reflection_by_call,
                    run_status=run_status,
                    interpretation={
                        "failure_domain": "unknown",
                        "confidence": "high",
                        "maintainer_summary": None,
                    },
                    learning={
                        "learning_status": "quarantined",
                        "skill_eligible": False,
                        "exclusion_reasons": ["explicit_failure_requires_review"],
                    },
                )
            )
        if tool == "repair_task_plan" and status == "repaired":
            span = [entry]
            if latest_physical_call_id:
                trigger = tool_by_call.get(latest_physical_call_id)
                if trigger is not None:
                    span.insert(0, trigger)
            units.append(
                _event_payload(
                    episode_id=episode_id,
                    trace_path=trace_path,
                    trace_hash=trace_hash,
                    task=task,
                    provenance=provenance,
                    entries=span,
                    event_type="failure_recovery",
                    official_stage_events=[],
                    reflection_by_call=reflection_by_call,
                    run_status=run_status,
                    interpretation={
                        "failure_domain": "agent_decision",
                        "confidence": "medium",
                        "maintainer_summary": "A lost prerequisite was reopened by the Runtime.",
                    },
                    learning={
                        "learning_status": "quarantined",
                        "skill_eligible": False,
                        "exclusion_reasons": ["recovery_outcome_requires_review"],
                    },
                )
            )
        if tool in _PHYSICAL_TOOLS:
            latest_physical_call_id = call_id

    for payload in units:
        experience_id = str(payload["experience_id"])
        _write_immutable(root / "experiences/units" / f"{experience_id}.json", payload)

    _write_units_index(root)
    return {
        "episode_id": episode_id,
        "experience_units": len(units),
        "attribution": "exact" if stage_events_present else "episode_only",
        **({"run_status": run_status} if run_status is not None else {}),
    }


def compile_batch(
    manifest_path: str | Path,
    workspace: str | Path,
    *,
    policy_role: str,
) -> dict[str, object]:
    """优先导入正常运行，并保留没有正常 attempt 的实质性中断轨迹。"""
    if policy_role not in {"baseline", "candidate", "active"}:
        raise ValueError("policy_role must be baseline, candidate, or active")
    root = Path(workspace).expanduser().resolve()
    _require_v3_workspace(root)
    source_manifest = Path(manifest_path).expanduser().resolve()
    batch_id, selected = _select_batch_runs(source_manifest)
    cohort_path = root / "experiences/cohorts" / f"{batch_id}.json"
    prior_registration = _read_object(cohort_path) if cohort_path.is_file() else {}
    prior_reuse = {
        str(row.get("episode_id")): bool(row.get("reused_existing"))
        for row in prior_registration.get("runs", [])
        if isinstance(row, dict)
    }
    run_records: list[dict[str, object]] = []
    expected_loaded_skills: list[dict[str, object]] | None = None
    reused_count = 0

    for row in selected:
        run_dir = _resolve_run_dir(str(row["run_dir"]), source_manifest)
        source = RoboMemArenaTrace.load(run_dir)
        task_id = _integer(source.run_config.get("task"))
        seed = _integer(source.run_config.get("seed"))
        if task_id != row["task_id"] or seed != row["seed"]:
            raise ValueError(f"batch manifest metadata does not match {run_dir}")
        policy_context = _policy_context(
            source.run_config,
            cohort_id=batch_id,
            policy_role=policy_role,
        )
        loaded_skills = cast(list[dict[str, object]], policy_context["loaded_skills"])
        if expected_loaded_skills is None:
            expected_loaded_skills = deepcopy(loaded_skills)
        elif loaded_skills != expected_loaded_skills:
            raise ValueError("all runs in one cohort must use the same loaded Skills")

        trace_hash = _hash_json(source.trace)
        episode_id = f"episode-{trace_hash[:16]}"
        episode_path = root / "experiences/episodes" / f"{episode_id}.json"
        reused_existing = episode_path.is_file()
        reused_count += int(reused_existing)
        if reused_existing:
            existing = _read_object(episode_path)
            if existing.get("source_trace_sha256") != trace_hash:
                raise ValueError(f"existing episode trace hash mismatch: {episode_path}")
        else:
            compile_run(
                run_dir,
                root,
                policy_context=policy_context,
                run_status=str(row["run_status"]),
            )
        run_records.append(
            {
                "task_id": task_id,
                "seed": seed,
                "run_dir": str(run_dir),
                "episode_id": episode_id,
                "record_schema_version": (
                    _read_object(episode_path).get("schema_version")
                    if episode_path.is_file()
                    else 3
                ),
                "run_status": row["run_status"],
                "official_score_eligible": row["run_status"] == "complete_run",
                "reused_existing": prior_reuse.get(episode_id, reused_existing),
            }
        )

    units_by_episode = _units_by_episode(root)
    experience_unit_count = 0
    for record in run_records:
        experience_ids = units_by_episode.get(str(record["episode_id"]), [])
        record["experience_ids"] = experience_ids
        experience_unit_count += len(experience_ids)

    complete_run_count = sum(
        record["run_status"] == "complete_run" for record in run_records
    )
    incomplete_run_count = len(run_records) - complete_run_count
    cohort = {
        "schema_version": 3,
        "cohort_id": batch_id,
        "policy_role": policy_role,
        "source_manifest": str(source_manifest),
        "loaded_skills": expected_loaded_skills or [],
        "complete_run_count": complete_run_count,
        "incomplete_run_count": incomplete_run_count,
        "runs": run_records,
    }
    _write_immutable(cohort_path, cohort)
    return {
        "cohort_id": batch_id,
        "policy_role": policy_role,
        "selected_runs": len(run_records),
        "complete_runs": complete_run_count,
        "incomplete_runs": incomplete_run_count,
        "episodes_reused": reused_count,
        "experience_units": experience_unit_count,
        "cohort_manifest": str(cohort_path),
    }


def compile_reviewed_events(
    manifest_path: str | Path,
    workspace: str | Path,
) -> dict[str, object]:
    """把人工审查的多调用窗口编译为不可变 Experience 事件。"""
    root = Path(workspace).expanduser().resolve()
    _require_v3_workspace(root)
    source_manifest = Path(manifest_path).expanduser().resolve()
    episode_root = root / "experiences/episodes"
    experience_ids: list[str] = []
    for line_number, line in enumerate(
        source_manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            reviewed = _object(json.loads(line), f"{source_manifest}:{line_number}")
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_manifest}:{line_number}: {error}") from error
        episode_id = str(reviewed.get("episode_id", "")).strip()
        episode = _read_object(episode_root / f"{episode_id}.json")
        trace_path = Path(str(episode.get("source_trace", "")))
        source = RoboMemArenaTrace.load(trace_path.parent)
        entries = _reviewed_span(source.tool_trace, reviewed, source_manifest, line_number)
        event_type = str(reviewed.get("event_type", "")).strip()
        interpretation = _reviewed_interpretation(reviewed, source_manifest, line_number)
        learning = _reviewed_learning(reviewed, source_manifest, line_number)
        action_ranges = _physical_action_ranges(source.tool_trace)
        stage_events = _stage_events(source.evaluation.get("stage_events", []))
        included_stage_events: list[dict[str, object]] = []
        for entry in entries:
            included_stage_events.extend(
                _events_for_call(
                    stage_events,
                    action_ranges,
                    str(entry.get("call_id", "")),
                )
            )
        reflection_by_call = {
            str(item.get("after_call_id")): str(item.get("text", "")).strip()
            for item in source.state_reflections
            if item.get("after_call_id")
        }
        payload = _event_payload(
            episode_id=episode_id,
            trace_path=trace_path,
            trace_hash=str(episode.get("source_trace_sha256", "")),
            task=str(episode.get("task", "")),
            provenance=_object(episode.get("provenance"), "episode.provenance"),
            entries=entries,
            event_type=event_type,
            official_stage_events=included_stage_events,
            reflection_by_call=reflection_by_call,
            run_status=str(episode.get("run_status", "")) or None,
            interpretation=interpretation,
            learning=learning,
        )
        experience_id = str(payload["experience_id"])
        _write_immutable(root / "experiences/units" / f"{experience_id}.json", payload)
        experience_ids.append(experience_id)
    if not experience_ids:
        raise ValueError("reviewed event manifest is empty")
    _write_units_index(root)
    return {
        "reviewed_events": len(experience_ids),
        "experience_ids": experience_ids,
        "source_manifest": str(source_manifest),
    }


def compile_paired_comparisons(
    manifest_path: str | Path,
    workspace: str | Path,
) -> dict[str, object]:
    """把人工审查的双 Episode 分叉证据登记为不可变 Experience Units。"""
    root = Path(workspace).expanduser().resolve()
    _require_v3_workspace(root)
    source_manifest = Path(manifest_path).expanduser().resolve()
    episode_root = root / "experiences/episodes"
    provenance_by_episode = _cohort_provenance_by_episode(root)
    existing_pairs = {
        str(unit.get("pair_id")): unit
        for path in sorted((root / "experiences/units").glob("*.json"))
        if (unit := _read_object(path)).get("unit_kind") == "paired_comparison"
    }
    experience_ids: list[str] = []

    for line_number, line in enumerate(
        source_manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            reviewed = _object(
                json.loads(line), f"{source_manifest}:{line_number}"
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_manifest}:{line_number}: {error}") from error
        pair_id = str(reviewed.get("pair_id", "")).strip()
        source_episode_ids = reviewed.get("source_episode_ids")
        if (
            not pair_id
            or not isinstance(source_episode_ids, list)
            or len(source_episode_ids) != 2
            or not all(isinstance(item, str) and item for item in source_episode_ids)
            or len(set(source_episode_ids)) != 2
        ):
            raise ValueError(
                f"{source_manifest}:{line_number}: pair_id and two unique source_episode_ids are required"
            )
        episodes = [
            _read_object(episode_root / f"{episode_id}.json")
            for episode_id in source_episode_ids
        ]
        if any(episode_id not in provenance_by_episode for episode_id in source_episode_ids):
            raise ValueError(
                f"{source_manifest}:{line_number}: source episodes must belong to cohorts"
            )
        provenance = [
            provenance_by_episode[str(episode_id)] for episode_id in source_episode_ids
        ]
        if provenance[0]["cohort_id"] == provenance[1]["cohort_id"]:
            raise ValueError(
                f"{source_manifest}:{line_number}: source episodes must use distinct cohorts"
            )
        if provenance[1]["policy_role"] not in {"candidate", "active"}:
            raise ValueError(
                f"{source_manifest}:{line_number}: second episode must be candidate or active"
            )
        task_keys = {(episode.get("task_id"), episode.get("seed")) for episode in episodes}
        if len(task_keys) != 1:
            raise ValueError(
                f"{source_manifest}:{line_number}: paired episodes must share task_id and seed"
            )
        _validate_reviewed_pair(reviewed, source_manifest, line_number)
        for episode, action_key in zip(
            episodes, ("baseline_action", "candidate_action"), strict=True
        ):
            trace_path = Path(str(episode.get("source_trace", "")))
            call_id = str(reviewed[action_key]["call_id"])
            if call_id not in _trace_call_ids(trace_path):
                raise ValueError(
                    f"{source_manifest}:{line_number}: {call_id} is absent from {trace_path}"
                )

        payload = deepcopy(reviewed)
        payload.update(
            {
                "schema_version": 3,
                "unit_kind": "paired_comparison",
                "derived": True,
                "independent_evidence": False,
                "learning": {
                    "learning_status": "active",
                    "skill_eligible": reviewed.get("skill_attribution", {}).get("verdict")
                    == "supported",
                    "exclusion_reasons": (
                        []
                        if reviewed.get("skill_attribution", {}).get("verdict") == "supported"
                        else ["paired_comparison_does_not_support_skill_attribution"]
                    ),
                },
                "task": {
                    "task_id": episodes[0].get("task_id"),
                    "seed": episodes[0].get("seed"),
                    "instruction": episodes[0].get("task"),
                },
                "source_provenance": deepcopy(provenance),
                "evidence": [
                    {
                        "episode_id": episode_id,
                        "trace": episode.get("source_trace"),
                        "call_id": reviewed[action_key].get("call_id"),
                    }
                    for episode_id, episode, action_key in zip(
                        source_episode_ids,
                        episodes,
                        ("baseline_action", "candidate_action"),
                        strict=True,
                    )
                ],
            }
        )
        experience_id = f"exp-{_hash_json(payload)[:16]}"
        payload["experience_id"] = experience_id
        existing = existing_pairs.get(pair_id)
        if existing is not None and existing != payload:
            raise ValueError(f"immutable paired comparison conflicts: {pair_id}")
        _write_immutable(root / "experiences/units" / f"{experience_id}.json", payload)
        experience_ids.append(experience_id)

    if not experience_ids:
        raise ValueError("paired comparison manifest is empty")
    _write_units_index(root)
    return {
        "paired_comparisons": len(experience_ids),
        "experience_ids": experience_ids,
        "source_manifest": str(source_manifest),
    }


def _provenance(
    run_config: dict[str, object],
    *,
    task_id: int | None,
    seed: int | None,
    policy_context: dict[str, object] | None,
) -> dict[str, object]:
    planner_model = str(
        run_config.get("resolved_model") or run_config.get("model") or ""
    ).strip()
    observer_model = str(run_config.get("observer_model") or "").strip()
    required = {
        "runtime_revision": str(run_config.get("runtime_revision") or "").strip(),
        "evaluator_revision": str(run_config.get("evaluator_revision") or "").strip(),
        "data_role": str(run_config.get("data_role") or "").strip(),
        "planner_model": planner_model,
        "observer_model": observer_model,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("V3 provenance is missing: " + ", ".join(missing))
    if required["data_role"] not in {
        "experience_collection",
        "candidate_gate",
        "final_evaluation",
    }:
        raise ValueError("invalid data_role in V3 provenance")
    if task_id is None or seed is None:
        raise ValueError("V3 provenance requires task_id and seed")
    loaded_skills = (
        deepcopy(policy_context.get("loaded_skills", []))
        if policy_context is not None
        else _loaded_skills(run_config)
    )
    return {
        **required,
        "cohort_id": policy_context.get("cohort_id") if policy_context else None,
        "skill_version": (
            loaded_skills[0].get("version") if loaded_skills else None
        ),
        "skill_sha256": (
            loaded_skills[0].get("content_sha256") if loaded_skills else None
        ),
        "planner_model": planner_model,
        "observer_model": observer_model,
        "seed": seed,
        "task_id": task_id,
    }


def _require_v3_workspace(root: Path) -> None:
    marker = root / "workspace.json"
    value = _read_object(marker) if marker.is_file() else {}
    if value.get("schema_version") != 3:
        raise ValueError(f"V3 compiler requires an initialized V3 workspace: {root}")


def _loaded_skills(run_config: dict[str, object]) -> list[dict[str, object]]:
    version = str(run_config.get("task_execution_skill_version") or "").strip()
    content_hash = str(run_config.get("task_execution_skill_sha256") or "").strip()
    if not version and not content_hash:
        return []
    if not version or not content_hash:
        raise ValueError("loaded Skill provenance requires version and content hash")
    return [{"version": version, "content_sha256": content_hash}]


def _event_payload(
    *,
    episode_id: str,
    trace_path: Path,
    trace_hash: str,
    task: str,
    provenance: dict[str, object],
    entries: list[dict[str, object]],
    event_type: str,
    official_stage_events: list[dict[str, object]],
    reflection_by_call: dict[str, str],
    run_status: str | None,
    interpretation: dict[str, object],
    learning: dict[str, object],
) -> dict[str, object]:
    if event_type not in _EVENT_TYPES or not entries:
        raise ValueError("Experience event requires a valid event_type and non-empty span")
    call_ids = [str(entry.get("call_id", "")) for entry in entries]
    result_values = [
        entry.get("result") for entry in entries if isinstance(entry.get("result"), dict)
    ]
    payload: dict[str, object] = {
        "schema_version": 3,
        "unit_kind": "event",
        "event_type": event_type,
        "episode_id": episode_id,
        "task": task,
        "call_span": {
            "start_call_id": call_ids[0],
            "end_call_id": call_ids[-1],
            "call_ids": call_ids,
        },
        "provenance": deepcopy(provenance),
        "evidence": {
            "source_trace": str(trace_path),
            "source_trace_sha256": trace_hash,
            "action_results": [
                {
                    "call_id": entry.get("call_id"),
                    "tool": entry.get("name"),
                    "arguments": deepcopy(entry.get("args", {})),
                    "result": _result_summary(entry.get("result")),
                }
                for entry in entries
            ],
            "robot_state_before": _robot_state(result_values[0], "start") if result_values else None,
            "robot_state_after": _robot_state(result_values[-1], "end") if result_values else None,
            "official_stage_events": deepcopy(official_stage_events),
            "visual_artifacts": [],
            "task_effect_status": "confirmed" if official_stage_events else "unverified",
        },
        "interpretation": {
            "observer_claims": _observer_claims(entries),
            "reflections": [
                reflection_by_call[call_id]
                for call_id in call_ids
                if reflection_by_call.get(call_id)
            ],
            **deepcopy(interpretation),
        },
        "learning": deepcopy(learning),
    }
    if run_status is not None:
        payload["run_status"] = run_status
        payload["official_score_eligible"] = run_status == "complete_run"
    experience_id = f"exp-{_hash_json(payload)[:16]}"
    payload["experience_id"] = experience_id
    return payload


def _robot_state(result: object, boundary: str) -> object:
    if not isinstance(result, dict):
        return None
    memory = result.get("working_memory")
    execution = memory.get("execution") if isinstance(memory, dict) else None
    value = execution.get(boundary) if isinstance(execution, dict) else None
    if isinstance(value, dict):
        return deepcopy(value.get("robot_state"))
    observation = result.get("observation")
    return deepcopy(observation.get("robot_state")) if isinstance(observation, dict) else None


def _observer_claims(entries: list[dict[str, object]]) -> list[str]:
    claims: list[str] = []
    for entry in entries:
        result = entry.get("result")
        memory = result.get("working_memory") if isinstance(result, dict) else None
        visual = memory.get("visual") if isinstance(memory, dict) else None
        observations = visual.get("observations") if isinstance(visual, dict) else None
        if isinstance(observations, list):
            claims.extend(str(item) for item in observations if str(item).strip())
    return claims


def _reviewed_span(
    tool_trace: list[dict[str, object]],
    reviewed: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> list[dict[str, object]]:
    prefix = f"{manifest_path}:{line_number}"
    start = str(reviewed.get("start_call_id", "")).strip()
    end = str(reviewed.get("end_call_id", "")).strip()
    indexes = {
        str(entry.get("call_id")): index
        for index, entry in enumerate(tool_trace)
        if entry.get("call_id")
    }
    if start not in indexes or end not in indexes or indexes[start] > indexes[end]:
        raise ValueError(f"{prefix}: invalid reviewed call span")
    return tool_trace[indexes[start] : indexes[end] + 1]


def _reviewed_interpretation(
    reviewed: dict[str, Any], manifest_path: Path, line_number: int
) -> dict[str, object]:
    value = reviewed.get("interpretation")
    prefix = f"{manifest_path}:{line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: interpretation must be an object")
    domain = str(value.get("failure_domain", "")).strip()
    confidence = str(value.get("confidence", "")).strip()
    summary = str(value.get("maintainer_summary", "")).strip()
    if domain not in _FAILURE_DOMAINS or confidence not in _CONFIDENCE_LEVELS or not summary:
        raise ValueError(f"{prefix}: invalid reviewed interpretation")
    return {
        "failure_domain": domain,
        "confidence": confidence,
        "maintainer_summary": summary,
    }


def _reviewed_learning(
    reviewed: dict[str, Any], manifest_path: Path, line_number: int
) -> dict[str, object]:
    value = reviewed.get("learning")
    prefix = f"{manifest_path}:{line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: learning must be an object")
    status = str(value.get("learning_status", "")).strip()
    eligible = value.get("skill_eligible")
    reasons = value.get("exclusion_reasons", [])
    if status not in _LEARNING_STATUSES or not isinstance(eligible, bool):
        raise ValueError(f"{prefix}: invalid reviewed learning metadata")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError(f"{prefix}: exclusion_reasons must be an array of strings")
    if status != "active" and eligible:
        raise ValueError(f"{prefix}: non-active Experience cannot be Skill eligible")
    return {
        "learning_status": status,
        "skill_eligible": eligible,
        "exclusion_reasons": reasons,
    }


def _select_batch_runs(
    manifest_path: Path,
) -> tuple[str, list[dict[str, object]]]:
    """每个任务优先取最后一次正常运行，否则取最后一次实质性中断。"""
    complete: dict[int, dict[str, object]] = {}
    incomplete: dict[int, dict[str, object]] = {}
    batch_ids: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = _object(json.loads(line), f"{manifest_path}:{line_number}")
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest_path}:{line_number}: {error}") from error
        batch_id = str(row.get("batch_id", "")).strip()
        task_id = _integer(row.get("task_id"))
        seed = _integer(row.get("seed"))
        run_dir = str(row.get("run_dir", "")).strip()
        if not batch_id or task_id is None or seed is None or not run_dir:
            raise ValueError(
                f"{manifest_path}:{line_number}: missing batch_id, task_id, seed, or run_dir"
            )
        batch_ids.add(batch_id)
        if row.get("artifact_complete") is not True:
            continue
        record = {
            "task_id": task_id,
            "seed": seed,
            "run_dir": run_dir,
        }
        if row.get("exit_code") == 0:
            complete[task_id] = {
                **record,
                "run_status": "complete_run",
            }
        elif _is_material_incomplete_run(run_dir, manifest_path):
            incomplete[task_id] = {
                **record,
                "run_status": "incomplete_run",
            }
    if len(batch_ids) != 1:
        raise ValueError("batch manifest must contain exactly one batch_id")
    selected = {
        task_id: incomplete[task_id]
        for task_id in incomplete
        if task_id not in complete
    }
    selected.update(complete)
    if not selected:
        raise ValueError("batch manifest contains no complete or material incomplete runs")
    return next(iter(batch_ids)), [selected[key] for key in sorted(selected)]


def _is_material_incomplete_run(run_dir: str, manifest_path: Path) -> bool:
    """仅把已经产生决策的异常 Episode 视为可保留的局部经验。"""
    try:
        source = RoboMemArenaTrace.load(_resolve_run_dir(run_dir, manifest_path))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return source.episode.get("status") == "error" and bool(source.tool_trace)


def _resolve_run_dir(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), *manifest_path.parents):
        candidate = (base / path).resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"batch run directory does not exist: {value}")


def _policy_context(
    run_config: dict[str, object],
    *,
    cohort_id: str,
    policy_role: str,
) -> dict[str, object]:
    skill_path = str(run_config.get("task_execution_skill") or "").strip()
    skill_version = str(run_config.get("task_execution_skill_version") or "").strip()
    skill_hash = str(run_config.get("task_execution_skill_sha256") or "").strip()
    loaded_skills: list[dict[str, object]] = []
    if skill_path or skill_version or skill_hash:
        if not skill_version or not skill_hash:
            raise ValueError("loaded Skill provenance requires version and content hash")
        loaded_skills.append(
            {
                "version": skill_version,
                "content_sha256": skill_hash,
                "source_path": skill_path or None,
            }
        )
    return {
        "cohort_id": cohort_id,
        "policy_role": policy_role,
        "loaded_skills": loaded_skills,
    }


def _cohort_provenance_by_episode(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted((root / "experiences/cohorts").glob("*.json")):
        cohort = _read_object(path)
        shared = {
            "cohort_id": cohort.get("cohort_id"),
            "policy_role": cohort.get("policy_role"),
            "loaded_skills": deepcopy(cohort.get("loaded_skills", [])),
        }
        for run in cohort.get("runs", []):
            if not isinstance(run, dict):
                continue
            episode_id = str(run.get("episode_id", "")).strip()
            if episode_id:
                episode = _read_object(
                    root / "experiences/episodes" / f"{episode_id}.json"
                )
                result[episode_id] = {
                    "episode_id": episode_id,
                    **deepcopy(shared),
                    "provenance": deepcopy(episode.get("provenance", {})),
                }
    return result


def _validate_reviewed_pair(
    value: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> None:
    prefix = f"{manifest_path}:{line_number}"
    directions = {"positive", "negative", "mixed", "retained"}
    if value.get("transfer_direction") not in directions:
        raise ValueError(
            f"{prefix}: transfer_direction must be positive, negative, mixed, or retained"
        )
    for key in ("baseline_action", "candidate_action"):
        action = value.get(key)
        if not isinstance(action, dict) or not action.get("call_id") or not action.get("tool"):
            raise ValueError(f"{prefix}: {key} requires call_id and tool")
    for key in ("divergence_reason", "immediate_effect", "later_consequence"):
        if not str(value.get(key, "")).strip():
            raise ValueError(f"{prefix}: {key} must be non-empty")
    attribution = value.get("skill_attribution")
    if (
        not isinstance(attribution, dict)
        or attribution.get("verdict") not in {"supported", "unsupported", "uncertain"}
        or not str(attribution.get("reason", "")).strip()
    ):
        raise ValueError(f"{prefix}: invalid skill_attribution")
    trace = value.get("distilled_execution_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{prefix}: distilled_execution_trace must be non-empty")


def _trace_call_ids(path: Path) -> set[str]:
    trace = _read_object(path)
    episode = trace.get("episode")
    details = episode.get("details") if isinstance(episode, dict) else None
    tool_trace = details.get("tool_trace") if isinstance(details, dict) else None
    if not isinstance(tool_trace, list):
        raise ValueError(f"{path}: missing episode.details.tool_trace")
    return {
        str(item.get("call_id"))
        for item in tool_trace
        if isinstance(item, dict) and item.get("call_id")
    }


def _units_by_episode(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted((root / "experiences/units").glob("*.json")):
        unit = _read_object(path)
        episode_id = str(unit.get("episode_id", "")).strip()
        experience_id = str(unit.get("experience_id", "")).strip()
        if episode_id and experience_id:
            result.setdefault(episode_id, []).append(experience_id)
    return result


def _physical_action_ranges(
    tool_trace: list[dict[str, object]],
) -> dict[str, tuple[int, int]]:
    cursor = 0
    ranges: dict[str, tuple[int, int]] = {}
    for entry in tool_trace:
        if entry.get("name") not in _PHYSICAL_TOOLS:
            continue
        result = entry.get("result")
        steps = _integer(result.get("executed_steps")) if isinstance(result, dict) else None
        if steps is None or steps < 0:
            continue
        call_id = str(entry.get("call_id", ""))
        ranges[call_id] = (cursor + 1, cursor + steps)
        cursor += steps
    return ranges


def _events_for_call(
    stage_events: list[dict[str, object]],
    action_ranges: dict[str, tuple[int, int]],
    call_id: str | None,
) -> list[dict[str, object]]:
    bounds = action_ranges.get(call_id or "")
    if bounds is None:
        return []
    start, end = bounds
    return [
        dict(event)
        for event in stage_events
        if (step := _integer(event.get("environment_step"))) is not None
        and start <= step <= end
    ]


def _result_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    fields = list(_RESULT_FIELDS)
    if value.get("status") == "rejected":
        fields.extend(_REJECTION_RESULT_FIELDS)
    return {key: deepcopy(value[key]) for key in fields if key in value}


def _stage_events(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("evaluation.stage_events must be an array")
    events: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("stage event must be an object")
        stage = str(item.get("stage", "")).strip()
        step = _integer(item.get("environment_step"))
        if not stage or step is None or step < 1:
            raise ValueError("stage event requires stage and positive environment_step")
        events.append({"stage": stage, "environment_step": step})
    return events


def _write_units_index(root: Path) -> None:
    path = root / "experiences/units.jsonl"
    records = [
        _read_object(item) for item in sorted((root / "experiences/units").glob("*.json"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def _write_immutable(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"immutable experience conflicts with existing file: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def _hash_json(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
def _integer(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "compile_batch",
    "compile_paired_comparisons",
    "compile_reviewed_events",
    "compile_run",
]
