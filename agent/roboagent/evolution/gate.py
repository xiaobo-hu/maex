"""比较真实闭环 rollout，并管理 Candidate Skill 的审计与晋升。"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..skills import load_task_execution_skill
from .workspace import (
    candidate_has_formal_patterns,
    pattern_source_run_dirs,
    validate_workspace,
)

_MATCHED_CONFIG_FIELDS = (
    "task",
    "seed",
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
    "runtime_revision",
    "evaluator_revision",
    "data_role",
)


def compare_rollouts(
    workspace: str | Path,
    candidate_dir: str | Path,
    manifest_path: str | Path,
    *,
    phase: str = "candidate-gate",
) -> dict[str, object]:
    """比较相同 seed 的独立 Candidate Gate 或 Final Evaluation rollout。"""
    if phase not in {"candidate-gate", "final-evaluation"}:
        raise ValueError("phase must be candidate-gate or final-evaluation")
    root = Path(workspace).expanduser().resolve()
    _require_valid_workspace(root)
    candidate_path = Path(candidate_dir).expanduser().resolve()
    candidate = load_task_execution_skill(candidate_path / "SKILL.md")
    candidate_manifest = _read_object(candidate_path / "manifest.json")
    version = str(candidate_manifest.get("version", "")).strip()
    replay_report = _read_object(root / "runs" / version / "replay.json")
    required_status = "replay_approved" if phase == "candidate-gate" else "gate_approved"
    if replay_report.get("passed") is not True or candidate_manifest.get("status") != required_status:
        raise ValueError(f"candidate is not ready for {phase}")
    if not candidate_has_formal_patterns(candidate_manifest):
        raise ValueError("formal rollout Gate requires probable or verified Wiki patterns")
    if phase == "final-evaluation":
        gate_report = _read_object(root / "runs" / version / "candidate-gate.json")
        if gate_report.get("passed") is not True:
            raise ValueError("candidate must pass candidate-gate before final evaluation")
    baseline_skill_hash = _parent_skill_hash(root, candidate_manifest)
    manifest = _read_object(Path(manifest_path).expanduser().resolve())
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("closed-loop manifest requires a non-empty pairs array")

    source_run_dirs = pattern_source_run_dirs(root, candidate_manifest)
    expected_role = "candidate_gate" if phase == "candidate-gate" else "final_evaluation"
    prior_run_dirs: set[Path] = set()
    if phase == "final-evaluation":
        for row in gate_report.get("pairs", []):
            if not isinstance(row, dict):
                continue
            for key in ("baseline_run", "candidate_run"):
                value = str(row.get(key, "")).strip()
                if value:
                    prior_run_dirs.add(Path(value).expanduser().resolve())
    rows: list[dict[str, object]] = []
    baseline_tsr = 0
    candidate_tsr = 0
    baseline_csr = 0.0
    candidate_csr = 0.0
    regressions: list[str] = []
    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"closed-loop pair {index} must be an object")
        baseline_run = Path(str(pair.get("baseline_run", ""))).expanduser().resolve()
        candidate_run = Path(str(pair.get("candidate_run", ""))).expanduser().resolve()
        baseline_config = _read_object(baseline_run / "run_config.json")
        candidate_config = _read_object(candidate_run / "run_config.json")
        _require_matching_configs(baseline_config, candidate_config, index)
        if candidate_config.get("data_role") != expected_role:
            raise ValueError(f"pair {index} must use data_role={expected_role}")
        if baseline_config.get("task_execution_skill_sha256") != baseline_skill_hash:
            raise ValueError(f"pair {index} baseline run did not use the Active Skill")
        task_id = _int_value(candidate_config.get("task"), f"pair {index} task")
        seed = _int_value(candidate_config.get("seed"), f"pair {index} seed")
        if baseline_run in source_run_dirs or candidate_run in source_run_dirs:
            raise ValueError(
                f"pair {index} reuses a Candidate source Episode run"
            )
        if baseline_run in prior_run_dirs or candidate_run in prior_run_dirs:
            raise ValueError(f"pair {index} reuses a Candidate Gate run")
        if candidate_config.get("task_execution_skill_sha256") != candidate.content_sha256:
            raise ValueError(f"pair {index} candidate run did not use the candidate Skill")

        baseline_eval = _evaluation(baseline_run)
        candidate_eval = _evaluation(candidate_run)
        base_tsr = _metric_int(baseline_eval.get("TSR"))
        cand_tsr = _metric_int(candidate_eval.get("TSR"))
        base_csr = _metric_float(baseline_eval.get("CSR"))
        cand_csr = _metric_float(candidate_eval.get("CSR"))
        baseline_tsr += base_tsr
        candidate_tsr += cand_tsr
        baseline_csr += base_csr
        candidate_csr += cand_csr
        pair_id = str(pair.get("pair_id") or f"task{task_id}-seed{seed}-pair{index}")
        if base_tsr == 1 and cand_tsr == 0:
            regressions.append(pair_id)
        rows.append(
            {
                "pair_id": pair_id,
                "baseline_run": str(baseline_run),
                "candidate_run": str(candidate_run),
                "baseline": {"TSR": base_tsr, "CSR": base_csr},
                "candidate": {"TSR": cand_tsr, "CSR": cand_csr},
            }
        )
    count = len(rows)
    baseline_score = (baseline_tsr, baseline_csr / count)
    candidate_score = (candidate_tsr, candidate_csr / count)
    passed = (
        candidate_score > baseline_score
        if phase == "candidate-gate"
        else candidate_score >= baseline_score
    ) and not regressions
    return {
        "schema_version": 3,
        "phase": phase,
        "candidate": candidate.summary(),
        "pair_count": count,
        "baseline_score": {"sum_TSR": baseline_score[0], "mean_CSR": baseline_score[1]},
        "candidate_score": {"sum_TSR": candidate_score[0], "mean_CSR": candidate_score[1]},
        "regressions": regressions,
        "passed": passed,
        "pairs": rows,
    }


def record_gate_report(
    workspace: str | Path,
    candidate_dir: str | Path,
    gate_name: str,
    report: dict[str, object],
) -> Path:
    """保存 Gate 报告，并为拒绝 proposal 保留不可变历史副本。"""
    if gate_name not in {"replay", "candidate-gate", "final-evaluation"}:
        raise ValueError("invalid gate_name")
    root = Path(workspace).expanduser().resolve()
    _require_valid_workspace(root)
    candidate_path = Path(candidate_dir).expanduser().resolve()
    manifest_path = candidate_path / "manifest.json"
    manifest = _read_object(manifest_path)
    version = str(manifest.get("version", "")).strip()
    if not version:
        raise ValueError("candidate manifest is missing version")
    report_path = root / "runs" / version / f"{gate_name}.json"
    _write_json(report_path, report)
    passed = report.get("passed") is True
    if gate_name == "replay":
        status = "replay_approved" if passed else "rejected"
    elif gate_name == "candidate-gate":
        status = "gate_approved" if passed else "rejected"
    else:
        status = "validated" if passed else "rejected"
    manifest["status"] = status
    _write_json(manifest_path, manifest)
    _append_impact(root, candidate_path, gate_name.upper(), "PASS" if passed else "REJECT")
    if not passed:
        _archive(candidate_path, root / "skills/history" / version)
    return report_path


def promote_candidate(
    workspace: str | Path,
    candidate_dir: str | Path,
) -> dict[str, object]:
    """Replay 和至少两次 Candidate rollout 通过 Gate 后替换 Active Skill。"""
    root = Path(workspace).expanduser().resolve()
    _require_valid_workspace(root)
    candidate_path = Path(candidate_dir).expanduser().resolve()
    candidate = load_task_execution_skill(candidate_path / "SKILL.md")
    replay = _read_object(root / "runs" / candidate.version / "replay.json")
    candidate_gate = _read_object(root / "runs" / candidate.version / "candidate-gate.json")
    if replay.get("passed") is not True or candidate_gate.get("passed") is not True:
        raise ValueError("candidate must pass replay and candidate-gate")
    candidate_runs = {
        str(pair.get("candidate_run", "")).strip()
        for pair in candidate_gate.get("pairs", [])
        if isinstance(pair, dict) and str(pair.get("candidate_run", "")).strip()
    }
    if len(candidate_runs) < 2:
        raise ValueError("candidate-gate must contain two distinct candidate rollouts")
    manifest = _read_object(candidate_path / "manifest.json")
    if manifest.get("status") not in {"gate_approved", "validated"}:
        raise ValueError("candidate manifest must have status=gate_approved or validated")
    if not candidate_has_formal_patterns(manifest):
        raise ValueError("single-shot Wiki patterns cannot be promoted")

    active = root / "skills/active"
    skills_root = active.parent
    skills_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".active-next-", dir=skills_root))
    try:
        for name in ("SKILL.md", "PURPOSE.md", "manifest.json"):
            shutil.copy2(candidate_path / name, staging / name)
        active_manifest = _read_object(staging / "manifest.json")
        active_manifest["status"] = "active"
        _write_json(staging / "manifest.json", active_manifest)

        if (active / "SKILL.md").is_file():
            previous = load_task_execution_skill(active / "SKILL.md")
            _archive(active, root / "skills/history" / previous.version)
        backup = skills_root / ".active-previous"
        if backup.exists():
            shutil.rmtree(backup)
        if active.exists():
            os.replace(active, backup)
        os.replace(staging, active)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    _append_impact(root, candidate_path, "PROMOTION", "ACCEPT")
    return dict(candidate.summary())


def _parent_skill_hash(root: Path, manifest: dict[str, object]) -> str | None:
    version = str(manifest.get("parent_version") or "").strip()
    if not version:
        return None
    for path in (
        root / "skills/candidates" / version / "SKILL.md",
        root / "skills/history" / version / "SKILL.md",
        root / "skills/active/SKILL.md",
    ):
        if path.is_file():
            skill = load_task_execution_skill(path)
            if skill.version == version:
                return skill.content_sha256
    raise ValueError(f"unknown parent Skill {version}")


def _require_valid_workspace(root: Path) -> None:
    errors = validate_workspace(root)
    if errors:
        raise ValueError("invalid evolution workspace: " + "; ".join(errors))


def _require_matching_configs(
    baseline: dict[str, object],
    candidate: dict[str, object],
    pair_index: int,
) -> None:
    mismatched = [field for field in _MATCHED_CONFIG_FIELDS if baseline.get(field) != candidate.get(field)]
    if mismatched:
        raise ValueError(f"closed-loop pair {pair_index} config mismatch: {', '.join(mismatched)}")


def _evaluation(run_dir: Path) -> dict[str, object]:
    path = run_dir / "evaluation.json"
    if path.is_file():
        return _read_object(path)
    trace = _read_object(run_dir / "trace.json")
    value = trace.get("evaluation")
    if not isinstance(value, dict):
        raise ValueError(f"{run_dir} has no evaluation object")
    return value


def _archive(source: Path, target: Path) -> None:
    if target.exists():
        existing = load_task_execution_skill(target / "SKILL.md")
        incoming = load_task_execution_skill(source / "SKILL.md")
        if existing.content_sha256 != incoming.content_sha256:
            raise ValueError(f"immutable Skill history conflicts at {target}")
        return
    shutil.copytree(source, target)


def _append_impact(root: Path, candidate_dir: Path, gate: str, outcome: str) -> None:
    candidate = load_task_execution_skill(candidate_dir / "SKILL.md")
    manifest = _read_object(candidate_dir / "manifest.json")
    source_patterns = [
        str(item.get("pattern_id"))
        for item in manifest.get("source_patterns", [])
        if isinstance(item, dict) and item.get("pattern_id")
    ]
    active_path = root / "skills/active/SKILL.md"
    active_text = active_path.read_text(encoding="utf-8") if active_path.is_file() else ""
    candidate_text = (candidate_dir / "SKILL.md").read_text(encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            active_text.splitlines(keepends=True),
            candidate_text.splitlines(keepends=True),
            fromfile="active/SKILL.md",
            tofile=f"candidate/{candidate.version}/SKILL.md",
        )
    )
    path = root / "wiki/skill-impact.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {candidate.version}: {gate} {outcome}\n\n"
            f"Skill: `{candidate.skill_id}`\n\n"
            f"Patterns: {', '.join(f'`{item}`' for item in source_patterns)}\n\n"
            f"```diff\n{diff}```\n"
        )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _int_value(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _metric_int(value: object) -> int:
    result = int(value) if isinstance(value, int | float | str) else 0
    return 1 if result > 0 else 0


def _metric_float(value: object) -> float:
    return float(value) if isinstance(value, int | float | str) else 0.0


__all__ = [
    "compare_rollouts",
    "promote_candidate",
    "record_gate_report",
]
