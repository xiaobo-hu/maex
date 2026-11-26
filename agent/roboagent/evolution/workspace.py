"""初始化并验证 RoboAgent-Evolve 人工维护的工作区。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..skills import load_task_execution_skill

_DIRECTORIES = (
    "experiences/episodes",
    "experiences/units",
    "experiences/cohorts",
    "experiences/paired_comparisons",
    "experiences/reviewed_events",
    "wiki/patterns",
    "skills/active",
    "skills/candidates",
    "skills/history",
    "replay",
    "replay/case_sets",
    "runs",
)
_WIKI_FILES = {
    "wiki/index.md": "# Embodied Wiki Index\n",
    "wiki/log.md": "# Embodied Wiki Log\n",
    "wiki/skill-impact.md": "# Skill Impact Log\n",
    "wiki/PATTERN_TEMPLATE.md": """---
pattern_id: example-pattern
category: state_progress
maturity: single-shot
scope: global
---

# Pattern title

## Applicability

State when this pattern applies and when it does not.

## Observation

Describe the repeated cross-episode evidence.

## Candidate guidance

Describe a guidance hypothesis, not an active rule.

## Counterexamples

List evidence that limits or contradicts the pattern.

## Evidence

- experience="exp-example" trace="path/to/trace.json" call_id="call-example"
""",
}
_PATTERN_HEADINGS = (
    "## Applicability",
    "## Observation",
    "## Candidate guidance",
    "## Counterexamples",
    "## Evidence",
)
_EVIDENCE_RE = re.compile(
    r'^- experience="([^"]+)" trace="([^"]+)" call_id="([^"]+)"$'
)
_CATEGORIES = {
    "planning",
    "state_progress",
    "plan_transition",
    "execution_recovery",
    "termination",
}
_SCOPES = {"task_specific", "task_family", "global"}
_MATURITIES = {"single-shot", "probable", "verified", "retired"}
_DATA_ROLES = {"experience_collection", "candidate_gate", "final_evaluation"}
_PROVENANCE_FIELDS = (
    "runtime_revision",
    "evaluator_revision",
    "data_role",
    "planner_model",
    "observer_model",
    "seed",
    "task_id",
)


def initialize_workspace(workspace: str | Path) -> Path:
    """创建 V3 工作目录和不会覆盖人工内容的 Markdown 模板。"""
    root = Path(workspace).expanduser().resolve()
    marker = root / "workspace.json"
    if (
        root.exists()
        and any(root.iterdir())
        and (not marker.is_file() or _safe_object(marker).get("schema_version") != 3)
    ):
        raise ValueError(f"refusing to modify legacy/non-V3 workspace: {root}")
    for relative in _DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative, content in _WIKI_FILES.items():
        path = root / relative
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    cases = root / "replay/cases.jsonl"
    if not cases.exists():
        cases.write_text("", encoding="utf-8")
    if not marker.exists():
        marker.write_text('{"schema_version": 3}\n', encoding="utf-8")
    return root


def validate_workspace(workspace: str | Path) -> list[str]:
    """检查 Wiki evidence、Replay 来源和 Skill manifest 引用。"""
    root = Path(workspace).expanduser().resolve()
    errors: list[str] = []
    marker = _safe_object(root / "workspace.json")
    if marker.get("schema_version") != 3:
        errors.append(f"{root}: workspace.json must declare schema_version 3")
    episode_ids = {
        path.stem for path in (root / "experiences/episodes").glob("episode-*.json")
    }
    experience_ids = _experience_ids(root, episode_ids, errors)
    units = {
        value["experience_id"]: value
        for path in sorted((root / "experiences/units").glob("*.json"))
        if isinstance((value := _safe_object(path)), dict) and value.get("experience_id")
    }
    episodes = {
        path.stem: _safe_object(path)
        for path in sorted((root / "experiences/episodes").glob("*.json"))
    }
    for episode_id, episode in episodes.items():
        _validate_episode(root, episode_id, episode, errors)
    errors.extend(_validate_cohorts(root, experience_ids))

    pattern_metadata: dict[str, dict[str, str]] = {}
    for path in sorted((root / "wiki/patterns").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter = _frontmatter(text)
        pattern_id = frontmatter.get("pattern_id", "")
        if pattern_id != path.stem:
            errors.append(f"{path}: pattern_id must match filename")
        if frontmatter.get("category") not in _CATEGORIES:
            errors.append(f"{path}: invalid category")
        maturity = frontmatter.get("maturity")
        if maturity not in _MATURITIES:
            errors.append(f"{path}: invalid maturity")
        scope = frontmatter.get("scope")
        if scope is not None and scope not in _SCOPES:
            errors.append(f"{path}: invalid scope")
        for heading in _PATTERN_HEADINGS:
            if heading not in text:
                errors.append(f"{path}: missing {heading}")
        evidence_text = text.split("## Evidence", 1)[1] if "## Evidence" in text else ""
        evidence_lines = [
            line.strip() for line in evidence_text.splitlines() if line.startswith("- ")
        ]
        if not evidence_lines:
            errors.append(f"{path}: missing evidence records")
        referenced_units: list[dict[str, object]] = []
        for line in evidence_lines:
            match = _EVIDENCE_RE.fullmatch(line)
            if match is None:
                errors.append(f"{path}: invalid evidence line: {line}")
            elif match.group(1) not in experience_ids:
                errors.append(f"{path}: unknown experience: {match.group(1)}")
            else:
                unit = units[match.group(1)]
                referenced_units.append(unit)
                if not _evidence_tuple_matches(unit, match.group(2), match.group(3)):
                    errors.append(f"{path}: evidence tuple does not match {match.group(1)}")
        independent_episodes = _independent_episode_ids(referenced_units)
        eligible_episodes = {
            episode_id
            for episode_id in independent_episodes
            if _episode_is_skill_eligible(episode_id, referenced_units)
        }
        task_ids = {
            episodes.get(episode_id, {}).get("task_id")
            for episode_id in eligible_episodes
            if isinstance(episodes.get(episode_id), dict)
        }
        if maturity == "single-shot" and len(eligible_episodes) < 1:
            errors.append(f"{path}: single-shot requires one eligible Episode")
        if maturity == "probable" and len(eligible_episodes) < 2:
            errors.append(f"{path}: probable requires two eligible Episodes")
        if maturity == "verified" and len(task_ids) < 2:
            errors.append(f"{path}: verified requires two distinct tasks")
        pattern_metadata[pattern_id] = {
            "maturity": maturity or "",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    errors.extend(_validate_replay_sources(root, experience_ids))
    known_versions = {
        path.parent.name
        for path in (root / "skills/candidates").glob("*/SKILL.md")
    } | {
        path.parent.name for path in (root / "skills/history").glob("*/SKILL.md")
    }
    active_manifest = _optional_manifest(root / "skills/active/manifest.json", errors)
    if active_manifest:
        known_versions.add(str(active_manifest.get("version", "")))
    for directory in (root / "skills/active", root / "skills/candidates"):
        skill_files = [directory / "SKILL.md"] if directory.name == "active" else list(
            directory.glob("*/SKILL.md")
        )
        for skill_file in skill_files:
            if not skill_file.exists():
                continue
            try:
                load_task_execution_skill(skill_file)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{skill_file}: {error}")
                continue
            purpose = skill_file.parent / "PURPOSE.md"
            if not purpose.is_file() or not purpose.read_text(encoding="utf-8").strip():
                errors.append(f"{skill_file.parent}: missing non-empty PURPOSE.md")
            manifest = json.loads((skill_file.parent / "manifest.json").read_text())
            expected_statuses = (
                {"active"}
                if directory.name == "active"
                else {
                    "candidate",
                    "replay_approved",
                    "gate_approved",
                    "validated",
                    "rejected",
                }
            )
            if manifest.get("status") not in expected_statuses:
                errors.append(f"{skill_file.parent}: invalid status {manifest.get('status')}")
            if directory.name == "candidates":
                parent = str(manifest.get("parent_version") or "")
                version = str(manifest.get("version") or skill_file.parent.name)
                if parent == version:
                    errors.append(f"{skill_file.parent}: parent_version cannot reference itself")
                elif parent and parent not in known_versions:
                    errors.append(
                        f"{skill_file.parent}: unknown parent_version {parent}"
                    )
            source_patterns = manifest.get("source_patterns", [])
            if "source_experiences" in manifest:
                errors.append(f"{skill_file.parent}: Skill must not reference Experience directly")
            if not isinstance(source_patterns, list) or not source_patterns:
                errors.append(f"{skill_file.parent}: source_patterns must be non-empty")
                source_patterns = []
            if not str(manifest.get("change_summary", "")).strip():
                errors.append(f"{skill_file.parent}: missing change_summary")
            if not str(manifest.get("changed_section", "")).strip():
                errors.append(f"{skill_file.parent}: missing changed_section")
            for source in source_patterns:
                if not isinstance(source, dict):
                    errors.append(f"{skill_file.parent}: source pattern must be an object")
                    continue
                pattern_id = str(source.get("pattern_id", ""))
                metadata = pattern_metadata.get(pattern_id)
                if metadata is None:
                    errors.append(f"{skill_file.parent}: unknown Wiki pattern {pattern_id}")
                    continue
                if source.get("maturity") != metadata["maturity"]:
                    errors.append(f"{skill_file.parent}: stale maturity for {pattern_id}")
                if source.get("content_sha256") != metadata["content_sha256"]:
                    errors.append(f"{skill_file.parent}: stale content hash for {pattern_id}")
            _validate_single_section_change(root, skill_file, manifest, errors)
    return errors


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return {}
    raw = text[4 : text.index("\n---\n", 4)]
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def _safe_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _evidence_tuple_matches(
    unit: dict[str, object], trace: str, call_id: str
) -> bool:
    if unit.get("unit_kind") == "paired_comparison":
        evidence = unit.get("evidence")
        return isinstance(evidence, list) and any(
            isinstance(item, dict)
            and item.get("trace") == trace
            and item.get("call_id") == call_id
            for item in evidence
        )
    evidence = unit.get("evidence")
    span = unit.get("call_span")
    call_ids = span.get("call_ids") if isinstance(span, dict) else None
    return (
        isinstance(evidence, dict)
        and evidence.get("source_trace") == trace
        and isinstance(call_ids, list)
        and call_id in call_ids
    )


def _independent_episode_ids(units: list[dict[str, object]]) -> set[str]:
    result: set[str] = set()
    for unit in units:
        if unit.get("unit_kind") == "paired_comparison":
            sources = unit.get("source_episode_ids")
            if isinstance(sources, list):
                result.update(str(item) for item in sources if str(item).strip())
        else:
            episode_id = str(unit.get("episode_id", "")).strip()
            if episode_id:
                result.add(episode_id)
    return result


def _episode_is_skill_eligible(
    episode_id: str, units: list[dict[str, object]]
) -> bool:
    for unit in units:
        if unit.get("unit_kind") == "paired_comparison":
            sources = unit.get("source_episode_ids")
            learning = unit.get("learning")
            if (
                isinstance(sources, list)
                and episode_id in sources
                and isinstance(learning, dict)
                and learning.get("skill_eligible") is True
            ):
                return True
        elif unit.get("episode_id") == episode_id:
            learning = unit.get("learning")
            if isinstance(learning, dict) and learning.get("skill_eligible") is True:
                return True
    return False


def _validate_single_section_change(
    root: Path,
    skill_file: Path,
    manifest: dict[str, object],
    errors: list[str],
) -> None:
    parent_version = str(manifest.get("parent_version") or "").strip()
    if not parent_version:
        return
    parent_file = _find_skill(root, parent_version)
    if parent_file is None:
        return
    changed_section = str(manifest.get("changed_section", "")).strip()
    parent_sections = _skill_sections(parent_file.read_text(encoding="utf-8"))
    candidate_sections = _skill_sections(skill_file.read_text(encoding="utf-8"))
    changed = {
        section
        for section in parent_sections.keys() | candidate_sections.keys()
        if parent_sections.get(section) != candidate_sections.get(section)
    }
    if changed != {changed_section}:
        errors.append(
            f"{skill_file.parent}: Skill diff must change only section {changed_section!r}"
        )


def _find_skill(root: Path, version: str) -> Path | None:
    candidates = [
        root / "skills/candidates" / version / "SKILL.md",
        root / "skills/history" / version / "SKILL.md",
        root / "skills/active/SKILL.md",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.parent.name != "active":
            return path
        manifest = _safe_object(path.parent / "manifest.json")
        if manifest.get("version") == version:
            return path
    return None


def _skill_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"__preamble__": []}
    current = "__preamble__"
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        sections[current].append(line)
    return {key: "".join(value) for key, value in sections.items()}


def _optional_manifest(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(f"{path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path}: must contain an object")
        return {}
    return value


def _experience_ids(
    root: Path,
    episode_ids: set[str],
    errors: list[str],
) -> set[str]:
    ids: set[str] = set()
    for path in sorted((root / "experiences/units").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        experience_id = str(value.get("experience_id", "")) if isinstance(value, dict) else ""
        if not experience_id:
            errors.append(f"{path}: missing experience_id")
        else:
            ids.add(experience_id)
        if isinstance(value, dict) and value.get("unit_kind") == "paired_comparison":
            _validate_paired_comparison(path, value, episode_ids, errors)
        elif isinstance(value, dict):
            _validate_event(path, value, episode_ids, errors)
    return ids


def _validate_event(
    path: Path,
    value: dict[str, object],
    episode_ids: set[str],
    errors: list[str],
) -> None:
    if value.get("schema_version") != 3 or value.get("unit_kind") != "event":
        errors.append(f"{path}: Experience must be a schema-3 event")
    if value.get("event_type") not in {
        "successful_state_change",
        "failure_recovery",
        "collateral_effect",
        "decision_contradiction",
    }:
        errors.append(f"{path}: invalid event_type")
    if value.get("episode_id") not in episode_ids:
        errors.append(f"{path}: unknown episode {value.get('episode_id')}")
    provenance = value.get("provenance")
    if not _valid_provenance(provenance):
        errors.append(f"{path}: incomplete provenance")
    interpretation = value.get("interpretation")
    if not isinstance(interpretation, dict):
        errors.append(f"{path}: missing interpretation")
    else:
        domain = interpretation.get("failure_domain")
        if domain is not None and domain not in {
            "runtime_safety",
            "tool_capability",
            "vla_execution",
            "observation_reflection",
            "agent_decision",
            "skill_related",
            "unknown",
        }:
            errors.append(f"{path}: invalid failure_domain")
        if interpretation.get("confidence") not in {"low", "medium", "high"}:
            errors.append(f"{path}: invalid confidence")
    learning = value.get("learning")
    if not isinstance(learning, dict):
        errors.append(f"{path}: missing learning metadata")
    else:
        status = learning.get("learning_status")
        eligible = learning.get("skill_eligible")
        if status not in {"active", "quarantined", "obsolete_runtime_issue"}:
            errors.append(f"{path}: invalid learning_status")
        if not isinstance(eligible, bool):
            errors.append(f"{path}: skill_eligible must be boolean")
        elif status != "active" and eligible:
            errors.append(f"{path}: non-active Experience cannot be Skill eligible")


def _validate_episode(
    root: Path,
    episode_id: str,
    episode: dict[str, object],
    errors: list[str],
) -> None:
    path = root / "experiences/episodes" / f"{episode_id}.json"
    if episode.get("schema_version") != 3 or episode.get("episode_id") != episode_id:
        errors.append(f"{path}: Episode must use schema 3 and match its filename")
    provenance = episode.get("provenance")
    if not _valid_provenance(provenance):
        errors.append(f"{path}: incomplete provenance")
        return
    assert isinstance(provenance, dict)
    if provenance.get("task_id") != episode.get("task_id"):
        errors.append(f"{path}: provenance task_id does not match Episode")
    if provenance.get("seed") != episode.get("seed"):
        errors.append(f"{path}: provenance seed does not match Episode")


def _valid_provenance(value: object) -> bool:
    if not isinstance(value, dict) or value.get("data_role") not in _DATA_ROLES:
        return False
    return all(
        value.get(key) is not None and str(value.get(key)).strip()
        for key in _PROVENANCE_FIELDS
    )


def _validate_paired_comparison(
    path: Path,
    value: dict[str, object],
    episode_ids: set[str],
    errors: list[str],
) -> None:
    if value.get("derived") is not True or value.get("independent_evidence") is not False:
        errors.append(f"{path}: paired comparison must be derived and non-independent")
    sources = value.get("source_episode_ids")
    if (
        not isinstance(sources, list)
        or len(sources) != 2
        or len(set(str(item) for item in sources)) != 2
    ):
        errors.append(f"{path}: paired comparison requires two source_episode_ids")
        sources = []
    for episode_id in sources:
        if episode_id not in episode_ids:
            errors.append(f"{path}: unknown source episode {episode_id}")
    for key in ("baseline_action", "candidate_action"):
        action = value.get(key)
        if not isinstance(action, dict) or not action.get("call_id") or not action.get("tool"):
            errors.append(f"{path}: invalid {key}")
    attribution = value.get("skill_attribution")
    if (
        not isinstance(attribution, dict)
        or attribution.get("verdict") not in {"supported", "unsupported", "uncertain"}
        or not str(attribution.get("reason", "")).strip()
    ):
        errors.append(f"{path}: invalid skill_attribution")
    trace = value.get("distilled_execution_trace")
    if not isinstance(trace, list) or not trace:
        errors.append(f"{path}: distilled_execution_trace must be non-empty")


def _validate_replay_sources(root: Path, experience_ids: set[str]) -> list[str]:
    errors: list[str] = []
    case_files = [root / "replay/cases.jsonl"] + sorted(
        (root / "replay/case_sets").glob("*.jsonl")
    )
    for cases in case_files:
        if not cases.is_file():
            continue
        for number, line in enumerate(
            cases.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{cases}:{number}: {error}")
                continue
            sources = value.get("source_experiences", []) if isinstance(value, dict) else []
            for source in sources:
                if source not in experience_ids:
                    errors.append(f"{cases}:{number}: unknown experience {source}")
    return errors


def _validate_cohorts(root: Path, experience_ids: set[str]) -> list[str]:
    errors: list[str] = []
    episode_root = root / "experiences/episodes"
    for path in sorted((root / "experiences/cohorts").glob("*.json")):
        try:
            cohort = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        if not isinstance(cohort, dict):
            errors.append(f"{path}: must contain an object")
            continue
        cohort_id = str(cohort.get("cohort_id", "")).strip()
        if cohort_id != path.stem:
            errors.append(f"{path}: cohort_id must match filename")
        if cohort.get("policy_role") not in {"baseline", "candidate", "active"}:
            errors.append(f"{path}: invalid policy_role")
        runs = cohort.get("runs")
        if not isinstance(runs, list) or not runs:
            errors.append(f"{path}: runs must be a non-empty array")
            continue
        for run in runs:
            if not isinstance(run, dict):
                errors.append(f"{path}: run entry must be an object")
                continue
            episode_id = str(run.get("episode_id", "")).strip()
            episode_path = episode_root / f"{episode_id}.json"
            if not episode_id or not episode_path.is_file():
                errors.append(f"{path}: unknown episode {episode_id}")
            sources = run.get("experience_ids")
            if not isinstance(sources, list):
                errors.append(f"{path}: run {episode_id} experience_ids must be an array")
                continue
            for experience_id in sources:
                if experience_id not in experience_ids:
                    errors.append(f"{path}: unknown experience {experience_id}")
    return errors


def pattern_source_episode_pairs(
    workspace: str | Path, manifest: dict[str, object]
) -> set[tuple[int, int]]:
    """沿 Candidate 的 Wiki Evidence 返回去重后的来源 task/seed。"""
    root = Path(workspace).expanduser().resolve()
    pairs: set[tuple[int, int]] = set()
    sources = manifest.get("source_patterns", [])
    if not isinstance(sources, list):
        return pairs
    for source in sources:
        if not isinstance(source, dict):
            continue
        path = root / "wiki/patterns" / f"{source.get('pattern_id')}.md"
        if not path.is_file():
            continue
        matches = (
            _EVIDENCE_RE.fullmatch(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        for match in (item for item in matches if item is not None):
            unit = _safe_object(root / "experiences/units" / f"{match.group(1)}.json")
            for episode_id in _independent_episode_ids([unit]):
                if not _episode_is_skill_eligible(episode_id, [unit]):
                    continue
                episode = _safe_object(
                    root / "experiences/episodes" / f"{episode_id}.json"
                )
                task_id = episode.get("task_id")
                seed = episode.get("seed")
                if isinstance(task_id, int) and isinstance(seed, int):
                    pairs.add((task_id, seed))
    return pairs


def pattern_source_run_dirs(
    workspace: str | Path, manifest: dict[str, object]
) -> set[Path]:
    """沿 Candidate 的 Wiki Evidence 返回来源 Episode 的运行目录。"""
    root = Path(workspace).expanduser().resolve()
    run_dirs: set[Path] = set()
    sources = manifest.get("source_patterns", [])
    if not isinstance(sources, list):
        return run_dirs
    for source in sources:
        if not isinstance(source, dict):
            continue
        path = root / "wiki/patterns" / f"{source.get('pattern_id')}.md"
        if not path.is_file():
            continue
        matches = (
            _EVIDENCE_RE.fullmatch(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        for match in (item for item in matches if item is not None):
            unit = _safe_object(root / "experiences/units" / f"{match.group(1)}.json")
            for episode_id in _independent_episode_ids([unit]):
                if not _episode_is_skill_eligible(episode_id, [unit]):
                    continue
                episode = _safe_object(
                    root / "experiences/episodes" / f"{episode_id}.json"
                )
                source_trace = str(episode.get("source_trace", "")).strip()
                if source_trace:
                    run_dirs.add(Path(source_trace).expanduser().resolve().parent)
    return run_dirs


def candidate_has_formal_patterns(manifest: dict[str, object]) -> bool:
    """仅 probable/verified Pattern 可以进入正式 rollout Gate。"""
    sources = manifest.get("source_patterns", [])
    return isinstance(sources, list) and bool(sources) and all(
        isinstance(source, dict)
        and source.get("maturity") in {"probable", "verified"}
        for source in sources
    )


__all__ = [
    "candidate_has_formal_patterns",
    "initialize_workspace",
    "pattern_source_episode_pairs",
    "pattern_source_run_dirs",
    "validate_workspace",
]
