"""从单次 RoboAgent episode 生成 RoboMemArena 官方风格评测产物。"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

_EPISODES_HEADER = [
    "task_id",
    "ep",
    "seed",
    "TSR",
    "CSR",
    "extra_pour_detected",
    "pour_1_step",
    "pour_2_step",
    "extra_monitor_end_step",
    "failure_reason",
    "prompt",
    "video_dir",
]
_TASK_SUMMARY_HEADER = [
    "task_id",
    "num_trials",
    "seed_start",
    "TSR",
    "CSR",
    "prompt",
    "video_dir",
]


def write_official_artifacts(
    run_dir: Path,
    *,
    task_id: int,
    seed: int,
    prompt: str,
    evaluation: Mapping[str, object],
    video_dir: Path,
) -> None:
    """Write one-episode TSV and JSON files using the official batch layout."""
    run_dir.mkdir(parents=True, exist_ok=True)
    csr = _metric(evaluation.get("CSR"))
    tsr = 100.0 if _metric(evaluation.get("TSR")) > 0 else 0.0
    episode = {
        "ep": 0,
        "seed": seed,
        "TSR": tsr,
        "CSR": csr,
        **dict(evaluation),
    }
    episode["TSR"] = tsr
    episode["CSR"] = csr
    result = {
        "task_id": task_id,
        "task_key": f"task{task_id}",
        "prompt": prompt,
        "video_dir": str(video_dir),
        "TSR": tsr,
        "CSR": csr,
        "episodes": [episode],
    }

    _write_tsv(
        run_dir / "episodes.tsv",
        _EPISODES_HEADER,
        [[
            task_id,
            0,
            seed,
            f"{tsr:.1f}",
            f"{csr:.1f}",
            "Y" if evaluation.get("extra_pour_detected", False) else "N",
            evaluation.get("pour_1_step"),
            evaluation.get("pour_2_step"),
            evaluation.get("extra_monitor_end_step"),
            evaluation.get("failure_reason"),
            prompt,
            str(video_dir),
        ]],
    )
    _write_tsv(
        run_dir / "task_summary.tsv",
        _TASK_SUMMARY_HEADER,
        [[task_id, 1, seed, f"{tsr:.1f}", f"{csr:.1f}", prompt, str(video_dir)]],
    )
    _write_json(run_dir / "evaluation.json", dict(evaluation))
    _write_json(run_dir / "summary.json", [result])
    _write_json(
        run_dir / "aggregate.json",
        {
            "num_tasks": 1,
            "num_episodes": 1,
            "seed_start": seed,
            "TSR": tsr,
            "CSR": csr,
        },
    )


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric(value: object) -> float:
    return float(value) if isinstance(value, (int, float, str)) else 0.0
