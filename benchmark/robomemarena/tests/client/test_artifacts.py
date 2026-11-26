from __future__ import annotations

import csv
import json

from benchmark.robomemarena.client.artifacts import write_official_artifacts


def test_write_official_artifacts_for_one_episode(tmp_path) -> None:
    video_dir = tmp_path / "videos"
    write_official_artifacts(
        tmp_path,
        task_id=3,
        seed=100,
        prompt="place cream then pudding",
        evaluation={
            "CSR": 50.0,
            "TSR": 0,
            "stage_done": {"cream": True, "pudding": False},
            "extra_pour_detected": False,
            "failure_reason": "incomplete_stage",
        },
        video_dir=video_dir,
    )

    with (tmp_path / "episodes.tsv").open(encoding="utf-8") as source:
        episode = next(csv.DictReader(source, delimiter="\t"))
    assert episode["task_id"] == "3"
    assert episode["TSR"] == "0.0"
    assert episode["CSR"] == "50.0"
    assert episode["failure_reason"] == "incomplete_stage"
    assert json.loads((tmp_path / "evaluation.json").read_text())["stage_done"] == {
        "cream": True,
        "pudding": False,
    }
    assert json.loads((tmp_path / "aggregate.json").read_text()) == {
        "num_tasks": 1,
        "num_episodes": 1,
        "seed_start": 100,
        "TSR": 0.0,
        "CSR": 50.0,
    }
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary[0]["episodes"][0]["seed"] == 100
    assert (tmp_path / "task_summary.tsv").is_file()
