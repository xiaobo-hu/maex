"""读取 RoboMemArena 运行目录，并隔离 benchmark 特有的 trace 结构。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RoboMemArenaTrace:
    """Experience Compiler 所需的只读 RoboMemArena trace 视图。"""

    run_dir: Path
    trace_path: Path
    trace: dict[str, Any]
    run_config: dict[str, Any]
    episode: dict[str, Any]
    evaluation: dict[str, Any]
    tool_trace: list[dict[str, object]]
    state_reflections: list[dict[str, object]]
    episode_memory: list[dict[str, object]]

    @classmethod
    def load(cls, run_dir: str | Path) -> RoboMemArenaTrace:
        """从一个 RoboMemArena run directory 读取并校验编译输入。"""
        run_path = Path(run_dir).expanduser().resolve()
        trace_path = run_path / "trace.json"
        trace = _read_object(trace_path)
        run_config = _read_object(run_path / "run_config.json", optional=True)
        episode = _object(trace.get("episode"), "trace.episode")
        evaluation = _object(trace.get("evaluation", {}), "trace.evaluation")
        details = _object(episode.get("details", {}), "trace.episode.details")
        return cls(
            run_dir=run_path,
            trace_path=trace_path,
            trace=trace,
            run_config=run_config,
            episode=episode,
            evaluation=evaluation,
            tool_trace=_object_list(details.get("tool_trace", []), "tool_trace"),
            state_reflections=_object_list(
                details.get("state_reflections", []),
                "state_reflections",
            ),
            episode_memory=_object_list(details.get("episode_memory", []), "episode_memory"),
        )


def _read_object(path: Path, *, optional: bool = False) -> dict[str, Any]:
    if optional and not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _object_list(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be an array of objects")
    return [dict(item) for item in value]


__all__ = ["RoboMemArenaTrace"]
