"""Load the fixed upstream RoboMemArena fork and official stage checker."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "cc156e519990ae43cf3b64281a548724f428fbbd"


def resolve_root() -> Path:
    """Resolve and validate the configured upstream checkout."""
    configured = os.environ.get("ROBOMEMARENA_ROOT")
    root = Path(configured).expanduser().resolve() if configured else _default_root()
    required = root / "evaluation_benchmark" / "scripts" / "task2_26_reference_stage.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"Invalid ROBOMEMARENA_ROOT={root}; expected fixed upstream commit "
            f"{EXPECTED_COMMIT} with {required.relative_to(root)}"
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify RoboMemArena checkout at {root}") from exc
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"RoboMemArena checkout must be {EXPECTED_COMMIT}, found {commit}"
        )
    return root


def load_official_modules() -> tuple[Any, Any]:
    """Import official environment helpers and stage checks from upstream."""
    root = resolve_root()
    scripts = root / "evaluation_benchmark" / "scripts"
    fork = root / "evaluation_benchmark" / "libero_fork"
    for path in (str(scripts), str(fork)):
        if path not in sys.path:
            sys.path.insert(0, path)
    eval_common = importlib.import_module("eval_common")
    stages = importlib.import_module("task2_26_reference_stage")
    return eval_common, stages


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".external" / "RoboMemArena"
