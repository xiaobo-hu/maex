"""LIBERO benchmark task and evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import DEFAULT_MAX_STEPS


@dataclass(frozen=True, slots=True)
class LiberoTaskSpec:
    """单个 LIBERO 任务的定位信息。"""

    suite: str = "libero_spatial"
    task_id: int = 0
    seed: int = 0
    init_state_id: int = 0


@dataclass(frozen=True, slots=True)
class LiberoEvalConfig:
    """一次 LIBERO episode 的运行参数。"""

    task: LiberoTaskSpec = LiberoTaskSpec()
    max_steps: int | None = None
    warmup_steps: int = 10

    def resolved_max_steps(self) -> int:
        """返回显式配置或当前 suite 的默认步数上限。"""
        if self.max_steps is not None:
            return self.max_steps
        try:
            return DEFAULT_MAX_STEPS[self.task.suite]
        except KeyError as exc:
            raise ValueError(f"Unknown LIBERO suite: {self.task.suite}") from exc
