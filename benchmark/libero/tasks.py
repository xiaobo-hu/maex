"""LIBERO task-suite metadata without importing the simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .config import LiberoTaskSpec


@dataclass(frozen=True, slots=True)
class LiberoTask:
    """供 Agent 使用的语言任务描述。"""

    spec: LiberoTaskSpec
    instruction: str


def make_task(spec: LiberoTaskSpec, instruction: str) -> LiberoTask:
    """构造一个不依赖 LIBERO 安装的任务对象。"""
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    return LiberoTask(spec=spec, instruction=instruction.strip())
