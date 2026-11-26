"""LIBERO environment boundary used by the benchmark adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LiberoStep:
    """统一 LIBERO ``step`` 返回值，屏蔽 robosuite 版本差异。"""

    observation: Mapping[str, object]
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: Mapping[str, object] | None = None


class LiberoEnv(Protocol):
    """LIBERO runtime 所需的最小环境接口。"""

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]: ...

    def step(self, action: Sequence[float]) -> LiberoStep: ...

    def close(self) -> None: ...
