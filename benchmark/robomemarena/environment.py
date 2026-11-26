"""Minimal RoboMemArena environment contract shared by client adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RoboMemArenaStep:
    """Normalized result of exactly one RoboMemArena physics step."""

    observation: Mapping[str, object]
    reward: float = 0.0
    done: bool = False
    success: bool = False
    details: Mapping[str, object] = field(default_factory=dict)


class RoboMemArenaEnv(Protocol):
    """Environment operations exposed across the Python 3.8 boundary."""

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]: ...

    def observe(self) -> Mapping[str, object]: ...

    def step(self, action: Sequence[float]) -> RoboMemArenaStep: ...

    def return_to_home(self, gripper_command: float) -> RoboMemArenaStep: ...

    def iter_return_to_home(self, gripper_command: float) -> Iterator[RoboMemArenaStep]: ...

    def settle(self) -> RoboMemArenaStep: ...

    def metrics(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...
