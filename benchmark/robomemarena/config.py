"""Validated RoboMemArena task and official evaluation defaults."""

from __future__ import annotations

from dataclasses import dataclass

ROBOMEMARENA_UPSTREAM_COMMIT = "cc156e519990ae43cf3b64281a548724f428fbbd"


@dataclass(frozen=True)
class RoboMemArenaConfig:
    """Configure one official model-agnostic RoboMemArena episode."""

    task_id: int = 1
    seed: int = 100
    warmup_steps: int = 10
    max_steps: int = 2500
    post_goal_steps: int = 200
    extra_pour_monitor_steps: int = 30
    camera_height: int = 480
    camera_width: int = 640
    enable_depth: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.task_id <= 26:
            raise ValueError("task_id must be in [1, 26]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("warmup_steps", "post_goal_steps", "extra_pour_monitor_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("max_steps", "camera_height", "camera_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_reset_params(self, seed: int | None = None) -> dict[str, object]:
        """Return the exact serializable reset configuration."""
        params = {
            "task_id": self.task_id,
            "seed": self.seed if seed is None else seed,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "post_goal_steps": self.post_goal_steps,
            "extra_pour_monitor_steps": self.extra_pour_monitor_steps,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
        }
        # 默认 RPC 保持原样；只有视觉工具客户端显式申请深度与标定。
        if self.enable_depth:
            params["enable_depth"] = True
        return params
