"""维护当前计划步骤的累计 VLA 执行次数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RecoveryState:
    """记录同一步骤生命周期内累计 VLA 次数，不参与工具路由。"""

    step_id: str | None = None
    consecutive_vla_attempts: int = 0

    def clear(self) -> None:
        """清除旧步骤的重试状态。"""
        self.step_id = None
        self.consecutive_vla_attempts = 0

    def attempts_for(self, step_id: str) -> int:
        """返回指定步骤从进入 running 状态起累计的 VLA 尝试次数。"""
        return self.consecutive_vla_attempts if self.step_id == step_id else 0

    def record_vla_attempt(self, step_id: str) -> int:
        """记录一次实际执行的 VLA 调用并返回本步骤累计次数。"""
        if self.step_id != step_id:
            self.step_id = step_id
            self.consecutive_vla_attempts = 0
        self.consecutive_vla_attempts += 1
        return self.consecutive_vla_attempts


__all__ = ["RecoveryState"]
