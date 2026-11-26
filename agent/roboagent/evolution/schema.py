"""定义 RoboAgent-Evolve V3 Historical Replay 的稳定磁盘记录结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AllowedAction:
    """Replay case 中一个可接受的工具选择及必要参数。"""

    tool: str
    args_contains: dict[str, object]

    @classmethod
    def from_dict(cls, value: object) -> AllowedAction:
        """校验并读取一个允许动作。"""
        if not isinstance(value, dict):
            raise ValueError("allowed action must be an object")
        tool = str(value.get("tool", "")).strip()
        args = value.get("args_contains", {})
        if not tool or not isinstance(args, dict):
            raise ValueError("allowed action requires tool and object args_contains")
        return cls(tool=tool, args_contains=dict(args))


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """一个不重放视觉识别、只重放决策层的历史 case。"""

    case_id: str
    task: str
    task_plan: dict[str, object]
    episode_memory: list[dict[str, object]]
    latest_reflection: str
    allowed_actions: tuple[AllowedAction, ...]
    source_experiences: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> ReplayCase:
        """从 JSONL 记录读取并严格校验必需字段。"""
        if not isinstance(value, dict):
            raise ValueError("replay case must be an object")
        case_id = str(value.get("case_id", "")).strip()
        task = str(value.get("task", "")).strip()
        task_plan = value.get("task_plan")
        episode_memory = value.get("episode_memory")
        reflection = str(value.get("latest_reflection", "")).strip()
        raw_actions = value.get("allowed_actions")
        raw_sources = value.get("source_experiences")
        if not case_id or not task or not reflection:
            raise ValueError("replay case requires case_id, task, and latest_reflection")
        if not isinstance(task_plan, dict) or not isinstance(episode_memory, list):
            raise ValueError("replay case task_plan and episode_memory have invalid types")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("replay case requires at least one allowed action")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("replay case requires source_experiences")
        memories = [item for item in episode_memory if isinstance(item, dict)]
        if len(memories) != len(episode_memory):
            raise ValueError("episode_memory entries must be objects")
        return cls(
            case_id=case_id,
            task=task,
            task_plan=dict(task_plan),
            episode_memory=[dict(item) for item in memories],
            latest_reflection=reflection,
            allowed_actions=tuple(AllowedAction.from_dict(item) for item in raw_actions),
            source_experiences=tuple(str(item).strip() for item in raw_sources),
        )


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    """Planner 在一个 replay case 中返回的首个工具选择。"""

    tool: str
    arguments: dict[str, object]

    def summary(self) -> dict[str, Any]:
        """返回 JSON-safe 决策。"""
        return {"tool": self.tool, "arguments": self.arguments}


__all__ = ["AllowedAction", "ReplayCase", "ReplayDecision"]
