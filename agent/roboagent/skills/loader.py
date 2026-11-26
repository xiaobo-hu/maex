"""读取单一 Active Task-Execution Skill，并验证其内容身份。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskExecutionSkill:
    """一个可注入正常决策回合的已加载 Skill。"""

    skill_id: str
    version: str
    content: str
    content_sha256: str
    source_path: str

    def summary(self) -> dict[str, str]:
        """返回不重复 Skill 正文的审计元数据。"""
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "source_path": self.source_path,
        }


def apply_task_execution_skill(
    base_prompt: str,
    skill: TaskExecutionSkill | None,
) -> str:
    """只为正常决策提示词追加一个全局任务执行 Skill。"""
    if skill is None:
        return base_prompt
    return (
        f"{base_prompt.rstrip()}\n\n"
        "# Active Embodied Task-Execution Skill\n\n"
        f"Skill id: {skill.skill_id}\n"
        f"Skill version: {skill.version}\n\n"
        f"{skill.content}"
    )


def load_task_execution_skill(path: str | Path) -> TaskExecutionSkill:
    """从 ``SKILL.md`` 及同目录 ``manifest.json`` 加载并校验 Skill。"""
    skill_path = Path(path).expanduser().resolve()
    manifest_path = skill_path.parent / "manifest.json"
    if skill_path.name != "SKILL.md":
        raise ValueError("Task-execution skill path must point to SKILL.md")
    if not skill_path.is_file() or not manifest_path.is_file():
        raise ValueError("Task-execution skill requires SKILL.md and sibling manifest.json")

    content = skill_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("Task-execution SKILL.md must not be empty")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Task-execution skill manifest must be a JSON object")
    skill_id = str(manifest.get("skill_id", "")).strip()
    version = str(manifest.get("version", "")).strip()
    expected_hash = str(manifest.get("content_sha256", "")).strip()
    if not skill_id or not version or not expected_hash:
        raise ValueError("Skill manifest requires skill_id, version, and content_sha256")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if content_hash != expected_hash:
        raise ValueError("SKILL.md content does not match manifest content_sha256")
    return TaskExecutionSkill(
        skill_id=skill_id,
        version=version,
        content=content,
        content_sha256=content_hash,
        source_path=str(skill_path),
    )


__all__ = [
    "TaskExecutionSkill",
    "apply_task_execution_skill",
    "load_task_execution_skill",
]
