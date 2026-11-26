"""加载 RoboAgent、视觉 Observer 和状态反思回合使用的系统提示词。"""

from importlib.resources import files

DEFAULT_SYSTEM_PROMPT = (
    files(__package__).joinpath("system.md").read_text(encoding="utf-8").strip()
)

OBSERVER_SYSTEM_PROMPT = (
    files(__package__).joinpath("observer.md").read_text(encoding="utf-8").strip()
)

STATE_REFLECTION_PROMPT = (
    files(__package__).joinpath("state_reflection.md").read_text(encoding="utf-8").strip()
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "OBSERVER_SYSTEM_PROMPT",
    "STATE_REFLECTION_PROMPT",
]
