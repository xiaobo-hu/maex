"""Agent Harness 的最小上下文与会话基础设施。

这里只承载 Customer Agent 需要的内存会话和上下文压缩，不包含文件、Shell、skills
或分支导航。它对应 Pi harness 中与消息历史直接相关的部分。
"""

from .compaction import CompactionSettings, compact_messages, estimate_tokens, should_compact
from .messages import CompactionSummaryMessage, convert_to_llm
from .session import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    Session,
    SessionEntry,
    SessionError,
    SessionHeader,
    SessionStorage,
)

# 兼容旧的类型导入名称；新代码使用 Session + Storage。
InMemorySessionRepository = InMemorySessionStorage
SessionRepository = SessionStorage

__all__ = [
    "CompactionSettings",
    "CompactionSummaryMessage",
    "InMemorySessionRepository",
    "InMemorySessionStorage",
    "JsonlSessionStorage",
    "Session",
    "SessionEntry",
    "SessionError",
    "SessionHeader",
    "SessionStorage",
    "SessionRepository",
    "compact_messages",
    "convert_to_llm",
    "estimate_tokens",
    "should_compact",
]
