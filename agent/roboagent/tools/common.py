"""把观察和结构化领域结果转换为 Agent ToolResult。"""

from __future__ import annotations

import json
from collections.abc import Iterable

from agent_core import AgentToolResult
from ai import ImageContent, TextContent

from ..types import Observation


def result_with_observation(
    payload: dict[str, object],
    observation: Observation | None = None,
    *,
    before_observation: Iterable[TextContent | ImageContent] = (),
) -> AgentToolResult:
    """生成简短文本、结构化 details 和最新图像。"""
    if observation is not None:
        payload = {**payload, "observation": observation.summary()}
    content: list[TextContent | ImageContent] = [
        TextContent(text=json.dumps(payload, ensure_ascii=False, default=str))
    ]
    content.extend(before_observation)
    if observation is not None:
        content.extend(observation.llm_content())
    return AgentToolResult(content=content, details=payload)
