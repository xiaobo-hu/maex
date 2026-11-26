"""Provider-neutral AI contracts adapted from Pi 0.83 and encyc/pi-py."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Api: TypeAlias = str
ProviderId: TypeAlias = str
ThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
CacheRetention: TypeAlias = Literal["none", "short", "long"]
Transport: TypeAlias = Literal["sse", "websocket", "websocket-cached", "auto"]
StopReason: TypeAlias = Literal["pending", "stop", "length", "toolUse", "error", "aborted"]


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TextContent(_StrictModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(default=None, alias="textSignature")


class ThinkingContent(_StrictModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")
    redacted: bool = False


class ImageContent(_StrictModel):
    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType")


class ToolCall(_StrictModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    thought_signature: str | None = Field(default=None, alias="thoughtSignature")


class JsonSchemaConstrainedSampling(_StrictModel):
    type: Literal["json_schema"] = "json_schema"
    strict: Literal["require", "prefer"] = "prefer"


class GrammarVariants(_StrictModel):
    openai_lark: str | None = None
    openai_regex: str | None = None


class GrammarSamplingConfig(_StrictModel):
    type: Literal["grammar"] = "grammar"
    variants: GrammarVariants


ConstrainedSamplingConfig: TypeAlias = Annotated[
    JsonSchemaConstrainedSampling | GrammarSamplingConfig,
    Field(discriminator="type"),
]


AssistantContent: TypeAlias = Annotated[
    TextContent | ThinkingContent | ToolCall,
    Field(discriminator="type"),
]
UserContent: TypeAlias = Annotated[TextContent | ImageContent, Field(discriminator="type")]
ToolResultContent: TypeAlias = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]


class UsageCost(_StrictModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(_StrictModel):
    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    cache_write_1h: int | None = Field(default=None, alias="cacheWrite1h")
    reasoning: int | None = None
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: UsageCost = Field(default_factory=UsageCost)


class UserMessage(_StrictModel):
    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int = Field(default_factory=now_ms)


class AssistantMessage(_StrictModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: Api = ""
    provider: ProviderId = ""
    model: str = ""
    response_model: str | None = Field(default=None, alias="responseModel")
    response_id: str | None = Field(default=None, alias="responseId")
    diagnostics: list[dict[str, object]] | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = Field(default="pending", alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    raw_stop_reason: str | None = Field(default=None, alias="rawStopReason")
    timestamp: int = Field(default_factory=now_ms)


class ToolResultMessage(_StrictModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[ToolResultContent] = Field(default_factory=list)
    details: object | None = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = Field(default=None, alias="addedToolNames")
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int = Field(default_factory=now_ms)


Message: TypeAlias = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]


class Tool(_StrictModel):
    name: str
    description: str
    parameters: dict[str, object] = Field(default_factory=dict)
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = Field(
        default=None,
        alias="constrainedSampling",
    )

    def to_json_schema(self) -> dict[str, object]:
        schema = dict(self.parameters)
        schema.pop("title", None)
        return schema


class Context(_StrictModel):
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


class ModelCostRates(_StrictModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class ModelCostTier(ModelCostRates):
    input_tokens_above: int = Field(alias="inputTokensAbove")


class ModelCost(ModelCostRates):
    tiers: list[ModelCostTier] | None = None


def _default_model_inputs() -> list[Literal["text", "image"]]:
    return ["text"]


class Model(_StrictModel):
    id: str
    name: str
    api: Api
    provider: ProviderId
    base_url: str = Field(default="", alias="baseUrl")
    reasoning: bool = False
    thinking_level_map: dict[str, str | None] | None = Field(
        default=None,
        alias="thinkingLevelMap",
    )
    input: list[Literal["text", "image"]] = Field(default_factory=_default_model_inputs)
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = Field(default=0, alias="contextWindow")
    max_tokens: int = Field(default=0, alias="maxTokens")
    headers: dict[str, str] | None = None
    compat: dict[str, object] | None = None


class StreamOptions(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    api_key: str | None = Field(default=None, alias="apiKey")
    transport: Transport | None = None
    cache_retention: CacheRetention | None = Field(default=None, alias="cacheRetention")
    session_id: str | None = Field(default=None, alias="sessionId")
    headers: dict[str, str | None] | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    max_retries: int | None = Field(default=None, alias="maxRetries")
    max_retry_delay_ms: int | None = Field(default=None, alias="maxRetryDelayMs")
    metadata: dict[str, object] | None = None
    env: dict[str, str] | None = None
    http_client: object | None = Field(default=None, alias="httpClient", exclude=True)
    cancel_event: asyncio.Event | None = Field(default=None, alias="cancelEvent", exclude=True)
    extra_body: dict[str, object] | None = Field(default=None, alias="extraBody")
    tool_choice: object | None = Field(default=None, alias="toolChoice")
    reasoning_effort: ThinkingLevel | None = Field(default=None, alias="reasoningEffort")
    thinking_enabled: bool | None = Field(default=None, alias="thinkingEnabled")
    thinking_budget_tokens: int | None = Field(default=None, alias="thinkingBudgetTokens")
    effort: str | None = None
    thinking_display: Literal["summarized", "omitted"] | None = Field(
        default=None, alias="thinkingDisplay"
    )


class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None
    thinking_budgets: dict[str, int] | None = Field(default=None, alias="thinkingBudgets")


__all__ = [
    "Api",
    "AssistantContent",
    "AssistantMessage",
    "CacheRetention",
    "Context",
    "ConstrainedSamplingConfig",
    "GrammarSamplingConfig",
    "GrammarVariants",
    "ImageContent",
    "JsonSchemaConstrainedSampling",
    "Message",
    "Model",
    "ModelCost",
    "ModelCostRates",
    "ModelCostTier",
    "ModelThinkingLevel",
    "ProviderId",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "Tool",
    "ToolCall",
    "ToolResultContent",
    "ToolResultMessage",
    "Transport",
    "Usage",
    "UsageCost",
    "UserContent",
    "UserMessage",
    "now_ms",
]
