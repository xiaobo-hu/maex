"""Agent 核心类型。

对应本地 Pi 0.83 的 ``packages/agent/src/types.ts``，Python Protocol/dataclass 的
组织方式参考 ``pi_agent_core/types.py``。核心层只声明可扩展消息、工具、循环配置和
生命周期事件，不拥有任何具体客户业务消息。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Context,
    ImageContent,
    Message,
    Model,
    ModelThinkingLevel,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)

TAppMessage = TypeVar("TAppMessage")
AgentMessage: TypeAlias = Message | TAppMessage
AgentToolCall: TypeAlias = ToolCall
ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]
_UNSET = object()


# ============================================================
# AgentTool 与工具 hooks
# ============================================================


@dataclass(slots=True)
class AgentToolResult:
    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: object | None = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool = False


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], None]


class AgentTool(Protocol):
    """应用工具协议；抛出异常表示执行失败，返回值表示成功。"""

    name: str
    label: str
    description: str
    parameters: dict[str, object]
    prepare_arguments: Callable[[dict[str, object]], dict[str, object]] | None
    execution_mode: ToolExecutionMode | None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        cancel_event: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult: ...


@dataclass(slots=True)
class AgentContext(Generic[TAppMessage]):
    """传给无状态 Agent Loop 的上下文快照。"""

    system_prompt: str = ""
    messages: list[AgentMessage[TAppMessage]] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)


StreamFn: TypeAlias = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]
ConvertToLlm: TypeAlias = Callable[
    [list[AgentMessage[TAppMessage]]],
    list[Message] | Awaitable[list[Message]],
]
TransformContext: TypeAlias = Callable[
    [list[AgentMessage[TAppMessage]], asyncio.Event | None],
    list[AgentMessage[TAppMessage]] | Awaitable[list[AgentMessage[TAppMessage]]],
]


@dataclass(frozen=True, slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass(slots=True)
class BeforeToolCallContext(Generic[TAppMessage]):
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: dict[str, object]
    context: AgentContext[TAppMessage]


@dataclass(slots=True)
class AfterToolCallContext(Generic[TAppMessage]):
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: dict[str, object]
    result: AgentToolResult
    is_error: bool
    context: AgentContext[TAppMessage]


@dataclass(slots=True)
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    details: object = _UNSET
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class ShouldStopAfterTurnContext(Generic[TAppMessage]):
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext[TAppMessage]
    new_messages: list[AgentMessage[TAppMessage]]


PrepareNextTurnContext: TypeAlias = ShouldStopAfterTurnContext[TAppMessage]


@dataclass(slots=True)
class AgentLoopTurnUpdate(Generic[TAppMessage]):
    context: AgentContext[TAppMessage] | None = None
    model: Model | None = None
    thinking_level: ModelThinkingLevel | None = None


BeforeToolCall: TypeAlias = Callable[
    [BeforeToolCallContext[TAppMessage], asyncio.Event | None],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolCall: TypeAlias = Callable[
    [AfterToolCallContext[TAppMessage], asyncio.Event | None],
    AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
]


# ============================================================
# Agent Loop 配置与公开状态
# ============================================================


@dataclass(slots=True)
class AgentLoopConfig(Generic[TAppMessage]):
    """一次低层循环的配置；所有 hook 均支持同步或异步返回值。"""

    model: Model
    convert_to_llm: ConvertToLlm[TAppMessage] | None = None
    transform_context: TransformContext[TAppMessage] | None = None
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext[TAppMessage]], bool | Awaitable[bool]] | None
    ) = None
    prepare_next_turn: (
        Callable[
            [PrepareNextTurnContext[TAppMessage]],
            AgentLoopTurnUpdate[TAppMessage]
            | None
            | Awaitable[AgentLoopTurnUpdate[TAppMessage] | None],
        ]
        | None
    ) = None
    get_steering_messages: (
        Callable[[], list[AgentMessage[TAppMessage]] | Awaitable[list[AgentMessage[TAppMessage]]]]
        | None
    ) = None
    get_follow_up_messages: (
        Callable[[], list[AgentMessage[TAppMessage]] | Awaitable[list[AgentMessage[TAppMessage]]]]
        | None
    ) = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCall[TAppMessage] | None = None
    after_tool_call: AfterToolCall[TAppMessage] | None = None
    stream_options: SimpleStreamOptions | None = None


def _unknown_model() -> Model:
    return Model(id="unknown", name="unknown", api="unknown", provider="unknown", input=[])


@dataclass(slots=True)
class AgentState(Generic[TAppMessage]):
    """有状态 Agent 的公开状态；数组和集合赋值会复制顶层容器。"""

    system_prompt: str = ""
    model: Model = field(default_factory=_unknown_model)
    thinking_level: ModelThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage[TAppMessage]] = field(default_factory=list)
    is_streaming: bool = False
    streaming_message: AgentMessage[TAppMessage] | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"tools", "messages"} and isinstance(value, list):
            value = list(value)
        elif name == "pending_tool_calls" and isinstance(value, set):
            value = set(value)
        object.__setattr__(self, name, value)


# ============================================================
# Agent/Turn/Message/Tool 四层生命周期事件
# ============================================================


@dataclass(frozen=True, slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEndEvent(Generic[TAppMessage]):
    messages: list[AgentMessage[TAppMessage]]
    type: Literal["agent_end"] = "agent_end"


@dataclass(frozen=True, slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEndEvent(Generic[TAppMessage]):
    message: AgentMessage[TAppMessage]
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True, slots=True)
class MessageStartEvent(Generic[TAppMessage]):
    message: AgentMessage[TAppMessage]
    type: Literal["message_start"] = "message_start"


@dataclass(frozen=True, slots=True)
class MessageUpdateEvent:
    message: AssistantMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(frozen=True, slots=True)
class MessageEndEvent(Generic[TAppMessage]):
    message: AgentMessage[TAppMessage]
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True, slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    partial_result: AgentToolResult
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(frozen=True, slots=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent[TAppMessage]
    | TurnStartEvent
    | TurnEndEvent[TAppMessage]
    | MessageStartEvent[TAppMessage]
    | MessageUpdateEvent
    | MessageEndEvent[TAppMessage]
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
AgentEventSink: TypeAlias = Callable[[AgentEvent[TAppMessage]], None | Awaitable[None]]
AgentEventListener: TypeAlias = Callable[
    [AgentEvent[TAppMessage], asyncio.Event], None | Awaitable[None]
]


__all__ = [
    "AfterToolCall",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventListener",
    "AgentEventSink",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentStartEvent",
    "AgentState",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "BeforeToolCall",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ConvertToLlm",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "PrepareNextTurnContext",
    "QueueMode",
    "ShouldStopAfterTurnContext",
    "StreamFn",
    "TAppMessage",
    "ToolExecutionEndEvent",
    "ToolExecutionMode",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TransformContext",
    "TurnEndEvent",
    "TurnStartEvent",
]
