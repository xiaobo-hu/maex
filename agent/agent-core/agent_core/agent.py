"""有状态 Agent 封装。

对应本地 Pi 0.83 的 ``packages/agent/src/agent.ts``，队列和生命周期组织参考
``pi_agent_core/agent.py``。本模块维护 transcript、运行状态、steering/follow-up
队列和订阅屏障，实际 Trace/Turn 执行委托给 :mod:`agent.agent_loop`。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from ai import (
    AssistantMessage,
    ImageContent,
    Message,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    Transport,
    UserMessage,
)

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCall,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventListener,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentState,
    BeforeToolCall,
    ConvertToLlm,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    QueueMode,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    TransformContext,
    TurnEndEvent,
)

TAppMessage = TypeVar("TAppMessage")


# ============================================================
# 构造配置与消息队列
# ============================================================


async def _default_convert_to_llm(
    messages: list[AgentMessage[TAppMessage]],
) -> list[Message]:
    return [
        message
        for message in messages
        if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
    ]


@dataclass(slots=True)
class AgentOptions(Generic[TAppMessage]):
    initial_state: AgentState[TAppMessage] | None = None
    stream_fn: StreamFn | None = None
    convert_to_llm: ConvertToLlm[TAppMessage] | None = None
    transform_context: TransformContext[TAppMessage] | None = None
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None
    before_tool_call: BeforeToolCall[TAppMessage] | None = None
    after_tool_call: AfterToolCall[TAppMessage] | None = None
    prepare_next_turn: (
        Callable[
            [asyncio.Event],
            AgentLoopTurnUpdate[TAppMessage]
            | None
            | Awaitable[AgentLoopTurnUpdate[TAppMessage] | None],
        ]
        | None
    ) = None
    prepare_next_turn_with_context: (
        Callable[
            [PrepareNextTurnContext[TAppMessage], asyncio.Event],
            AgentLoopTurnUpdate[TAppMessage]
            | None
            | Awaitable[AgentLoopTurnUpdate[TAppMessage] | None],
        ]
        | None
    ) = None
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext[TAppMessage]], bool | Awaitable[bool]] | None
    ) = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    tool_execution: ToolExecutionMode = "parallel"
    stream_options: SimpleStreamOptions | None = None
    session_id: str | None = None
    thinking_budgets: dict[str, int] | None = None
    transport: Transport | None = None
    max_retry_delay_ms: int | None = None


class _PendingMessageQueue(Generic[TAppMessage]):
    """支持 all / one-at-a-time 两种消费方式的 FIFO 队列。"""

    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._messages: list[AgentMessage[TAppMessage]] = []

    def enqueue(self, message: AgentMessage[TAppMessage]) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def drain(self) -> list[AgentMessage[TAppMessage]]:
        if self.mode == "all":
            messages = list(self._messages)
            self._messages.clear()
            return messages
        if not self._messages:
            return []
        return [self._messages.pop(0)]

    def clear(self) -> None:
        self._messages.clear()


@dataclass(slots=True)
class _ActiveRun:
    done: asyncio.Future[None]
    cancel_event: asyncio.Event


class Agent(Generic[TAppMessage]):
    """维护 transcript 和运行生命周期的通用 Agent。"""

    def __init__(self, options: AgentOptions[TAppMessage] | None = None) -> None:
        resolved = options or AgentOptions()
        initial = resolved.initial_state or AgentState()
        self._state = AgentState[TAppMessage](
            system_prompt=initial.system_prompt,
            model=initial.model.model_copy(deep=True),
            thinking_level=initial.thinking_level,
            tools=list(initial.tools),
            messages=list(initial.messages),
        )
        self.convert_to_llm = resolved.convert_to_llm or _default_convert_to_llm
        self.transform_context = resolved.transform_context
        self.stream_function = resolved.stream_fn or get_default_stream_fn()
        self.get_api_key = resolved.get_api_key
        self.before_tool_call = resolved.before_tool_call
        self.after_tool_call = resolved.after_tool_call
        self.prepare_next_turn = resolved.prepare_next_turn
        self.prepare_next_turn_with_context = resolved.prepare_next_turn_with_context
        self.should_stop_after_turn = resolved.should_stop_after_turn
        self.tool_execution = resolved.tool_execution
        self.stream_options = (
            resolved.stream_options.model_copy(deep=True)
            if resolved.stream_options is not None
            else SimpleStreamOptions()
        )
        self.session_id = resolved.session_id or self.stream_options.session_id
        self.thinking_budgets = resolved.thinking_budgets or self.stream_options.thinking_budgets
        self.transport: Transport = resolved.transport or self.stream_options.transport or "auto"
        self.max_retry_delay_ms = (
            resolved.max_retry_delay_ms
            if resolved.max_retry_delay_ms is not None
            else self.stream_options.max_retry_delay_ms
        )
        self._steering_queue = _PendingMessageQueue[TAppMessage](resolved.steering_mode)
        self._follow_up_queue = _PendingMessageQueue[TAppMessage](resolved.follow_up_mode)
        self._listeners: list[AgentEventListener[TAppMessage]] = []
        self._active_run: _ActiveRun | None = None

    @property
    def state(self) -> AgentState[TAppMessage]:
        return self._state

    @property
    def signal(self) -> asyncio.Event | None:
        return self._active_run.cancel_event if self._active_run else None

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def subscribe(self, listener: AgentEventListener[TAppMessage]) -> Callable[[], None]:
        """按订阅顺序广播事件；异步 listener 属于本次运行的完成屏障。"""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def steer(self, message: AgentMessage[TAppMessage]) -> None:
        """在当前 assistant turn 完成后、下一次模型调用前注入消息。"""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage[TAppMessage]) -> None:
        """在 Agent 原本将要停止时追加消息。"""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def abort(self) -> None:
        """设置当前运行共享的取消事件；Provider、hook 和工具可协作响应。"""
        if self._active_run is not None:
            self._active_run.cancel_event.set()

    async def wait_for_idle(self) -> None:
        """等待循环和所有已 await 的订阅者完全结束。"""
        active = self._active_run
        if active is not None:
            await asyncio.shield(active.done)

    def reset(self) -> None:
        """清空 transcript、运行派生状态以及两个待处理队列。"""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_all_queues()

    async def prompt(
        self,
        message: str | AgentMessage[TAppMessage] | list[AgentMessage[TAppMessage]],
        images: list[ImageContent] | None = None,
    ) -> None:
        """从字符串、单条应用消息或消息批次开始一次运行。"""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue "
                "messages, or wait for completion."
            )
        await self._run_prompt(self._normalize_input(message, images))

    async def continue_(self) -> None:
        """从 user/toolResult 尾部继续，或消费 assistant 尾部后的排队消息。"""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )
        if not self._state.messages:
            raise RuntimeError("No messages to continue from")
        if isinstance(self._state.messages[-1], AssistantMessage):
            steering = self._steering_queue.drain()
            if steering:
                await self._run_prompt(steering, skip_initial_steering=True)
                return
            follow_ups = self._follow_up_queue.drain()
            if follow_ups:
                await self._run_prompt(follow_ups)
                return
            raise RuntimeError("Cannot continue from message role: assistant")
        await self._run_continuation()

    def _normalize_input(
        self,
        message: str | AgentMessage[TAppMessage] | list[AgentMessage[TAppMessage]],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage[TAppMessage]]:
        if isinstance(message, list):
            if images:
                raise ValueError("Images can only accompany a string prompt")
            return list(message)
        if not isinstance(message, str):
            if images:
                raise ValueError("Images can only accompany a string prompt")
            return [message]
        content: list[TextContent | ImageContent] = [TextContent(text=message)]
        content.extend(images or [])
        return [UserMessage(content=content)]

    def _context_snapshot(self) -> AgentContext[TAppMessage]:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _loop_config(self, *, skip_initial_steering: bool = False) -> AgentLoopConfig[TAppMessage]:
        skip = skip_initial_steering

        async def steering() -> list[AgentMessage[TAppMessage]]:
            nonlocal skip
            if skip:
                skip = False
                return []
            return self._steering_queue.drain()

        async def follow_up() -> list[AgentMessage[TAppMessage]]:
            return self._follow_up_queue.drain()

        async def prepare(
            context: PrepareNextTurnContext[TAppMessage],
        ) -> AgentLoopTurnUpdate[TAppMessage] | None:
            signal = self.signal
            assert signal is not None
            if self.prepare_next_turn_with_context is not None:
                result = self.prepare_next_turn_with_context(context, signal)
            elif self.prepare_next_turn is not None:
                result = self.prepare_next_turn(signal)
            else:
                return None
            return await result if inspect.isawaitable(result) else result

        options = self.stream_options.model_copy(deep=True)
        options.session_id = self.session_id
        options.thinking_budgets = self.thinking_budgets
        options.transport = self.transport
        options.max_retry_delay_ms = self.max_retry_delay_ms
        options.reasoning = (
            None if self._state.thinking_level == "off" else self._state.thinking_level
        )
        return AgentLoopConfig(
            model=self._state.model.model_copy(deep=True),
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            should_stop_after_turn=self.should_stop_after_turn,
            prepare_next_turn=(
                prepare
                if self.prepare_next_turn is not None
                or self.prepare_next_turn_with_context is not None
                else None
            ),
            get_steering_messages=steering,
            get_follow_up_messages=follow_up,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            stream_options=options,
        )

    async def _run_prompt(
        self,
        prompts: list[AgentMessage[TAppMessage]],
        *,
        skip_initial_steering: bool = False,
    ) -> None:
        async def execute(cancel_event: asyncio.Event) -> None:
            await run_agent_loop(
                prompts,
                self._context_snapshot(),
                self._loop_config(skip_initial_steering=skip_initial_steering),
                self._process_event,
                self.stream_function,
                cancel_event,
            )

        await self._run_with_lifecycle(execute)

    async def _run_continuation(self) -> None:
        async def execute(cancel_event: asyncio.Event) -> None:
            await run_agent_loop_continue(
                self._context_snapshot(),
                self._loop_config(),
                self._process_event,
                self.stream_function,
                cancel_event,
            )

        await self._run_with_lifecycle(execute)

    async def _run_with_lifecycle(
        self, executor: Callable[[asyncio.Event], Awaitable[None]]
    ) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing")
        loop = asyncio.get_running_loop()
        active = _ActiveRun(done=loop.create_future(), cancel_event=asyncio.Event())
        self._active_run = active
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None
        try:
            await executor(active.cancel_event)
        except BaseException as error:
            await self._handle_run_failure(error, active.cancel_event.is_set())
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state.pending_tool_calls = set()
            if not active.done.done():
                active.done.set_result(None)
            self._active_run = None

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        failure = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
        )
        await self._process_event(MessageStartEvent(message=failure))
        await self._process_event(MessageEndEvent(message=failure))
        await self._process_event(TurnEndEvent(message=failure, tool_results=[]))
        await self._process_event(AgentEndEvent(messages=[failure]))

    async def _process_event(self, event: AgentEvent[TAppMessage]) -> None:
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._state.streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            self._state.pending_tool_calls = {
                *self._state.pending_tool_calls,
                event.tool_call_id,
            }
        elif isinstance(event, ToolExecutionEndEvent):
            self._state.pending_tool_calls = self._state.pending_tool_calls - {event.tool_call_id}
        elif isinstance(event, TurnEndEvent) and isinstance(event.message, AssistantMessage):
            if event.message.error_message:
                self._state.error_message = event.message.error_message
        elif isinstance(event, AgentEndEvent):
            self._state.streaming_message = None

        signal = self.signal
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in tuple(self._listeners):
            parameters = inspect.signature(listener).parameters
            callback = cast(Callable[..., object], listener)
            result = callback(event, signal) if len(parameters) >= 2 else callback(event)
            if inspect.isawaitable(result):
                await result


__all__ = ["Agent", "AgentOptions"]
