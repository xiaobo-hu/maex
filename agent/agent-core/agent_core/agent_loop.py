"""无状态 Agent 循环引擎。

核心是双层循环：

- 外层：Agent 本来应该结束时，继续检查 follow-up 消息。
- 内层：处理模型工具调用以及 steering 消息。

重要约束：

- 循环内部始终保存 ``AgentMessage``，只在模型调用边界执行 ``convert_to_llm``。
- Assistant partial 使用“插入占位、增量替换、终态替换”维护上下文。
- ``stop_reason == "length"`` 时绝不执行可能被截断的工具参数。
- 并行工具先顺序完成 preflight，再并发执行；事件按完成顺序、结果按来源顺序。
- Provider 失败由 AI 流编码为 error/aborted 消息；编程错误继续抛给有状态 Agent 归约。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TypeVar, cast

from ai import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    EventStream,
    Message,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    validate_tool_arguments,
)

from .stream_fn import get_default_stream_fn
from .types import (
    _UNSET,
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

TAppMessage = TypeVar("TAppMessage")


# ============================================================
# 公共入口与 Agent 事件流
# ============================================================


def _create_agent_stream() -> EventStream[AgentEvent[TAppMessage], list[AgentMessage[TAppMessage]]]:
    def is_terminal(event: AgentEvent[TAppMessage]) -> bool:
        return isinstance(event, AgentEndEvent)

    def extract_result(event: AgentEvent[TAppMessage]) -> list[AgentMessage[TAppMessage]]:
        if isinstance(event, AgentEndEvent):
            return event.messages
        raise ValueError(f"Event is not terminal: {event.type}")

    return EventStream(is_terminal, extract_result)


async def _emit(sink: AgentEventSink[TAppMessage], event: AgentEvent[TAppMessage]) -> None:
    result = sink(event)
    if inspect.isawaitable(result):
        await result


def agent_loop(
    prompts: list[AgentMessage[TAppMessage]],
    context: AgentContext[TAppMessage],
    config: AgentLoopConfig[TAppMessage],
    stream_fn: StreamFn | None = None,
    cancel_event: asyncio.Event | None = None,
) -> EventStream[AgentEvent[TAppMessage], list[AgentMessage[TAppMessage]]]:
    """从一批新消息启动循环，并返回可异步迭代的 Agent 事件流。"""
    stream: EventStream[AgentEvent[TAppMessage], list[AgentMessage[TAppMessage]]] = (
        _create_agent_stream()
    )

    async def drive() -> None:
        try:
            await run_agent_loop(prompts, context, config, stream.push, stream_fn, cancel_event)
        except BaseException as error:
            stream.fail(error)

    asyncio.create_task(drive())
    return stream


def agent_loop_continue(
    context: AgentContext[TAppMessage],
    config: AgentLoopConfig[TAppMessage],
    stream_fn: StreamFn | None = None,
    cancel_event: asyncio.Event | None = None,
) -> EventStream[AgentEvent[TAppMessage], list[AgentMessage[TAppMessage]]]:
    """从现有上下文继续；尾消息必须能转换成 user 或 toolResult。"""
    _validate_continue_context(context)
    stream: EventStream[AgentEvent[TAppMessage], list[AgentMessage[TAppMessage]]] = (
        _create_agent_stream()
    )

    async def drive() -> None:
        try:
            await run_agent_loop_continue(context, config, stream.push, stream_fn, cancel_event)
        except BaseException as error:
            stream.fail(error)

    asyncio.create_task(drive())
    return stream


async def run_agent_loop(
    prompts: list[AgentMessage[TAppMessage]],
    context: AgentContext[TAppMessage],
    config: AgentLoopConfig[TAppMessage],
    emit: AgentEventSink[TAppMessage],
    stream_fn: StreamFn | None = None,
    cancel_event: asyncio.Event | None = None,
) -> list[AgentMessage[TAppMessage]]:
    """执行新 prompt 流程；返回值只包含本次调用产生的消息。"""
    new_messages = list(prompts)
    current = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=list(context.tools),
    )
    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for prompt in prompts:
        await _emit(emit, MessageStartEvent(message=prompt))
        await _emit(emit, MessageEndEvent(message=prompt))
    await _run_loop(
        current,
        new_messages,
        config,
        cancel_event,
        emit,
        stream_fn or get_default_stream_fn(),
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext[TAppMessage],
    config: AgentLoopConfig[TAppMessage],
    emit: AgentEventSink[TAppMessage],
    stream_fn: StreamFn | None = None,
    cancel_event: asyncio.Event | None = None,
) -> list[AgentMessage[TAppMessage]]:
    """执行 continuation；不会重复发送既有消息的 message 生命周期事件。"""
    _validate_continue_context(context)
    new_messages: list[AgentMessage[TAppMessage]] = []
    current = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools),
    )
    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    await _run_loop(
        current,
        new_messages,
        config,
        cancel_event,
        emit,
        stream_fn or get_default_stream_fn(),
    )
    return new_messages


def _validate_continue_context(context: AgentContext[TAppMessage]) -> None:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")


async def _drain_messages(
    getter: object,
) -> list[AgentMessage[TAppMessage]]:
    if getter is None or not callable(getter):
        return []
    result = getter()
    resolved = await result if inspect.isawaitable(result) else result
    return list(cast(list[AgentMessage[TAppMessage]] | None, resolved) or [])


# ============================================================
# 双层循环核心
# ============================================================


async def _run_loop(
    initial_context: AgentContext[TAppMessage],
    new_messages: list[AgentMessage[TAppMessage]],
    initial_config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
    stream_fn: StreamFn,
) -> None:
    """运行 tool/steering 内层循环以及 follow-up 外层循环。"""
    current = initial_context
    config = initial_config
    first_turn = True
    pending: list[AgentMessage[TAppMessage]] = await _drain_messages(config.get_steering_messages)

    while True:
        has_more_tool_calls = True

        # 内层：只要模型需要继续执行工具，或存在 steering，就继续请求模型。
        while has_more_tool_calls or pending:
            if first_turn:
                first_turn = False
            else:
                await _emit(emit, TurnStartEvent())

            # steering/follow-up 必须在下一次 assistant 请求之前进入 transcript。
            for queued_message in pending:
                await _emit(emit, MessageStartEvent(message=queued_message))
                await _emit(emit, MessageEndEvent(message=queued_message))
                current.messages.append(queued_message)
                new_messages.append(queued_message)
            pending = []

            # 一轮定义为：一次 assistant 响应，以及该响应触发的整批工具结果。
            message = await _stream_assistant_response(
                current, config, cancel_event, emit, stream_fn
            )
            new_messages.append(message)
            if message.stop_reason in {"error", "aborted"}:
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                batch = (
                    await _fail_truncated_tool_calls(tool_calls, emit)
                    if message.stop_reason == "length"
                    else await _execute_tool_calls(
                        current, message, tool_calls, config, cancel_event, emit
                    )
                )
                tool_results.extend(batch.messages)
                has_more_tool_calls = not batch.terminate
                current.messages.extend(tool_results)
                new_messages.extend(tool_results)

            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))
            # turn hooks 在 turn_end 后运行，并且可以替换下一轮上下文/模型/thinking。
            turn_context = _turn_context(message, tool_results, current, new_messages)
            if config.prepare_next_turn is not None:
                update_result = config.prepare_next_turn(turn_context)
                update = (
                    await update_result if inspect.isawaitable(update_result) else update_result
                )
                if update is not None:
                    current = update.context or current
                    config = replace(config, model=update.model or config.model)
                    if update.thinking_level is not None:
                        options = (
                            config.stream_options.model_copy(deep=True)
                            if config.stream_options
                            else SimpleStreamOptions()
                        )
                        options.reasoning = (
                            None if update.thinking_level == "off" else update.thinking_level
                        )
                        config = replace(config, stream_options=options)

            if config.should_stop_after_turn is not None:
                stop_result = config.should_stop_after_turn(
                    _turn_context(message, tool_results, current, new_messages)
                )
                should_stop = await stop_result if inspect.isawaitable(stop_result) else stop_result
                if should_stop:
                    await _emit(emit, AgentEndEvent(messages=new_messages))
                    return

            pending = await _drain_messages(config.get_steering_messages)

        # 内层自然结束后才消费 follow-up；它不会打断当前工具批次。
        follow_ups: list[AgentMessage[TAppMessage]] = await _drain_messages(
            config.get_follow_up_messages
        )
        if follow_ups:
            pending = follow_ups
            continue
        break

    await _emit(emit, AgentEndEvent(messages=new_messages))


def _turn_context(
    message: AssistantMessage,
    tool_results: list[ToolResultMessage],
    context: AgentContext[TAppMessage],
    new_messages: list[AgentMessage[TAppMessage]],
) -> ShouldStopAfterTurnContext[TAppMessage]:
    return ShouldStopAfterTurnContext(
        message=message,
        tool_results=tool_results,
        context=context,
        new_messages=new_messages,
    )


async def _default_convert_to_llm(
    messages: list[AgentMessage[TAppMessage]],
) -> list[Message]:
    return [
        message
        for message in messages
        if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage))
    ]


# ============================================================
# LLM 调用边界
# ============================================================


async def _call_transform(
    transform: object,
    messages: list[AgentMessage[TAppMessage]],
    cancel_event: asyncio.Event | None,
) -> list[AgentMessage[TAppMessage]]:
    if not callable(transform):
        return messages
    parameters = inspect.signature(transform).parameters
    result = transform(messages, cancel_event) if len(parameters) >= 2 else transform(messages)
    resolved = await result if inspect.isawaitable(result) else result
    return list(cast(list[AgentMessage[TAppMessage]], resolved))


async def _stream_assistant_response(
    context: AgentContext[TAppMessage],
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
    stream_fn: StreamFn,
) -> AssistantMessage:
    """桥接 AgentMessage、Provider Context 与底层 Assistant 流事件。"""
    # 1. AgentMessage 层上下文变换，例如裁剪、压缩或外部信息注入。
    messages = await _call_transform(config.transform_context, list(context.messages), cancel_event)
    # 2. 仅在此处转换为 Provider 能理解的标准 Message。
    converter = config.convert_to_llm or _default_convert_to_llm
    converted = converter(messages)
    llm_messages = await converted if inspect.isawaitable(converted) else converted
    # 3. AgentTool 只把模型可见的声明传给 AI 层，execute 仍留在 Agent 层。
    llm_context = Context(
        system_prompt=context.system_prompt or None,
        messages=llm_messages,
        tools=[
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                constrained_sampling=getattr(tool, "constrained_sampling", None),
            )
            for tool in context.tools
        ]
        or None,
    )

    # 4. 每轮重新解析调用选项和短期 API key，并透传同一个取消事件。
    options = (
        config.stream_options.model_copy(deep=True)
        if config.stream_options is not None
        else SimpleStreamOptions()
    )
    options.cancel_event = cancel_event
    if config.get_api_key is not None:
        key_result = config.get_api_key(config.model.provider)
        key = await key_result if inspect.isawaitable(key_result) else key_result
        if key:
            options.api_key = key

    stream_result = stream_fn(config.model, llm_context, options)
    stream = await stream_result if inspect.isawaitable(stream_result) else stream_result
    # 5. partial 占位替换；最终 transcript 中只保留终态 AssistantMessage。
    partial_added = False
    async for event in stream:
        if isinstance(event, StartEvent):
            partial_added = True
            context.messages.append(event.partial)
            await _emit(emit, MessageStartEvent(message=event.partial.model_copy(deep=True)))
        elif isinstance(event, (DoneEvent, ErrorEvent)):
            final = await stream.result()
            if partial_added:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await _emit(emit, MessageStartEvent(message=final.model_copy(deep=True)))
            await _emit(emit, MessageEndEvent(message=final))
            return final
        elif partial_added:
            context.messages[-1] = event.partial
            await _emit(
                emit,
                MessageUpdateEvent(
                    message=event.partial.model_copy(deep=True),
                    assistant_message_event=event,
                ),
            )

    final = await stream.result()
    if partial_added:
        context.messages[-1] = final
    else:
        context.messages.append(final)
        await _emit(emit, MessageStartEvent(message=final.model_copy(deep=True)))
    await _emit(emit, MessageEndEvent(message=final))
    return final


# ============================================================
# 工具调用的准备、执行与归约
# ============================================================


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, object]


@dataclass(slots=True)
class _FinalizedToolCall:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class _ExecutedBatch:
    messages: list[ToolResultMessage]
    terminate: bool


def _find_tool(tools: list[AgentTool], name: str) -> AgentTool | None:
    return next((tool for tool in tools if tool.name == name), None)


def _validate_arguments(tool: AgentTool, arguments: dict[str, object]) -> dict[str, object]:
    """先应用兼容参数转换，再严格执行 Draft 2020-12 JSON Schema 校验。"""
    prepare = getattr(tool, "prepare_arguments", None)
    prepared = prepare(deepcopy(arguments)) if prepare is not None else deepcopy(arguments)
    prepared_call = ToolCall(id="validation", name=tool.name, arguments=prepared)
    return validate_tool_arguments(tool, prepared_call)


async def _prepare_tool_call(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
) -> _PreparedToolCall | _FinalizedToolCall:
    """顺序执行工具 preflight：查找、转换、校验、before hook 和取消检查。"""
    tool = _find_tool(context.tools, tool_call.name)
    if tool is None:
        return _FinalizedToolCall(
            tool_call, _error_tool_result(f"Tool {tool_call.name} not found"), True
        )
    try:
        args = _validate_arguments(tool, tool_call.arguments)
        if config.before_tool_call is not None:
            hook_result = config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=context,
                ),
                cancel_event,
            )
            before = await hook_result if inspect.isawaitable(hook_result) else hook_result
            if cancel_event is not None and cancel_event.is_set():
                return _FinalizedToolCall(tool_call, _error_tool_result("Operation aborted"), True)
            if before is not None and before.block:
                return _FinalizedToolCall(
                    tool_call,
                    _error_tool_result(before.reason or "Tool execution was blocked"),
                    True,
                )
        if cancel_event is not None and cancel_event.is_set():
            return _FinalizedToolCall(tool_call, _error_tool_result("Operation aborted"), True)
        return _PreparedToolCall(tool_call, tool, args)
    except Exception as error:
        return _FinalizedToolCall(tool_call, _error_tool_result(str(error)), True)


async def _invoke_tool_execute(
    prepared: _PreparedToolCall,
    cancel_event: asyncio.Event | None,
    on_update: object,
) -> AgentToolResult:
    """调用工具，并兼容新式 cancel_event 签名与已有三参数 Python 工具。"""
    callable_execute = cast(
        Callable[..., AgentToolResult | Awaitable[AgentToolResult]],
        prepared.tool.execute,
    )
    parameters = inspect.signature(callable_execute).parameters
    kwargs: dict[str, object] = {}
    for name in parameters:
        if name in {"cancel_event", "signal"}:
            kwargs[name] = cancel_event
        elif name in {"on_update", "onUpdate"}:
            kwargs[name] = on_update
    result = callable_execute(prepared.tool_call.id, prepared.args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
) -> tuple[AgentToolResult, bool]:
    """执行工具并关闭 update 作用域；工具完成后的迟到更新会被忽略。"""
    update_tasks: list[asyncio.Task[None]] = []
    accepting_updates = True

    def on_update(partial: AgentToolResult) -> None:
        if accepting_updates:
            update_tasks.append(
                asyncio.create_task(
                    _emit(
                        emit,
                        ToolExecutionUpdateEvent(
                            tool_call_id=prepared.tool_call.id,
                            tool_name=prepared.tool_call.name,
                            args=dict(prepared.tool_call.arguments),
                            partial_result=partial,
                        ),
                    )
                )
            )

    try:
        result = await _invoke_tool_execute(prepared, cancel_event, on_update)
        return result, False
    except asyncio.CancelledError:
        return _error_tool_result("Operation aborted"), True
    except Exception as error:
        return _error_tool_result(str(error)), True
    finally:
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)


async def _finalize_tool_call(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    result: AgentToolResult,
    is_error: bool,
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
) -> _FinalizedToolCall:
    """应用 after hook 的字段级覆盖，并把 hook 异常转换为工具错误。"""
    if config.after_tool_call is not None:
        try:
            hook_result = config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=context,
                ),
                cancel_event,
            )
            after = await hook_result if inspect.isawaitable(hook_result) else hook_result
            if after is not None:
                result = AgentToolResult(
                    content=after.content if after.content is not None else result.content,
                    details=after.details if after.details is not _UNSET else result.details,
                    usage=after.usage if after.usage is not None else result.usage,
                    added_tool_names=result.added_tool_names,
                    terminate=after.terminate if after.terminate is not None else result.terminate,
                )
                is_error = after.is_error if after.is_error is not None else is_error
        except Exception as error:
            result = _error_tool_result(str(error))
            is_error = True
    return _FinalizedToolCall(prepared.tool_call, result, is_error)


async def _emit_tool_end(finalized: _FinalizedToolCall, emit: AgentEventSink[TAppMessage]) -> None:
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )


def _create_tool_result_message(finalized: _FinalizedToolCall) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content or [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        added_tool_names=finalized.result.added_tool_names or None,
        is_error=finalized.is_error,
    )


async def _emit_tool_result(message: ToolResultMessage, emit: AgentEventSink[TAppMessage]) -> None:
    await _emit(emit, MessageStartEvent(message=message))
    await _emit(emit, MessageEndEvent(message=message))


async def _finish_prepared(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
) -> _FinalizedToolCall:
    result, is_error = await _execute_prepared_tool_call(prepared, cancel_event, emit)
    finalized = await _finalize_tool_call(
        context,
        assistant_message,
        prepared,
        result,
        is_error,
        config,
        cancel_event,
    )
    await _emit_tool_end(finalized, emit)
    return finalized


async def _execute_tool_calls(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
) -> _ExecutedBatch:
    """按照全局模式和每工具覆盖选择整批执行策略。"""
    force_sequential = config.tool_execution == "sequential" or any(
        getattr(_find_tool(context.tools, call.name), "execution_mode", None) == "sequential"
        for call in tool_calls
    )
    if force_sequential:
        return await _execute_sequential(
            context, assistant_message, tool_calls, config, cancel_event, emit
        )
    return await _execute_parallel(
        context, assistant_message, tool_calls, config, cancel_event, emit
    )


async def _execute_sequential(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
) -> _ExecutedBatch:
    """逐个完成 start、preflight、execute、end 与 ToolResult 消息。"""
    finalized_calls: list[_FinalizedToolCall] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=dict(tool_call.arguments),
            ),
        )
        prepared = await _prepare_tool_call(
            context, assistant_message, tool_call, config, cancel_event
        )
        if isinstance(prepared, _FinalizedToolCall):
            finalized = prepared
            await _emit_tool_end(finalized, emit)
        else:
            finalized = await _finish_prepared(
                context,
                assistant_message,
                prepared,
                config,
                cancel_event,
                emit,
            )
        await _emit_tool_result(_create_tool_result_message(finalized), emit)
        finalized_calls.append(finalized)
        if cancel_event is not None and cancel_event.is_set():
            break
    return _batch_from_finalized(finalized_calls)


async def _execute_parallel(
    context: AgentContext[TAppMessage],
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig[TAppMessage],
    cancel_event: asyncio.Event | None,
    emit: AgentEventSink[TAppMessage],
) -> _ExecutedBatch:
    """顺序 preflight，随后并发执行，最后按工具来源顺序写入结果消息。"""
    prepared_entries: list[_FinalizedToolCall | _PreparedToolCall] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=dict(tool_call.arguments),
            ),
        )
        prepared = await _prepare_tool_call(
            context, assistant_message, tool_call, config, cancel_event
        )
        if isinstance(prepared, _FinalizedToolCall):
            await _emit_tool_end(prepared, emit)
        prepared_entries.append(prepared)
        if cancel_event is not None and cancel_event.is_set():
            break

    # 先创建全部 Task，再按来源顺序 await；Task 内的 end 事件仍按真实完成顺序发出。
    entries: list[_FinalizedToolCall | asyncio.Task[_FinalizedToolCall]] = [
        entry
        if isinstance(entry, _FinalizedToolCall)
        else asyncio.create_task(
            _finish_prepared(
                context,
                assistant_message,
                entry,
                config,
                cancel_event,
                emit,
            )
        )
        for entry in prepared_entries
    ]
    finalized_calls = [
        await entry if isinstance(entry, asyncio.Task) else entry for entry in entries
    ]
    for finalized in finalized_calls:
        await _emit_tool_result(_create_tool_result_message(finalized), emit)
    return _batch_from_finalized(finalized_calls)


def _batch_from_finalized(finalized_calls: list[_FinalizedToolCall]) -> _ExecutedBatch:
    """构造来源顺序的 ToolResult，并计算“全部 terminate”批次语义。"""
    messages = [_create_tool_result_message(finalized) for finalized in finalized_calls]
    terminate = bool(finalized_calls) and all(
        finalized.result.terminate for finalized in finalized_calls
    )
    return _ExecutedBatch(messages=messages, terminate=terminate)


async def _fail_truncated_tool_calls(
    tool_calls: list[ToolCall], emit: AgentEventSink[TAppMessage]
) -> _ExecutedBatch:
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=dict(tool_call.arguments),
            ),
        )
        finalized = _FinalizedToolCall(
            tool_call,
            _error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output '
                "token limit, so its arguments may be truncated. Re-issue the tool call with "
                "complete arguments."
            ),
            True,
        )
        await _emit_tool_end(finalized, emit)
        message = _create_tool_result_message(finalized)
        await _emit_tool_result(message, emit)
        messages.append(message)
    return _ExecutedBatch(messages=messages, terminate=False)


def _error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


__all__ = [
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
]
