"""OpenAI Chat Completions translator for OpenAI-compatible services."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from openai import AsyncOpenAI

from ..constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from ..events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ..models import ProviderStreams, clamp_thinking_level
from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_retry import retry_provider_stream
from ..utils.sanitize_unicode import sanitize_surrogates
from ._shared import calculate_cost, error_message, parse_streaming_json, value

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": "stop",
    "end_turn": "stop",
    "length": "length",
    "tool_calls": "toolUse",
    "function_call": "toolUse",
    "content_filter": "error",
}
_THINKING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def _require_api_key(options: StreamOptions) -> str:
    if options.api_key:
        return options.api_key
    headers = {key.lower(): item for key, item in (options.headers or {}).items()}
    if headers.get("authorization"):
        return "unused"
    raise ValueError("No API key configured for provider")


def _create_client(model: Model, options: StreamOptions) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_require_api_key(options),
        base_url=model.base_url or None,
        default_headers={
            key: item for key, item in (options.headers or {}).items() if item is not None
        }
        or None,
        max_retries=0,
        http_client=cast(Any, options.http_client),
    )


def _convert_messages(
    context: Context,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if context.system_prompt:
        messages.append({"role": "system", "content": sanitize_surrogates(context.system_prompt)})
    for message in context.messages:
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                content: object = sanitize_surrogates(message.content)
            else:
                parts: list[dict[str, object]] = []
                for user_block in message.content:
                    if user_block.type == "text":
                        parts.append({"type": "text", "text": sanitize_surrogates(user_block.text)})
                    else:
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{user_block.mime_type};base64,{user_block.data}"
                                },
                            }
                        )
                content = parts
            messages.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            texts: list[str] = []
            calls: list[dict[str, object]] = []
            for assistant_block in message.content:
                if isinstance(assistant_block, TextContent):
                    texts.append(sanitize_surrogates(assistant_block.text))
                elif isinstance(assistant_block, ThinkingContent) and assistant_block.thinking:
                    texts.append(sanitize_surrogates(assistant_block.thinking))
                elif isinstance(assistant_block, ToolCall):
                    input_property = (grammar_tool_input_properties or {}).get(assistant_block.name)
                    if input_property is not None:
                        calls.append(
                            {
                                "id": assistant_block.id,
                                "type": "custom",
                                "custom": {
                                    "name": assistant_block.name,
                                    "input": get_grammar_tool_input(
                                        assistant_block.name,
                                        assistant_block.arguments,
                                        input_property,
                                    ),
                                },
                            }
                        )
                        continue
                    calls.append(
                        {
                            "id": assistant_block.id,
                            "type": "function",
                            "function": {
                                "name": assistant_block.name,
                                "arguments": sanitize_surrogates(
                                    json.dumps(assistant_block.arguments, ensure_ascii=False)
                                ),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
        elif isinstance(message, ToolResultMessage):
            text = "\n".join(
                sanitize_surrogates(block.text)
                for block in message.content
                if isinstance(block, TextContent)
            )
            messages.append({"role": "tool", "content": text, "tool_call_id": message.tool_call_id})
            images = [block for block in message.content if isinstance(block, ImageContent)]
            if images:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Latest robot observation after the tool call.",
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image.mime_type};base64,{image.data}"
                                    },
                                }
                                for image in images
                            ],
                        ],
                    }
                )
    return messages


def _convert_tools(context: Context, model: Model | None = None) -> list[dict[str, object]] | None:
    if not context.tools:
        return None
    compat = model.compat or {} if model is not None else {}
    supports_strict = bool(compat.get("supportsStrictMode", True))
    supports_grammar = bool(compat.get("supportsOpenAIGrammarTools", False))
    converted: list[dict[str, object]] = []
    for tool in context.tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_grammar)
        if grammar is not None:
            converted.append(
                {
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": grammar.format, "definition": grammar.definition},
                        },
                    },
                }
            )
            continue
        strict = resolve_json_schema_strict_sampling(tool, supports_strict)
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.to_json_schema(),
                    "strict": strict or False,
                },
            }
        )
    return converted


def _parse_usage(raw: object, model: Model) -> Usage:
    prompt = int(value(raw, "prompt_tokens", 0) or 0)
    completion = int(value(raw, "completion_tokens", 0) or 0)
    prompt_details = value(raw, "prompt_tokens_details")
    completion_details = value(raw, "completion_tokens_details")
    cache_read = int(value(prompt_details, "cached_tokens", 0) or 0)
    cache_write = int(value(prompt_details, "cache_write_tokens", 0) or 0)
    reasoning = int(value(completion_details, "reasoning_tokens", 0) or 0)
    usage = Usage(
        input=max(0, prompt - cache_read - cache_write),
        output=completion,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning or None,
        total_tokens=prompt + completion,
    )
    calculate_cost(usage, model)
    return usage


def _has_stream_progress(chunk: object) -> bool:
    choices = value(chunk, "choices", []) or []
    if not choices:
        return False
    choice = choices[0]
    if value(choice, "finish_reason"):
        return True
    delta = value(choice, "delta")
    return bool(
        value(delta, "content")
        or value(delta, "tool_calls", [])
        or any(value(delta, field) for field in _THINKING_FIELDS)
    )


class _ToolState:
    def __init__(self, content_index: int, custom_input_property: str | None = None) -> None:
        self.content_index = content_index
        self.call = ToolCall(id="", name="", arguments={})
        self.arguments = ""
        self.custom_input_property = custom_input_property
        self.custom_input_buffer = (
            GrammarToolInputJsonBuffer() if custom_input_property is not None else None
        )
        if custom_input_property is not None:
            self.call.arguments = {custom_input_property: ""}


class OpenAICompletionsApi(ProviderStreams):
    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        resolved = options or StreamOptions()
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._drive(stream, model, context, resolved))
        return stream

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        resolved = options or SimpleStreamOptions()
        if resolved.reasoning is not None and resolved.reasoning_effort is None:
            level = clamp_thinking_level(model, resolved.reasoning)
            if level != "off":
                resolved.reasoning_effort = level
        return self.stream(model, context, resolved)

    async def _drive(
        self,
        event_stream: AssistantMessageEventStream,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> None:
        output = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, stop_reason="pending"
        )
        event_stream.push(StartEvent(partial=output.model_copy(deep=True)))
        try:
            compat = model.compat or {}
            grammar_properties = create_grammar_tool_input_properties(
                context.tools,
                bool(compat.get("supportsOpenAIGrammarTools", False)),
            )
            params: dict[str, Any] = {
                "model": model.id,
                "messages": _convert_messages(context, grammar_properties),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if (tools := _convert_tools(context, model)) is not None:
                params["tools"] = tools
            if options.temperature is not None:
                params["temperature"] = options.temperature
            if options.max_tokens is not None:
                params["max_tokens"] = options.max_tokens
            if options.timeout_ms is not None:
                params["timeout"] = options.timeout_ms / 1000
            if options.tool_choice is not None:
                params["tool_choice"] = options.tool_choice
            if options.reasoning_effort is not None:
                params["reasoning_effort"] = options.reasoning_effort
            extra_body = dict(options.extra_body or {})
            thinking_format = (model.compat or {}).get("thinkingFormat")
            if model.reasoning and thinking_format == "kimi" and "thinking" not in extra_body:
                extra_body["thinking"] = {
                    "type": "enabled" if options.reasoning_effort else "disabled"
                }
            if extra_body:
                params["extra_body"] = extra_body

            client = _create_client(model, options)
            response = retry_provider_stream(
                lambda: client.chat.completions.create(**params),
                max_retries=options.max_retries or 0,
                max_retry_delay_ms=options.max_retry_delay_ms,
                cancel_event=options.cancel_event,
                # 配置重试时先缓冲完整响应，避免 partial 已提交后断线无法安全重放。
                is_committed=(
                    (lambda _chunk: False)
                    if (options.max_retries or 0) > 0
                    else _has_stream_progress
                ),
            )
            text_block: TextContent | None = None
            text_index = -1
            thinking_block: ThinkingContent | None = None
            thinking_index = -1
            tools_by_index: dict[int, _ToolState] = {}
            tools_by_id: dict[str, _ToolState] = {}
            last_tool_state: _ToolState | None = None
            saw_finish = False
            async for chunk in response:
                if options.cancel_event is not None and options.cancel_event.is_set():
                    raise asyncio.CancelledError
                if (usage := value(chunk, "usage")) is not None:
                    output.usage = _parse_usage(usage, model)
                choices = value(chunk, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = value(choice, "delta")
                finish = value(choice, "finish_reason")
                if finish:
                    saw_finish = True
                    output.raw_stop_reason = str(finish)
                    output.stop_reason = _STOP_REASON_MAP.get(str(finish), "error")
                    if output.stop_reason == "error":
                        output.error_message = f"Unhandled finish_reason: {finish}"
                content = value(delta, "content")
                if content:
                    if text_block is None:
                        text_block = TextContent(text="")
                        text_index = len(output.content)
                        output.content.append(text_block)
                        event_stream.push(
                            TextStartEvent(
                                content_index=text_index,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    text_block.text += str(content)
                    event_stream.push(
                        TextDeltaEvent(
                            content_index=text_index,
                            delta=str(content),
                            partial=output.model_copy(deep=True),
                        )
                    )
                for field in _THINKING_FIELDS:
                    thinking = value(delta, field)
                    if thinking:
                        if thinking_block is None:
                            thinking_block = ThinkingContent(thinking="")
                            thinking_index = len(output.content)
                            output.content.append(thinking_block)
                            event_stream.push(
                                ThinkingStartEvent(
                                    content_index=thinking_index,
                                    partial=output.model_copy(deep=True),
                                )
                            )
                        thinking_block.thinking += str(thinking)
                        event_stream.push(
                            ThinkingDeltaEvent(
                                content_index=thinking_index,
                                delta=str(thinking),
                                partial=output.model_copy(deep=True),
                            )
                        )
                        break
                for tool_delta in value(delta, "tool_calls", []) or []:
                    raw_index = value(tool_delta, "index")
                    index = int(raw_index) if raw_index is not None else None
                    identifier = str(value(tool_delta, "id", "") or "")
                    state = tools_by_index.get(index) if index is not None else None
                    if state is None and identifier:
                        state = tools_by_id.get(identifier)
                    if state is None and index is None and not identifier:
                        state = last_tool_state
                    function = value(tool_delta, "function")
                    custom = value(tool_delta, "custom")
                    name_value = value(function, "name") or value(custom, "name")
                    if state is None:
                        custom_property = (
                            grammar_properties.get(str(name_value or "")) if custom else None
                        )
                        state = _ToolState(len(output.content), custom_property)
                        output.content.append(state.call)
                        event_stream.push(
                            ToolCallStartEvent(
                                content_index=state.content_index,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    last_tool_state = state
                    if index is not None:
                        tools_by_index[index] = state
                    if identifier:
                        state.call.id = identifier
                        tools_by_id[identifier] = state
                    if name_value:
                        state.call.name = str(name_value)
                    arguments_delta = str(value(function, "arguments", "") or "")
                    if arguments_delta:
                        state.arguments += arguments_delta
                        state.call.arguments = parse_streaming_json(state.arguments)
                        event_stream.push(
                            ToolCallDeltaEvent(
                                content_index=state.content_index,
                                delta=arguments_delta,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    elif custom is not None and state.custom_input_property:
                        custom_delta = str(value(custom, "input", "") or "")
                        if custom_delta:
                            current = str(state.call.arguments[state.custom_input_property])
                            next_input = current + custom_delta
                            assert state.custom_input_buffer is not None
                            json_delta = append_grammar_tool_input_json_delta(
                                state.custom_input_buffer,
                                state.custom_input_property,
                                next_input,
                                close=False,
                            )
                            state.call.arguments = {state.custom_input_property: next_input}
                            if json_delta:
                                event_stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=state.content_index,
                                        delta=json_delta,
                                        partial=output.model_copy(deep=True),
                                    )
                                )
            if not saw_finish:
                raise RuntimeError("OpenAI stream ended without finish_reason")
            if text_block is not None:
                event_stream.push(
                    TextEndEvent(
                        content_index=text_index,
                        content=text_block.text,
                        partial=output.model_copy(deep=True),
                    )
                )
            if thinking_block is not None:
                event_stream.push(
                    ThinkingEndEvent(
                        content_index=thinking_index,
                        content=thinking_block.thinking,
                        partial=output.model_copy(deep=True),
                    )
                )
            all_states = [*tools_by_index.values(), *tools_by_id.values()]
            if last_tool_state is not None:
                all_states.append(last_tool_state)
            unique_states = {id(state): state for state in all_states}
            for state in unique_states.values():
                if state.custom_input_property and state.custom_input_buffer:
                    closing = append_grammar_tool_input_json_delta(
                        state.custom_input_buffer,
                        state.custom_input_property,
                        str(state.call.arguments[state.custom_input_property]),
                        close=True,
                    )
                    if closing:
                        event_stream.push(
                            ToolCallDeltaEvent(
                                content_index=state.content_index,
                                delta=closing,
                                partial=output.model_copy(deep=True),
                            )
                        )
                event_stream.push(
                    ToolCallEndEvent(
                        content_index=state.content_index,
                        tool_call=state.call.model_copy(deep=True),
                        partial=output.model_copy(deep=True),
                    )
                )
            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "OpenAI provider error")
            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
        except asyncio.CancelledError:
            output.stop_reason = "aborted"
            output.error_message = None
            event_stream.push(ErrorEvent(reason="aborted", error=output))
        except BaseException as error:
            output.stop_reason = "error"
            output.error_message = error_message(error)
            event_stream.push(ErrorEvent(reason="error", error=output))


def openai_completions_api() -> OpenAICompletionsApi:
    return OpenAICompletionsApi()


__all__ = ["OpenAICompletionsApi", "openai_completions_api"]
