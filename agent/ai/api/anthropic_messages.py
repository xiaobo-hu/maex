"""Anthropic Messages protocol translator with extended thinking support."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from anthropic import AsyncAnthropic

from ..constrained_sampling import resolve_json_schema_strict_sampling
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
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ._shared import calculate_cost, error_message, parse_streaming_json, value

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "toolUse",
    "max_tokens": "length",
    "refusal": "error",
}


def _create_client(model: Model, options: StreamOptions) -> AsyncAnthropic:
    if not options.api_key:
        raise ValueError("No API key configured for provider")
    return AsyncAnthropic(
        api_key=options.api_key,
        base_url=model.base_url or None,
        default_headers={
            key: item for key, item in (options.headers or {}).items() if item is not None
        }
        or None,
        max_retries=0,
        http_client=cast(Any, options.http_client),
    )


def _user_content(message: UserMessage) -> str | list[dict[str, object]]:
    if isinstance(message.content, str):
        return sanitize_surrogates(message.content)
    result: list[dict[str, object]] = []
    for block in message.content:
        if block.type == "text":
            result.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            result.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )
    return result


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_results: list[dict[str, object]] = []

    def flush_results() -> None:
        if pending_results:
            messages.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in context.messages:
        if isinstance(message, ToolResultMessage):
            content: list[dict[str, object]] = []
            for result_block in message.content:
                if result_block.type == "text":
                    content.append({"type": "text", "text": sanitize_surrogates(result_block.text)})
                else:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": result_block.mime_type,
                                "data": result_block.data,
                            },
                        }
                    )
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": content,
                    "is_error": message.is_error,
                }
            )
            continue
        flush_results()
        if isinstance(message, UserMessage):
            messages.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            blocks: list[dict[str, object]] = []
            for assistant_block in message.content:
                if isinstance(assistant_block, TextContent):
                    blocks.append(
                        {"type": "text", "text": sanitize_surrogates(assistant_block.text)}
                    )
                elif isinstance(assistant_block, ThinkingContent):
                    if assistant_block.redacted:
                        blocks.append(
                            {
                                "type": "redacted_thinking",
                                "data": assistant_block.thinking_signature or "",
                            }
                        )
                    elif assistant_block.thinking_signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(assistant_block.thinking),
                                "signature": sanitize_surrogates(
                                    assistant_block.thinking_signature
                                ),
                            }
                        )
                    elif assistant_block.thinking:
                        blocks.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(assistant_block.thinking),
                            }
                        )
                elif isinstance(assistant_block, ToolCall):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": assistant_block.id,
                            "name": assistant_block.name,
                            "input": assistant_block.arguments,
                        }
                    )
            messages.append({"role": "assistant", "content": blocks})
    flush_results()
    return messages


def _convert_tools(context: Context, model: Model | None = None) -> list[dict[str, object]] | None:
    if not context.tools:
        return None
    supports_strict = bool((model.compat or {}).get("supportsStrictTools", True)) if model else True
    converted: list[dict[str, object]] = []
    for tool in context.tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict)
        item: dict[str, object] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                **tool.to_json_schema(),
            },
        }
        if strict is not None:
            item["strict"] = strict
        converted.append(item)
    return converted


def _thinking_config(model: Model, options: StreamOptions) -> dict[str, object] | None:
    if options.thinking_enabled is False:
        return {"type": "disabled"}
    if options.thinking_enabled is not True:
        return None
    if (model.compat or {}).get("forceAdaptiveThinking"):
        result: dict[str, object] = {
            "type": "adaptive",
            "display": options.thinking_display or "summarized",
        }
        if options.effort:
            result["effort"] = options.effort
        return result
    return {"type": "enabled", "budget_tokens": options.thinking_budget_tokens or 1024}


class _Block:
    def __init__(self, content_index: int, call: ToolCall | None = None) -> None:
        self.content_index = content_index
        self.call = call
        self.partial_json = ""


class AnthropicMessagesApi(ProviderStreams):
    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._drive(stream, model, context, options or StreamOptions()))
        return stream

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        resolved = options or SimpleStreamOptions()
        if resolved.reasoning is not None:
            level = clamp_thinking_level(model, resolved.reasoning)
            resolved.thinking_enabled = level != "off"
            if level == "off":
                pass
            elif (model.compat or {}).get("forceAdaptiveThinking"):
                resolved.effort = (
                    model.thinking_level_map.get(level, level)
                    if model.thinking_level_map
                    else level
                )
            else:
                budgets = resolved.thinking_budgets or {}
                resolved.thinking_budget_tokens = budgets.get(level, 1024)
        elif model.reasoning:
            resolved.thinking_enabled = False
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
            params: dict[str, Any] = {
                "model": model.id,
                "messages": _convert_messages(context),
                "max_tokens": options.max_tokens or model.max_tokens or 4096,
                "stream": True,
            }
            if context.system_prompt:
                params["system"] = [
                    {"type": "text", "text": sanitize_surrogates(context.system_prompt)}
                ]
            if (tools := _convert_tools(context, model)) is not None:
                params["tools"] = tools
            if options.temperature is not None and options.thinking_enabled is not True:
                params["temperature"] = options.temperature
            if options.timeout_ms is not None:
                params["timeout"] = options.timeout_ms / 1000
            if (thinking_config := _thinking_config(model, options)) is not None:
                params["thinking"] = thinking_config
            if options.extra_body:
                params.update(options.extra_body)
            client = _create_client(model, options)
            response = await retry_provider_request(
                lambda: client.messages.create(**params),
                max_retries=options.max_retries or 0,
                max_retry_delay_ms=options.max_retry_delay_ms,
                cancel_event=options.cancel_event,
            )
            blocks: dict[int, _Block] = {}
            async for event in response:
                if options.cancel_event is not None and options.cancel_event.is_set():
                    raise asyncio.CancelledError
                event_type = str(value(event, "type", ""))
                if event_type == "message_start":
                    message = value(event, "message")
                    output.response_id = value(message, "id")
                    usage = value(message, "usage")
                    if usage is not None:
                        output.usage.input = int(value(usage, "input_tokens", 0) or 0)
                        output.usage.cache_read = int(
                            value(usage, "cache_read_input_tokens", 0) or 0
                        )
                        output.usage.cache_write = int(
                            value(usage, "cache_creation_input_tokens", 0) or 0
                        )
                elif event_type == "content_block_start":
                    source = value(event, "content_block")
                    provider_index = int(value(event, "index", 0) or 0)
                    content_index = len(output.content)
                    block_type = value(source, "type")
                    block = _Block(content_index)
                    blocks[provider_index] = block
                    if block_type == "text":
                        output.content.append(TextContent(text=""))
                        event_stream.push(
                            TextStartEvent(
                                content_index=content_index, partial=output.model_copy(deep=True)
                            )
                        )
                    elif block_type in {"thinking", "redacted_thinking"}:
                        output.content.append(
                            ThinkingContent(
                                thinking="[Reasoning redacted]"
                                if block_type == "redacted_thinking"
                                else "",
                                thinking_signature=value(source, "data", "") or "",
                                redacted=block_type == "redacted_thinking",
                            )
                        )
                        event_stream.push(
                            ThinkingStartEvent(
                                content_index=content_index, partial=output.model_copy(deep=True)
                            )
                        )
                    elif block_type == "tool_use":
                        call = ToolCall(
                            id=str(value(source, "id", "")),
                            name=str(value(source, "name", "")),
                            arguments=value(source, "input", {}) or {},
                        )
                        block.call = call
                        output.content.append(call)
                        event_stream.push(
                            ToolCallStartEvent(
                                content_index=content_index, partial=output.model_copy(deep=True)
                            )
                        )
                elif event_type == "content_block_delta":
                    provider_index = int(value(event, "index", 0) or 0)
                    current_block = blocks.get(provider_index)
                    delta = value(event, "delta")
                    if current_block is None:
                        continue
                    target = output.content[current_block.content_index]
                    delta_type = value(delta, "type")
                    if delta_type == "text_delta" and isinstance(target, TextContent):
                        text = str(value(delta, "text", ""))
                        target.text += text
                        event_stream.push(
                            TextDeltaEvent(
                                content_index=current_block.content_index,
                                delta=text,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    elif delta_type == "thinking_delta" and isinstance(target, ThinkingContent):
                        thinking_delta = str(value(delta, "thinking", ""))
                        target.thinking += thinking_delta
                        event_stream.push(
                            ThinkingDeltaEvent(
                                content_index=current_block.content_index,
                                delta=thinking_delta,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    elif delta_type == "signature_delta" and isinstance(target, ThinkingContent):
                        target.thinking_signature = (target.thinking_signature or "") + str(
                            value(delta, "signature", "")
                        )
                    elif delta_type == "input_json_delta" and isinstance(target, ToolCall):
                        fragment = str(value(delta, "partial_json", ""))
                        current_block.partial_json += fragment
                        target.arguments = parse_streaming_json(current_block.partial_json)
                        event_stream.push(
                            ToolCallDeltaEvent(
                                content_index=current_block.content_index,
                                delta=fragment,
                                partial=output.model_copy(deep=True),
                            )
                        )
                elif event_type == "content_block_stop":
                    completed_block = blocks.get(int(value(event, "index", 0) or 0))
                    if completed_block is None:
                        continue
                    target = output.content[completed_block.content_index]
                    if isinstance(target, TextContent):
                        event_stream.push(
                            TextEndEvent(
                                content_index=completed_block.content_index,
                                content=target.text,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    elif isinstance(target, ThinkingContent):
                        event_stream.push(
                            ThinkingEndEvent(
                                content_index=completed_block.content_index,
                                content=target.thinking,
                                partial=output.model_copy(deep=True),
                            )
                        )
                    elif isinstance(target, ToolCall):
                        event_stream.push(
                            ToolCallEndEvent(
                                content_index=completed_block.content_index,
                                tool_call=target.model_copy(deep=True),
                                partial=output.model_copy(deep=True),
                            )
                        )
                elif event_type == "message_delta":
                    delta = value(event, "delta")
                    raw_reason = value(delta, "stop_reason")
                    if raw_reason:
                        output.raw_stop_reason = str(raw_reason)
                        output.stop_reason = _STOP_REASON_MAP.get(str(raw_reason), "error")
                    usage = value(event, "usage")
                    if usage is not None:
                        output.usage.output = int(value(usage, "output_tokens", 0) or 0)
                        details = value(usage, "output_tokens_details")
                        reasoning = int(value(details, "thinking_tokens", 0) or 0)
                        output.usage.reasoning = reasoning or None
            if output.stop_reason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stop_reason == "error":
                raise RuntimeError("Anthropic provider returned an error stop reason")
            output.usage.total_tokens = (
                output.usage.input
                + output.usage.output
                + output.usage.cache_read
                + output.usage.cache_write
            )
            calculate_cost(output.usage, model)
            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
        except asyncio.CancelledError:
            output.stop_reason = "aborted"
            output.error_message = None
            event_stream.push(ErrorEvent(reason="aborted", error=output))
        except BaseException as error:
            output.stop_reason = "error"
            output.error_message = error_message(error)
            event_stream.push(ErrorEvent(reason="error", error=output))


def anthropic_messages_api() -> AnthropicMessagesApi:
    return AnthropicMessagesApi()


__all__ = ["AnthropicMessagesApi", "anthropic_messages_api"]
