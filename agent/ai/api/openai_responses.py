"""Core OpenAI Responses API translator used by the official provider."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from openai import AsyncOpenAI

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
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ._shared import calculate_cost, error_message, parse_streaming_json, value


def _create_client(model: Model, options: StreamOptions) -> AsyncOpenAI:
    if not options.api_key:
        raise ValueError("No API key configured for provider")
    return AsyncOpenAI(
        api_key=options.api_key,
        base_url=model.base_url or None,
        default_headers={
            key: item for key, item in (options.headers or {}).items() if item is not None
        }
        or None,
        max_retries=0,
        http_client=cast(Any, options.http_client),
    )


def _convert_input(context: Context) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in context.messages:
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                content: object = sanitize_surrogates(message.content)
            else:
                content = [
                    {"type": "input_text", "text": sanitize_surrogates(block.text)}
                    if block.type == "text"
                    else {
                        "type": "input_image",
                        "image_url": f"data:{block.mime_type};base64,{block.data}",
                    }
                    for block in message.content
                ]
            result.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            text = "\n".join(
                sanitize_surrogates(block.text)
                for block in message.content
                if isinstance(block, TextContent)
            )
            if text:
                result.append({"role": "assistant", "content": text})
            for block in message.content:
                if isinstance(block, ToolCall):
                    result.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": sanitize_surrogates(json.dumps(block.arguments)),
                        }
                    )
        elif isinstance(message, ToolResultMessage):
            text = "\n".join(
                sanitize_surrogates(block.text)
                for block in message.content
                if isinstance(block, TextContent)
            )
            result.append(
                {"type": "function_call_output", "call_id": message.tool_call_id, "output": text}
            )
    return result


class _CallState:
    def __init__(self, index: int, identifier: str = "", name: str = "") -> None:
        self.index = index
        self.call = ToolCall(id=identifier, name=name, arguments={})
        self.arguments = ""


class OpenAIResponsesApi(ProviderStreams):
    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._drive(stream, model, context, options or StreamOptions()))
        return stream

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
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
            params: dict[str, Any] = {
                "model": model.id,
                "input": _convert_input(context),
                "stream": True,
            }
            if context.system_prompt:
                params["instructions"] = sanitize_surrogates(context.system_prompt)
            if context.tools:
                supports_strict = bool((model.compat or {}).get("supportsStrictMode", True))
                converted_tools: list[dict[str, object]] = []
                for tool in context.tools:
                    strict = resolve_json_schema_strict_sampling(tool, supports_strict)
                    converted_tools.append(
                        {
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.to_json_schema(),
                            "strict": strict or False,
                        }
                    )
                params["tools"] = converted_tools
            if options.max_tokens is not None:
                params["max_output_tokens"] = options.max_tokens
            if options.temperature is not None:
                params["temperature"] = options.temperature
            if options.reasoning_effort is not None:
                params["reasoning"] = {"effort": options.reasoning_effort}
            if options.extra_body:
                params["extra_body"] = options.extra_body
            client = _create_client(model, options)
            response = await retry_provider_request(
                lambda: client.responses.create(**params),
                max_retries=options.max_retries or 0,
                max_retry_delay_ms=options.max_retry_delay_ms,
                cancel_event=options.cancel_event,
            )
            text: TextContent | None = None
            text_index = -1
            thinking: ThinkingContent | None = None
            thinking_index = -1
            calls: dict[str, _CallState] = {}
            completed = None
            async for event in response:
                if options.cancel_event is not None and options.cancel_event.is_set():
                    raise asyncio.CancelledError
                event_type = str(value(event, "type", ""))
                if event_type == "response.created":
                    output.response_id = value(value(event, "response"), "id")
                elif event_type == "response.output_text.delta":
                    delta = str(value(event, "delta", ""))
                    if text is None:
                        text = TextContent(text="")
                        text_index = len(output.content)
                        output.content.append(text)
                        event_stream.push(
                            TextStartEvent(
                                content_index=text_index, partial=output.model_copy(deep=True)
                            )
                        )
                    text.text += delta
                    event_stream.push(
                        TextDeltaEvent(
                            content_index=text_index,
                            delta=delta,
                            partial=output.model_copy(deep=True),
                        )
                    )
                elif event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    delta = str(value(event, "delta", ""))
                    if thinking is None:
                        thinking = ThinkingContent(thinking="")
                        thinking_index = len(output.content)
                        output.content.append(thinking)
                        event_stream.push(
                            ThinkingStartEvent(
                                content_index=thinking_index, partial=output.model_copy(deep=True)
                            )
                        )
                    thinking.thinking += delta
                    event_stream.push(
                        ThinkingDeltaEvent(
                            content_index=thinking_index,
                            delta=delta,
                            partial=output.model_copy(deep=True),
                        )
                    )
                elif event_type == "response.output_item.added":
                    item = value(event, "item")
                    if value(item, "type") == "function_call":
                        key = str(value(item, "id", value(item, "call_id", "")))
                        state = _CallState(
                            len(output.content),
                            str(value(item, "call_id", key)),
                            str(value(item, "name", "")),
                        )
                        calls[key] = state
                        output.content.append(state.call)
                        event_stream.push(
                            ToolCallStartEvent(
                                content_index=state.index, partial=output.model_copy(deep=True)
                            )
                        )
                elif event_type == "response.function_call_arguments.delta":
                    key = str(value(event, "item_id", ""))
                    delta_state = calls.get(key)
                    if delta_state is not None:
                        delta = str(value(event, "delta", ""))
                        delta_state.arguments += delta
                        delta_state.call.arguments = parse_streaming_json(delta_state.arguments)
                        event_stream.push(
                            ToolCallDeltaEvent(
                                content_index=delta_state.index,
                                delta=delta,
                                partial=output.model_copy(deep=True),
                            )
                        )
                elif event_type == "response.completed":
                    completed = value(event, "response")
            if text is not None:
                event_stream.push(
                    TextEndEvent(
                        content_index=text_index,
                        content=text.text,
                        partial=output.model_copy(deep=True),
                    )
                )
            if thinking is not None:
                event_stream.push(
                    ThinkingEndEvent(
                        content_index=thinking_index,
                        content=thinking.thinking,
                        partial=output.model_copy(deep=True),
                    )
                )
            for state in calls.values():
                event_stream.push(
                    ToolCallEndEvent(
                        content_index=state.index,
                        tool_call=state.call.model_copy(deep=True),
                        partial=output.model_copy(deep=True),
                    )
                )
            if completed is None:
                raise RuntimeError("OpenAI Responses stream ended without response.completed")
            usage = value(completed, "usage")
            if usage is not None:
                input_tokens = int(value(usage, "input_tokens", 0) or 0)
                output_tokens = int(value(usage, "output_tokens", 0) or 0)
                output.usage = Usage(
                    input=input_tokens,
                    output=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
                calculate_cost(output.usage, model)
            output.stop_reason = "toolUse" if calls else "stop"
            event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
        except asyncio.CancelledError:
            output.stop_reason = "aborted"
            output.error_message = None
            event_stream.push(ErrorEvent(reason="aborted", error=output))
        except BaseException as error:
            output.stop_reason = "error"
            output.error_message = error_message(error)
            event_stream.push(ErrorEvent(reason="error", error=output))


def openai_responses_api() -> OpenAIResponsesApi:
    return OpenAIResponsesApi()


__all__ = ["OpenAIResponsesApi", "openai_responses_api"]
