"""Instance-scoped scripted provider adapted from Pi 0.83 and pi-py."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias
from uuid import uuid4

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
from ..models import Provider, ProviderStreams, create_provider
from ..types import (
    AssistantContent,
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    now_ms,
)
from ..utils.event_stream import AssistantMessageEventStream


@dataclass(slots=True)
class FauxState:
    call_count: int = 0


FauxResponseFactory: TypeAlias = Callable[
    [Context, StreamOptions | None, FauxState, Model],
    AssistantMessage | Awaitable[AssistantMessage],
]
FauxResponseStep: TypeAlias = AssistantMessage | FauxResponseFactory


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, object],
    *,
    tool_call_id: str | None = None,
) -> ToolCall:
    return ToolCall(id=tool_call_id or f"tool:{uuid4()}", name=name, arguments=arguments)


def faux_assistant_message(
    content: str | AssistantContent | list[AssistantContent],
    *,
    stop_reason: StopReason = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
) -> AssistantMessage:
    if isinstance(content, str):
        blocks: list[AssistantContent] = [TextContent(text=content)]
    elif isinstance(content, list):
        blocks = content
    else:
        blocks = [content]
    return AssistantMessage(
        content=blocks,
        api="faux",
        provider="faux",
        model="faux-1",
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
    )


class _FauxStreams(ProviderStreams):
    def __init__(
        self, responses: deque[FauxResponseStep], state: FauxState, chunk_size: int
    ) -> None:
        self._responses = responses
        self._state = state
        self._chunk_size = max(1, chunk_size)

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        step = self._responses.popleft() if self._responses else None
        self._state.call_count += 1
        asyncio.create_task(self._produce(stream, step, model, context, options))
        return stream

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self.stream(model, context, options)

    async def _produce(
        self,
        stream: AssistantMessageEventStream,
        step: FauxResponseStep | None,
        model: Model,
        context: Context,
        options: StreamOptions | None,
    ) -> None:
        try:
            if step is None:
                raise RuntimeError("No more faux responses queued")
            if options and options.cancel_event and options.cancel_event.is_set():
                raise asyncio.CancelledError
            candidate = step(context, options, self._state, model) if callable(step) else step
            if inspect.isawaitable(candidate):
                candidate_task = asyncio.ensure_future(candidate)
                if options and options.cancel_event:
                    cancel_task = asyncio.create_task(options.cancel_event.wait())
                    done, _ = await asyncio.wait(
                        {candidate_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if cancel_task in done and options.cancel_event.is_set():
                        candidate_task.cancel()
                        await asyncio.gather(candidate_task, return_exceptions=True)
                        raise asyncio.CancelledError
                    cancel_task.cancel()
                    await asyncio.gather(cancel_task, return_exceptions=True)
                message = await candidate_task
            else:
                message = candidate
            final = message.model_copy(deep=True)
            final.api = model.api
            final.provider = model.provider
            final.model = model.id
            final.timestamp = now_ms()
            await self._emit_message(stream, final, options.cancel_event if options else None)
        except asyncio.CancelledError:
            failed = AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason="aborted",
            )
            stream.push(ErrorEvent(reason="aborted", error=failed))
        except Exception as error:
            failed = AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason="error",
                error_message=str(error),
            )
            stream.push(ErrorEvent(reason="error", error=failed))

    def _chunks(self, text: str) -> Iterable[str]:
        if text == "":
            return ("",)
        return (
            text[index : index + self._chunk_size]
            for index in range(0, len(text), self._chunk_size)
        )

    async def _emit_message(
        self,
        stream: AssistantMessageEventStream,
        message: AssistantMessage,
        cancel_event: asyncio.Event | None,
    ) -> None:
        partial = message.model_copy(deep=True)
        partial.content = []
        partial.stop_reason = "pending"
        stream.push(StartEvent(partial=partial.model_copy(deep=True)))

        for index, block in enumerate(message.content):
            await asyncio.sleep(0)
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            if isinstance(block, TextContent):
                current = TextContent(text="")
                partial.content.append(current)
                stream.push(
                    TextStartEvent(content_index=index, partial=partial.model_copy(deep=True))
                )
                for chunk in self._chunks(block.text):
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError
                    current.text += chunk
                    stream.push(
                        TextDeltaEvent(
                            content_index=index,
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                stream.push(
                    TextEndEvent(
                        content_index=index,
                        content=block.text,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif isinstance(block, ThinkingContent):
                current_thinking = ThinkingContent(thinking="")
                partial.content.append(current_thinking)
                stream.push(
                    ThinkingStartEvent(content_index=index, partial=partial.model_copy(deep=True))
                )
                for chunk in self._chunks(block.thinking):
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError
                    current_thinking.thinking += chunk
                    stream.push(
                        ThinkingDeltaEvent(
                            content_index=index,
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                stream.push(
                    ThinkingEndEvent(
                        content_index=index,
                        content=block.thinking,
                        partial=partial.model_copy(deep=True),
                    )
                )
            else:
                current_tool_call = ToolCall(id=block.id, name=block.name, arguments={})
                partial.content.append(current_tool_call)
                stream.push(
                    ToolCallStartEvent(content_index=index, partial=partial.model_copy(deep=True))
                )
                raw_arguments = json.dumps(
                    block.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for chunk in self._chunks(raw_arguments):
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError
                    stream.push(
                        ToolCallDeltaEvent(
                            content_index=index,
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                current_tool_call.arguments = dict(block.arguments)
                stream.push(
                    ToolCallEndEvent(
                        content_index=index,
                        tool_call=block.model_copy(deep=True),
                        partial=partial.model_copy(deep=True),
                    )
                )

        if message.stop_reason == "pending":
            raise RuntimeError("Faux response ended without a stop reason")
        if message.stop_reason in {"error", "aborted"}:
            stream.push(ErrorEvent(reason=message.stop_reason, error=message))
        else:
            stream.push(DoneEvent(reason=message.stop_reason, message=message))


class FauxProviderHandle:
    def __init__(
        self,
        provider: Provider,
        models: tuple[Model, ...],
        responses: deque[FauxResponseStep],
        state: FauxState,
    ) -> None:
        self.provider = provider
        self.models = tuple(model.model_copy(deep=True) for model in models)
        self.state = state
        self._responses = responses

    def get_model(self, model_id: str | None = None) -> Model | None:
        requested = model_id or self.models[0].id
        model = next((item for item in self.models if item.id == requested), None)
        return model.model_copy(deep=True) if model is not None else None

    def set_responses(self, responses: Iterable[FauxResponseStep]) -> None:
        self._responses.clear()
        self._responses.extend(responses)

    def append_responses(self, responses: Iterable[FauxResponseStep]) -> None:
        self._responses.extend(responses)

    @property
    def pending_response_count(self) -> int:
        return len(self._responses)


def faux_provider(*, chunk_size: int = 8) -> FauxProviderHandle:
    model = Model(
        id="faux-1",
        name="Faux Model",
        api="faux",
        provider="faux",
        base_url="http://localhost:0",
        reasoning=True,
        input=["text", "image"],
        context_window=128_000,
        max_tokens=16_384,
    )
    responses: deque[FauxResponseStep] = deque()
    state = FauxState()
    streams = _FauxStreams(responses, state, chunk_size)
    provider = create_provider(
        provider_id="faux",
        name="Faux",
        models=[model],
        streams=streams,
    )
    return FauxProviderHandle(provider, provider.models, responses, state)


__all__ = [
    "FauxProviderHandle",
    "FauxResponseFactory",
    "FauxResponseStep",
    "FauxState",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
]
