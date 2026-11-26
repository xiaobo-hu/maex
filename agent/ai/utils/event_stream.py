"""Strict asyncio event streams following Pi's terminal-event contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar, cast

from ..events import AssistantMessageEvent, DoneEvent, ErrorEvent
from ..types import AssistantMessage

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")
_SENTINEL = object()


class EventStream(Generic[TEvent, TResult], AsyncIterator[TEvent]):
    def __init__(
        self,
        is_terminal: Callable[[TEvent], bool],
        extract_result: Callable[[TEvent], TResult],
    ) -> None:
        self._queue: asyncio.Queue[TEvent | object] = asyncio.Queue()
        self._result: asyncio.Future[TResult] = asyncio.get_running_loop().create_future()
        self._is_terminal = is_terminal
        self._extract_result = extract_result
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def push(self, event: TEvent) -> None:
        if self._ended:
            return
        self._queue.put_nowait(event)
        if self._is_terminal(event):
            self._ended = True
            self._result.set_result(self._extract_result(event))
            self._queue.put_nowait(_SENTINEL)

    def end(self, result: TResult) -> None:
        if self._ended:
            return
        self._ended = True
        self._result.set_result(result)
        self._queue.put_nowait(_SENTINEL)

    def fail(self, error: BaseException) -> None:
        if self._ended:
            return
        self._ended = True
        self._result.set_exception(error)
        self._queue.put_nowait(_SENTINEL)

    async def result(self) -> TResult:
        return await self._result

    def __aiter__(self) -> AsyncIterator[TEvent]:
        return self

    async def __anext__(self) -> TEvent:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return cast(TEvent, item)


def _extract_assistant_result(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise ValueError(f"Event is not terminal: {event.type}")


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            lambda event: isinstance(event, (DoneEvent, ErrorEvent)),
            _extract_assistant_result,
        )


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()


__all__ = [
    "AssistantMessageEventStream",
    "EventStream",
    "create_assistant_message_event_stream",
]
