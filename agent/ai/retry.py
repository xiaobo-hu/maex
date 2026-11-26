"""Retry complete Assistant responses for transient provider failures."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .types import AssistantMessage

_NON_RETRYABLE_RE = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|"
    r"insufficient_quota|out of budget|quota exceeded|billing",
    re.IGNORECASE,
)
_RETRYABLE_RE = re.compile(
    r"overloaded|rate.?limit|too many requests|429|500|502|503|504|524|"
    r"service.?unavailable|server.?error|internal.?error|provider.?returned.?error|"
    r"network.?error|connection.?error|connection.?refused|connection.?lost|"
    r"other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|upstream.?connect|"
    r"reset before headers|socket hang up|socket connection was closed|timed? out|timeout|"
    r"terminated|websocket.?closed|websocket.?error|ended without|"
    r"stream ended before message_stop|stream ended before a terminal response event|"
    r"http2 request did not get a response|retry delay|you can retry your request|"
    r"try your request again|please retry your request|ResourceExhausted",
    re.IGNORECASE,
)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE_RE.search(message.error_message):
        return False
    return bool(_RETRYABLE_RE.search(message.error_message))


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    enabled: bool = False
    max_retries: int = 0
    base_delay_ms: int = 1000


@dataclass(frozen=True, slots=True)
class RetryCallbacks:
    on_retry_scheduled: Callable[[int, int, int, str], Awaitable[None]] | None = None
    on_retry_attempt_start: Callable[[], Awaitable[None]] | None = None
    on_retry_finished: Callable[[bool, int, str | None], Awaitable[None]] | None = None


async def _cancelable_sleep(delay_ms: int, cancel_event: asyncio.Event | None) -> bool:
    if cancel_event is None:
        await asyncio.sleep(delay_ms / 1000)
        return False
    if cancel_event.is_set():
        return True
    sleep_task = asyncio.create_task(asyncio.sleep(delay_ms / 1000))
    cancel_task = asyncio.create_task(cancel_event.wait())
    done, pending = await asyncio.wait(
        {sleep_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return cancel_task in done and cancel_event.is_set()


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    cancel_event: asyncio.Event | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    max_retries = policy.max_retries if policy and policy.enabled else 0
    attempt = 0
    retried = False
    while True:
        response = await produce()
        if response.stop_reason == "aborted":
            if retried and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(False, attempt, None)
            return response
        if response.stop_reason != "error":
            if retried and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(True, attempt, None)
            return response
        if attempt >= max_retries or not is_retryable_assistant_error(response):
            if retried and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(False, attempt, response.error_message)
            return response

        attempt += 1
        retried = True
        error_message = response.error_message or "Unknown error"
        delay_ms = (policy.base_delay_ms if policy else 1000) * 2 ** (attempt - 1)
        if callbacks and callbacks.on_retry_scheduled:
            await callbacks.on_retry_scheduled(attempt, max_retries, delay_ms, error_message)
        try:
            cancelled = await _cancelable_sleep(delay_ms, cancel_event)
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            if callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(False, attempt, error_message)
            response.stop_reason = "aborted"
            response.error_message = None
            return response
        if callbacks and callbacks.on_retry_attempt_start:
            await callbacks.on_retry_attempt_start()


__all__ = [
    "RetryCallbacks",
    "RetryPolicy",
    "is_retryable_assistant_error",
    "retry_assistant_call",
]
