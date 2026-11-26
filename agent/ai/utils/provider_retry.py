"""Bounded retries for provider SDK requests."""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import random
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from typing import TypeVar, cast

T = TypeVar("T")
DEFAULT_MAX_RETRY_DELAY_MS = 60_000


def _headers(error: BaseException) -> Mapping[str, str]:
    headers = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", None)
    return headers if isinstance(headers, Mapping) else {}


def _status(error: BaseException) -> int | None:
    value = getattr(error, "status_code", getattr(error, "status", None))
    return value if isinstance(value, int) else None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((str(value) for key, value in headers.items() if key.lower() == name), None)


def _is_retryable(error: BaseException) -> bool:
    headers = _headers(error)
    directive = _header(headers, "x-should-retry")
    if directive in {"true", "false"}:
        return directive == "true"
    status = _status(error)
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    error_name = type(error).__name__.lower()
    return isinstance(error, (ConnectionError, TimeoutError)) or any(
        marker in error_name for marker in ("connection", "timeout")
    )


def _retry_delay_ms(error: BaseException, retry_index: int, maximum: int | None) -> float:
    headers = _headers(error)
    delay: float | None = None
    retry_after_ms = _header(headers, "retry-after-ms")
    if retry_after_ms is not None:
        with contextlib.suppress(ValueError):
            delay = float(retry_after_ms)
    if delay is None and (retry_after := _header(headers, "retry-after")) is not None:
        try:
            delay = float(retry_after) * 1000
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(retry_after)
            delay = parsed.timestamp() * 1000 - time.time() * 1000
    if delay is None:
        delay = min(0.5 * (2**retry_index), 8) * 1000 * (1 - random.random() * 0.25)
    maximum_delay = DEFAULT_MAX_RETRY_DELAY_MS if maximum is None else maximum
    if maximum_delay > 0 and delay > maximum_delay:
        raise RuntimeError(
            f"Server requested {int((delay + 999) // 1000)}s retry delay "
            f"(max: {int((maximum_delay + 999) // 1000)}s). {error}"
        )
    return max(0.0, delay)


async def _wait_before_retry(
    error: BaseException,
    retry_index: int,
    max_retry_delay_ms: int | None,
    cancel_event: asyncio.Event | None,
) -> None:
    delay = _retry_delay_ms(error, retry_index, max_retry_delay_ms) / 1000
    if cancel_event is None:
        await asyncio.sleep(delay)
        return
    delay_task = asyncio.create_task(asyncio.sleep(delay))
    cancel_task = asyncio.create_task(cancel_event.wait())
    done, pending = await asyncio.wait(
        {delay_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if cancel_task in done and cancel_event.is_set():
        raise asyncio.CancelledError from error


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 0,
    max_retry_delay_ms: int | None = None,
    cancel_event: asyncio.Event | None = None,
) -> T:
    retries_remaining = max_retries
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        try:
            request_task: asyncio.Future[T] = asyncio.ensure_future(request())
            if cancel_event is None:
                return await request_task
            cancel_task = asyncio.create_task(cancel_event.wait())
            waiters = {
                cast(asyncio.Future[object], request_task),
                cast(asyncio.Future[object], cancel_task),
            }
            done, pending = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done and cancel_event.is_set():
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                raise asyncio.CancelledError from None
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            return request_task.result()
        except Exception as error:
            if retries_remaining <= 0 or not _is_retryable(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _wait_before_retry(
                error,
                retry_index,
                max_retry_delay_ms,
                cancel_event,
            )


async def retry_provider_stream(
    request: Callable[[], Awaitable[AsyncIterable[T]]],
    *,
    max_retries: int = 0,
    max_retry_delay_ms: int | None = None,
    cancel_event: asyncio.Event | None = None,
    is_committed: Callable[[T], bool] | None = None,
) -> AsyncIterator[T]:
    """Retry transport failures before a response commits model output."""
    retries_remaining = max_retries
    while True:
        committed = False
        buffered: list[T] = []
        try:
            response = await request()
            async for chunk in response:
                if committed:
                    yield chunk
                    continue
                buffered.append(chunk)
                if is_committed is None or is_committed(chunk):
                    committed = True
                    for pending in buffered:
                        yield pending
                    buffered.clear()
            for pending in buffered:
                yield pending
            return
        except Exception as error:
            if committed or retries_remaining <= 0 or not _is_retryable(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _wait_before_retry(
                error,
                retry_index,
                max_retry_delay_ms,
                cancel_event,
            )


__all__ = ["retry_provider_request", "retry_provider_stream"]
