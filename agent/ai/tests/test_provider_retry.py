"""Verify retries at the provider request and streaming transport boundary."""

from __future__ import annotations

import pytest

from ai.utils.provider_retry import retry_provider_stream


class _ImmediateConnectionError(ConnectionError):
    headers = {"retry-after-ms": "0"}


@pytest.mark.asyncio
async def test_provider_stream_retries_connection_error_before_first_chunk() -> None:
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1

        async def response():
            if attempts == 1:
                raise _ImmediateConnectionError("Connection error.")
            yield "complete"

        return response()

    chunks = [
        chunk
        async for chunk in retry_provider_stream(
            request,
            max_retries=1,
            max_retry_delay_ms=1,
        )
    ]

    assert chunks == ["complete"]
    assert attempts == 2


@pytest.mark.asyncio
async def test_provider_stream_does_not_retry_after_a_chunk_was_emitted() -> None:
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1

        async def response():
            yield "partial"
            raise ConnectionError("Connection error.")

        return response()

    received: list[str] = []
    with pytest.raises(ConnectionError, match="Connection error"):
        async for chunk in retry_provider_stream(request, max_retries=2):
            received.append(chunk)

    assert received == ["partial"]
    assert attempts == 1


@pytest.mark.asyncio
async def test_provider_stream_retries_after_buffered_partial_chunk() -> None:
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1

        async def response():
            if attempts == 1:
                yield "discarded-partial"
                raise _ImmediateConnectionError("Connection error.")
            yield "complete"

        return response()

    chunks = [
        chunk
        async for chunk in retry_provider_stream(
            request,
            max_retries=1,
            max_retry_delay_ms=1,
            is_committed=lambda _chunk: False,
        )
    ]

    assert chunks == ["complete"]
    assert attempts == 2


@pytest.mark.asyncio
async def test_provider_stream_retries_after_uncommitted_protocol_chunk() -> None:
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1

        async def response():
            yield "role"
            if attempts == 1:
                raise _ImmediateConnectionError("Connection error.")
            yield "complete"

        return response()

    chunks = [
        chunk
        async for chunk in retry_provider_stream(
            request,
            max_retries=1,
            max_retry_delay_ms=1,
            is_committed=lambda chunk: chunk != "role",
        )
    ]

    assert chunks == ["role", "complete"]
    assert attempts == 2
