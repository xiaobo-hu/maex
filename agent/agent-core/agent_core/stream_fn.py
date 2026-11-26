"""Optional default StreamFn configuration, matching Pi's stream-fn module."""

from __future__ import annotations

from .types import StreamFn

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call "
            "set_default_stream_fn()."
        )
    return _default_stream_fn


__all__ = ["get_default_stream_fn", "set_default_stream_fn"]
