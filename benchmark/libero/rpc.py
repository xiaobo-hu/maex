"""Small synchronous MsgPack RPC used across the Python 3.8 simulator boundary."""

from __future__ import annotations

from ._vendor.openpi_client import msgpack_numpy


class RpcError(RuntimeError):
    """Raised when the remote LIBERO environment rejects a request."""


def pack_message(message: dict[str, object]) -> bytes:
    """Encode one RPC message with NumPy support."""
    return msgpack_numpy.packb(message)


def unpack_message(data: object) -> dict[str, object]:
    """Decode and validate one binary RPC message."""
    if isinstance(data, str):
        raise RpcError("RPC messages must be binary")
    message = msgpack_numpy.unpackb(data)
    if not isinstance(message, dict):
        raise RpcError("RPC message must be a mapping")
    return message
