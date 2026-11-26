"""Extract stable, bounded diagnostics from dynamic provider SDK errors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

MAX_ERROR_BODY_LENGTH = 8_000


@dataclass(frozen=True, slots=True)
class ProviderErrorDetails:
    message: str
    status_code: int | None = None
    body: str | None = None


def _stringify_body(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > MAX_ERROR_BODY_LENGTH:
        return text[:MAX_ERROR_BODY_LENGTH] + "…"
    return text


def normalize_provider_error(error: BaseException) -> ProviderErrorDetails:
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", getattr(error, "status", None))
    if not isinstance(status, int) and response is not None:
        response_status = getattr(response, "status_code", getattr(response, "status", None))
        status = response_status if isinstance(response_status, int) else None
    body: object = getattr(error, "body", None)
    if body is None and response is not None:
        body = getattr(response, "text", None)
    if body is None and isinstance(getattr(error, "args", None), tuple):
        for argument in error.args:
            if isinstance(argument, Mapping):
                body = argument
                break
    body_text = _stringify_body(body)
    message = str(error).strip() or error.__class__.__name__
    if body_text and body_text not in message:
        message = f"{message}: {body_text}"
    if isinstance(status, int) and str(status) not in message:
        message = f"HTTP {status}: {message}"
    return ProviderErrorDetails(message=message, status_code=status, body=body_text)


def format_provider_error(error: BaseException) -> str:
    return normalize_provider_error(error).message


__all__ = [
    "MAX_ERROR_BODY_LENGTH",
    "ProviderErrorDetails",
    "format_provider_error",
    "normalize_provider_error",
]
