"""Shared helpers kept private to provider protocol adapters."""

from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json

from ..models import calculate_cost as calculate_model_cost
from ..types import Model, Usage
from ..utils.error_body import format_provider_error


def parse_streaming_json(value: str) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = repair_json(value, return_objects=True)
    return parsed if isinstance(parsed, dict) else {}


def value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def calculate_cost(usage: Usage, model: Model) -> None:
    calculate_model_cost(model, usage)


def error_message(error: BaseException) -> str:
    return format_provider_error(error)


__all__ = ["calculate_cost", "error_message", "parse_streaming_json", "value"]
