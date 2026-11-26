"""工具参数转换与验证，对应 Pi AI 层的 validation 工具。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ..types import ToolCall


class ToolSchema(Protocol):
    """可由 AI Tool 和 AgentTool 共同满足的最小 Schema 协议。"""

    name: str
    parameters: dict[str, object]


def _schema_types(schema: dict[str, object]) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [item for item in schema_type if isinstance(item, str)]
    return []


def _matches_json_type(value: object, schema_type: str) -> bool:
    if schema_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "null":
        return value is None
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return False


def _coerce_primitive(value: object, schema_type: str) -> object:
    if schema_type in {"number", "integer"}:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
            except ValueError:
                return value
            if not math.isfinite(parsed):
                return value
            if schema_type == "integer":
                return int(parsed) if parsed.is_integer() else value
            return int(parsed) if parsed.is_integer() else parsed
        return value
    if schema_type == "boolean":
        if value is None:
            return False
        if value == "true" or value == 1:
            return True
        if value == "false" or value == 0:
            return False
        return value
    if schema_type == "string":
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        return value
    if schema_type == "null" and (value == "" or value == 0 or value is False):
        return None
    return value


def _is_valid(value: object, schema: dict[str, object]) -> bool:
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError:
        return False
    return True


def _coerce_union(value: object, schemas: list[object]) -> object:
    for item in schemas:
        if not isinstance(item, dict):
            continue
        schema = cast(dict[str, object], item)
        candidate = _coerce_with_schema(deepcopy(value), schema)
        if _is_valid(candidate, schema):
            return candidate
    return value


def _coerce_object(value: dict[str, object], schema: dict[str, object]) -> None:
    properties = schema.get("properties")
    defined_keys: set[str] = set()
    if isinstance(properties, dict):
        defined_keys = {key for key in properties if isinstance(key, str)}
        for key, property_schema in properties.items():
            if key in value and isinstance(property_schema, dict):
                value[key] = _coerce_with_schema(
                    value[key], cast(dict[str, object], property_schema)
                )

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        additional_schema = cast(dict[str, object], additional)
        for key in value.keys() - defined_keys:
            value[key] = _coerce_with_schema(value[key], additional_schema)


def _coerce_array(value: list[object], schema: dict[str, object]) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index, item_schema in enumerate(items[: len(value)]):
            if isinstance(item_schema, dict):
                value[index] = _coerce_with_schema(
                    value[index], cast(dict[str, object], item_schema)
                )
        return
    if isinstance(items, dict):
        item_schema = cast(dict[str, object], items)
        for index, item in enumerate(value):
            value[index] = _coerce_with_schema(item, item_schema)


def _coerce_with_schema(value: object, schema: dict[str, object]) -> object:
    next_value = value

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            if isinstance(nested, dict):
                next_value = _coerce_with_schema(next_value, cast(dict[str, object], nested))

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_union(next_value, any_of)
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_union(next_value, one_of)

    schema_types = _schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, schema_type) for schema_type in schema_types
    )
    if schema_types and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive(next_value, schema_type)
            if candidate is not next_value and (
                type(candidate) is not type(next_value) or candidate != next_value
            ):
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _coerce_object(next_value, schema)
    if "array" in schema_types and isinstance(next_value, list):
        _coerce_array(next_value, schema)
    return next_value


def _format_path(error: ValidationError) -> str:
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "required" and isinstance(error.validator_value, list):
        missing = next(
            (
                item
                for item in error.validator_value
                if isinstance(item, str)
                and isinstance(error.instance, dict)
                and item not in error.instance
            ),
            None,
        )
        if missing is not None:
            parts.append(missing)
    return ".".join(parts) or "root"


def validate_tool_arguments(tool: ToolSchema, tool_call: ToolCall) -> dict[str, object]:
    """按工具 JSON Schema 转换并验证参数，不修改原始 ToolCall。"""

    Draft202012Validator.check_schema(tool.parameters)
    original = deepcopy(tool_call.arguments)
    coerced = _coerce_with_schema(deepcopy(original), tool.parameters)
    validator = Draft202012Validator(tool.parameters)
    errors = sorted(
        validator.iter_errors(coerced),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "\n".join(f"  - {_format_path(error)}: {error.message}" for error in errors)
        received = json.dumps(original, ensure_ascii=False, indent=2)
        raise ValueError(
            f'Validation failed for tool "{tool_call.name}":\n{details}'
            f"\n\nReceived arguments:\n{received}"
        )
    if not isinstance(coerced, dict):
        raise ValueError(f'Tool "{tool_call.name}" arguments must be an object')
    return coerced


def validate_tool_call(tools: list[ToolSchema], tool_call: ToolCall) -> dict[str, object]:
    """查找工具并验证一次模型工具调用。"""

    tool = next((candidate for candidate in tools if candidate.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


__all__ = ["ToolSchema", "validate_tool_arguments", "validate_tool_call"]
