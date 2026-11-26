from __future__ import annotations

from copy import deepcopy

import pytest
from jsonschema.exceptions import SchemaError

from ai import Tool, ToolCall, validate_tool_arguments, validate_tool_call


def test_validation_coerces_nested_values_without_mutating_tool_call() -> None:
    tool = Tool(
        name="sample",
        description="Validate nested values",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "numeric_flag": {"type": "integer"},
                "boolean_flag": {"type": "boolean"},
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": [
                "count",
                "enabled",
                "numeric_flag",
                "boolean_flag",
                "metadata",
                "items",
            ],
            "additionalProperties": False,
        },
    )
    call = ToolCall(
        id="call-1",
        name="sample",
        arguments={
            "count": "2",
            "enabled": "false",
            "numeric_flag": True,
            "boolean_flag": 1,
            "metadata": {"ratio": "1.5"},
            "items": ["3", 4],
        },
    )
    original = deepcopy(call.arguments)

    validated = validate_tool_arguments(tool, call)

    assert validated == {
        "count": 2,
        "enabled": False,
        "numeric_flag": 1,
        "boolean_flag": True,
        "metadata": {"ratio": 1.5},
        "items": [3, 4],
    }
    assert call.arguments == original


def test_validation_chooses_a_valid_union_branch() -> None:
    tool = Tool(
        name="union",
        description="Validate a union",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "string", "minLength": 3},
                    ]
                }
            },
            "required": ["value"],
        },
    )

    validated = validate_tool_arguments(
        tool,
        ToolCall(id="call-1", name="union", arguments={"value": "2"}),
    )

    assert validated == {"value": 2}


def test_validation_error_contains_tool_path_and_original_arguments() -> None:
    tool = Tool(
        name="sample",
        description="Validate input",
        parameters={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                }
            },
            "required": ["payload"],
        },
    )

    with pytest.raises(ValueError) as caught:
        validate_tool_arguments(
            tool,
            ToolCall(id="call-1", name="sample", arguments={"payload": {"count": "no"}}),
        )

    message = str(caught.value)
    assert 'Validation failed for tool "sample"' in message
    assert "payload.count" in message
    assert "Received arguments" in message
    assert '"count": "no"' in message


def test_validate_tool_call_rejects_unknown_tool() -> None:
    call = ToolCall(id="call-1", name="missing", arguments={})

    with pytest.raises(ValueError, match='Tool "missing" not found'):
        validate_tool_call([], call)


def test_validation_rejects_an_invalid_schema() -> None:
    tool = Tool(
        name="invalid",
        description="Invalid schema",
        parameters={"type": "unknown"},
    )

    with pytest.raises(SchemaError):
        validate_tool_arguments(tool, ToolCall(id="call-1", name="invalid", arguments={}))
