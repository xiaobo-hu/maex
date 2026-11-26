"""Provider-neutral constrained tool sampling helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .types import GrammarSamplingConfig, JsonSchemaConstrainedSampling, Tool


@dataclass(frozen=True, slots=True)
class GrammarConstrainedSampling:
    format: str
    definition: str
    input_property: str


@dataclass(slots=True)
class GrammarToolInputJsonBuffer:
    input: str = ""
    started: bool = False
    closed: bool = False


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    *,
    close: bool,
) -> str | None:
    """Convert a growing raw grammar input into JSON argument deltas."""
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed after it was closed'
        )
    if not next_input.startswith(buffer.input):
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed non-monotonically'
        )
    input_delta = next_input[len(buffer.input) :]
    if not close and not input_delta:
        return None
    delta = ""
    if not buffer.started:
        delta = "{" + json.dumps(input_property) + ':"'
        buffer.started = True
    delta += json.dumps(input_delta)[1:-1]
    buffer.input = next_input
    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError(
            "grammar constrained sampling requires exactly one required string property"
        )
    input_property = required[0]
    properties = schema.get("properties")
    if not isinstance(properties, dict) or input_property not in properties:
        raise ValueError(
            f"grammar constrained sampling requires a properties entry for {input_property}"
        )
    property_schema = properties[input_property]
    if not isinstance(property_schema, dict) or property_schema.get("type") != "string":
        raise ValueError(
            f"grammar constrained sampling property {input_property} must have type string"
        )
    return input_property


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = tool.constrained_sampling
    if not isinstance(config, JsonSchemaConstrainedSampling):
        return None
    if supports_strict_mode:
        return True
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, '
            "but strict tools are unsupported."
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not isinstance(config, GrammarSamplingConfig):
        return None
    if not supports_openai_grammar_tools:
        raise ValueError(
            f'Tool "{tool.name}" requires grammar constrained sampling, '
            "but grammar tools are unsupported."
        )
    lark = config.variants.openai_lark
    regex = config.variants.openai_regex
    has_lark = isinstance(lark, str) and bool(lark.strip())
    has_regex = isinstance(regex, str) and bool(regex.strip())
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: '
            "no supported grammar variant was provided."
        )
    try:
        input_property = _infer_grammar_input_property(tool)
    except ValueError as error:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.'
        ) from error
    definition = lark if has_lark else regex
    assert isinstance(definition, str)
    return GrammarConstrainedSampling(
        format="lark" if has_lark else "regex",
        definition=definition,
        input_property=input_property,
    )


def create_grammar_tool_input_properties(
    tools: list[Tool] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties


def get_grammar_tool_input(
    tool_name: str,
    arguments: dict[str, object],
    input_property: str,
) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(
            f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.'
        )
    return value


__all__ = [
    "GrammarConstrainedSampling",
    "GrammarToolInputJsonBuffer",
    "append_grammar_tool_input_json_delta",
    "create_grammar_tool_input_properties",
    "get_grammar_tool_input",
    "resolve_grammar_constrained_sampling",
    "resolve_json_schema_strict_sampling",
]
