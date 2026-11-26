"""Pi 0.83 assistant streaming event protocol."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .types import AssistantMessage, ToolCall


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StartEvent(_Event):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(_Event):
    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class TextDeltaEvent(_Event):
    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class TextEndEvent(_Event):
    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(_Event):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ThinkingDeltaEvent(_Event):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(_Event):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(_Event):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ToolCallDeltaEvent(_Event):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(_Event):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")
    tool_call: ToolCall = Field(alias="toolCall")
    partial: AssistantMessage


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent: TypeAlias = Annotated[
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


__all__ = [
    "AssistantMessageEvent",
    "DoneEvent",
    "ErrorEvent",
    "StartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "TextStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ThinkingStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
]
