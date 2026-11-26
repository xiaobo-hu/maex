"""Utilities shared by AI providers."""

from .error_body import ProviderErrorDetails, format_provider_error, normalize_provider_error
from .event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)
from .overflow import is_context_overflow
from .provider_retry import retry_provider_request
from .sanitize_unicode import sanitize_surrogates
from .validation import ToolSchema, validate_tool_arguments, validate_tool_call

__all__ = [
    "AssistantMessageEventStream",
    "EventStream",
    "ProviderErrorDetails",
    "create_assistant_message_event_stream",
    "format_provider_error",
    "is_context_overflow",
    "normalize_provider_error",
    "retry_provider_request",
    "sanitize_surrogates",
    "ToolSchema",
    "validate_tool_arguments",
    "validate_tool_call",
]
