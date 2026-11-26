"""Instance-scoped Provider and model dispatch following Pi 0.83."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, TypeVar

from .events import ErrorEvent
from .types import (
    AssistantMessage,
    Context,
    Model,
    ModelCostRates,
    ModelThinkingLevel,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    UsageCost,
)
from .utils.event_stream import AssistantMessageEventStream


class ProviderStreams(Protocol):
    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...


ApiImplementations = ProviderStreams | dict[str, ProviderStreams]
TOptions = TypeVar("TOptions", bound=StreamOptions)


def normalize_base_url(base_url: str) -> str:
    """Append ``/v1`` only when a URL contains no path component."""
    from urllib.parse import urlsplit, urlunsplit

    value = base_url.rstrip("/")
    parts = urlsplit(value)
    if parts.scheme and parts.netloc and parts.path in {"", "/"}:
        return urlunsplit((parts.scheme, parts.netloc, "/v1", parts.query, parts.fragment))
    return value


def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost = UsageCost(
        input=rates.input * usage.input / 1_000_000,
        output=rates.output * usage.output / 1_000_000,
        cache_read=rates.cache_read * usage.cache_read / 1_000_000,
        cache_write=(rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000,
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
    return usage.cost


_THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def get_supported_thinking_levels(model: Model) -> tuple[ModelThinkingLevel, ...]:
    if not model.reasoning:
        return ("off",)
    mapping = model.thinking_level_map or {}
    supported: list[ModelThinkingLevel] = []
    for level in _THINKING_LEVELS:
        if level in mapping and mapping[level] is None:
            continue
        if level in {"xhigh", "max"} and level not in mapping:
            continue
        supported.append(level)
    return tuple(supported)


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    requested = _THINKING_LEVELS.index(level)
    for candidate in _THINKING_LEVELS[requested:]:
        if candidate in available:
            return candidate
    for candidate in reversed(_THINKING_LEVELS[:requested]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(first: Model | None, second: Model | None) -> bool:
    return bool(
        first is not None
        and second is not None
        and first.id == second.id
        and first.provider == second.provider
    )


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    name: str
    models: tuple[Model, ...]
    api: ApiImplementations
    base_url: str = ""
    headers: dict[str, str] | None = None
    api_key_envs: tuple[str, ...] = ()

    def get_models(self) -> tuple[Model, ...]:
        return tuple(self._resolved_model(model) for model in self.models)

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        resolved_model = self._resolved_model(model)
        resolved_options = self._resolved_options(options, StreamOptions)
        return self._api_for(model).stream(resolved_model, context, resolved_options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        resolved_model = self._resolved_model(model)
        resolved_options = self._resolved_options(options, SimpleStreamOptions)
        return self._api_for(model).stream_simple(resolved_model, context, resolved_options)

    def _api_for(self, model: Model) -> ProviderStreams:
        if not isinstance(self.api, dict):
            return self.api
        implementation = self.api.get(model.api)
        if implementation is None:
            raise ValueError(f'Provider {self.id} has no API implementation for "{model.api}"')
        return implementation

    def _resolved_model(self, model: Model) -> Model:
        update: dict[str, object] = {}
        if not model.base_url and self.base_url:
            update["base_url"] = self.base_url
        merged_headers = dict(self.headers or {})
        merged_headers.update(model.headers or {})
        if merged_headers:
            update["headers"] = merged_headers
        return model.model_copy(update=update, deep=True)

    def _resolved_options(self, options: TOptions | None, option_type: type[TOptions]) -> TOptions:
        resolved = (
            options.model_copy(update={"cancel_event": None, "http_client": None}).model_copy(
                deep=True
            )
            if options is not None
            else option_type()
        )
        if options is not None:
            # Runtime synchronization/client objects must retain identity and are not serializable.
            resolved.cancel_event = options.cancel_event
            resolved.http_client = options.http_client
        if resolved.api_key is None:
            environment = resolved.env or {}
            resolved.api_key = next(
                (
                    value
                    for name in self.api_key_envs
                    if (value := environment.get(name) or os.environ.get(name))
                ),
                None,
            )
        merged_headers: dict[str, str | None] = dict(self.headers or {})
        for key, value in (resolved.headers or {}).items():
            if value is None:
                merged_headers.pop(key, None)
            else:
                merged_headers[key] = value
        resolved.headers = merged_headers or None
        return resolved


def create_provider(
    *,
    provider_id: str,
    name: str | None = None,
    models: list[Model] | tuple[Model, ...],
    api: ApiImplementations | None = None,
    streams: ProviderStreams | None = None,
    base_url: str = "",
    headers: dict[str, str] | None = None,
    api_key_envs: list[str] | tuple[str, ...] = (),
) -> Provider:
    implementation = api or streams
    if implementation is None:
        raise ValueError("create_provider requires api or streams")
    normalized = tuple(model.model_copy(deep=True) for model in models)
    for model in normalized:
        if model.provider != provider_id:
            raise ValueError(
                f'Model "{model.id}" belongs to provider "{model.provider}", '
                f'expected "{provider_id}"'
            )
    return Provider(
        id=provider_id,
        name=name or provider_id,
        models=normalized,
        api=implementation,
        base_url=base_url.rstrip("/"),
        headers=dict(headers) if headers else None,
        api_key_envs=tuple(api_key_envs),
    )


def _error_stream(model: Model, message: str) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    failed = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="error",
        error_message=message,
    )
    stream.push(ErrorEvent(reason="error", error=failed))
    return stream


class Models:
    """Mutable provider collection; each instance owns all registration state."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def set_provider(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def get_providers(self) -> tuple[Provider, ...]:
        return tuple(self._providers.values())

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        return next((item for item in provider.get_models() if item.id == model_id), None)

    def get_models(self, provider_id: str | None = None) -> tuple[Model, ...]:
        providers = (
            tuple(self._providers.values())
            if provider_id is None
            else tuple(
                provider for provider in [self._providers.get(provider_id)] if provider is not None
            )
        )
        return tuple(model for provider in providers for model in provider.get_models())

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        provider = self._resolve_provider(model)
        if isinstance(provider, str):
            return _error_stream(model, provider)
        try:
            return provider.stream(model, context, options)
        except Exception as error:
            return _error_stream(model, str(error))

    async def complete(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessage:
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        provider = self._resolve_provider(model)
        if isinstance(provider, str):
            return _error_stream(model, provider)
        try:
            return provider.stream_simple(model, context, options)
        except Exception as error:
            return _error_stream(model, str(error))

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options).result()

    def _resolve_provider(self, model: Model) -> Provider | str:
        provider = self._providers.get(model.provider)
        if provider is None:
            return f"Unknown provider: {model.provider}"
        if not any(candidate.id == model.id for candidate in provider.models):
            return f"Unknown model for provider {model.provider}: {model.id}"
        return provider


def create_models() -> Models:
    return Models()


__all__ = [
    "ApiImplementations",
    "Models",
    "Provider",
    "ProviderStreams",
    "calculate_cost",
    "clamp_thinking_level",
    "create_models",
    "create_provider",
    "get_supported_thinking_levels",
    "models_are_equal",
    "normalize_base_url",
]
