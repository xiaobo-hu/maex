"""Factories for custom OpenAI Chat Completions-compatible services."""

from __future__ import annotations

from collections.abc import Iterable

from ..api import openai_completions_api
from ..models import Provider, create_provider, normalize_base_url
from ..types import Model, ModelCost


def create_openai_compatible_model(
    *,
    model_id: str,
    provider_id: str,
    base_url: str,
    name: str | None = None,
    max_tokens: int = 8192,
    context_window: int = 128_000,
    reasoning: bool = False,
    thinking_format: str | None = None,
    cost: ModelCost | None = None,
) -> Model:
    compat: dict[str, object] | None = None
    thinking_level_map: dict[str, str | None] | None = None
    if thinking_format:
        compat = {"thinkingFormat": thinking_format}
        thinking_level_map = {"off": "disabled"}
    return Model(
        id=model_id,
        name=name or model_id,
        api="openai-completions",
        provider=provider_id,
        base_url=normalize_base_url(base_url),
        reasoning=reasoning,
        thinking_level_map=thinking_level_map,
        input=["text", "image"],
        cost=cost or ModelCost(),
        context_window=context_window,
        max_tokens=max_tokens,
        compat=compat,
    )


def openai_compatible_provider(
    *,
    provider_id: str,
    base_url: str,
    models: Iterable[Model],
    name: str | None = None,
    api_key_envs: Iterable[str] = ("LLM_API_KEY", "OPENAI_API_KEY"),
    headers: dict[str, str] | None = None,
) -> Provider:
    return create_provider(
        provider_id=provider_id,
        name=name or provider_id,
        base_url=normalize_base_url(base_url),
        api_key_envs=tuple(api_key_envs),
        headers=headers,
        models=tuple(models),
        api=openai_completions_api(),
    )


__all__ = ["create_openai_compatible_model", "openai_compatible_provider"]
