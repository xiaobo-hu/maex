"""Official Anthropic Provider using the Messages API."""

from __future__ import annotations

from ..api import anthropic_messages_api
from ..models import Provider, create_provider
from ..types import Model, ModelCost

ANTHROPIC_MODELS: tuple[Model, ...] = (
    Model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        thinking_level_map={"off": "disabled", "low": "low", "medium": "medium", "high": "high"},
        input=["text", "image"],
        cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        context_window=200_000,
        max_tokens=16_000,
        compat={"forceAdaptiveThinking": True},
    ),
    Model(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        api="anthropic-messages",
        provider="anthropic",
        reasoning=True,
        thinking_level_map={"off": "disabled", "low": "low", "medium": "medium", "high": "high"},
        input=["text", "image"],
        cost=ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
        context_window=200_000,
        max_tokens=16_000,
        compat={"forceAdaptiveThinking": True},
    ),
)


def anthropic_provider() -> Provider:
    return create_provider(
        provider_id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        api_key_envs=("ANTHROPIC_API_KEY",),
        models=ANTHROPIC_MODELS,
        api=anthropic_messages_api(),
    )


__all__ = ["ANTHROPIC_MODELS", "anthropic_provider"]
