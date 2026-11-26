"""Official OpenAI Provider using the Responses API."""

from __future__ import annotations

from ..api import openai_responses_api
from ..models import Provider, create_provider
from ..types import Model, ModelCost

OPENAI_MODELS: tuple[Model, ...] = (
    Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=2.5, output=10.0, cache_read=1.25),
        context_window=128_000,
        max_tokens=16_384,
    ),
    Model(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        api="openai-responses",
        provider="openai",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(input=0.15, output=0.6, cache_read=0.075),
        context_window=128_000,
        max_tokens=16_384,
    ),
)


def openai_provider() -> Provider:
    return create_provider(
        provider_id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_envs=("OPENAI_API_KEY",),
        models=OPENAI_MODELS,
        api=openai_responses_api(),
    )


__all__ = ["OPENAI_MODELS", "openai_provider"]
