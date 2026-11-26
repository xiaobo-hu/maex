"""Built-in provider factories with lazy optional SDK imports."""

from .faux import (
    FauxProviderHandle,
    FauxResponseFactory,
    FauxResponseStep,
    FauxState,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)

__all__ = [
    "ANTHROPIC_MODELS",
    "FauxProviderHandle",
    "FauxResponseFactory",
    "FauxResponseStep",
    "FauxState",
    "OPENAI_MODELS",
    "anthropic_provider",
    "create_openai_compatible_model",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "openai_compatible_provider",
    "openai_provider",
]


def __getattr__(name: str) -> object:
    """Load provider SDK modules only when that provider is requested."""
    if name in {"ANTHROPIC_MODELS", "anthropic_provider"}:
        from .anthropic import ANTHROPIC_MODELS, anthropic_provider

        return {"ANTHROPIC_MODELS": ANTHROPIC_MODELS, "anthropic_provider": anthropic_provider}[name]
    if name in {"OPENAI_MODELS", "openai_provider"}:
        from .openai import OPENAI_MODELS, openai_provider

        return {"OPENAI_MODELS": OPENAI_MODELS, "openai_provider": openai_provider}[name]
    if name in {"create_openai_compatible_model", "openai_compatible_provider"}:
        from .openai_compatible import create_openai_compatible_model, openai_compatible_provider

        return {
            "create_openai_compatible_model": create_openai_compatible_model,
            "openai_compatible_provider": openai_compatible_provider,
        }[name]
    raise AttributeError(name)
