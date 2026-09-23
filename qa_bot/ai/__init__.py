from qa_bot.config import MAX_CONCURRENT_API_CALLS, is_openrouter_model

from .base import AIProvider, AgentAction, FatalProviderError, WorkerActionResponse
from .claude_provider import ClaudeProvider
from .openrouter_provider import OpenRouterProvider

__all__ = [
    "AIProvider", "AgentAction", "WorkerActionResponse", "FatalProviderError",
    "ClaudeProvider", "OpenRouterProvider", "MissingAPIKeyError", "create_provider",
]


class MissingAPIKeyError(ValueError):
    """The chosen model's provider has no API key. ``env_var`` names the one
    to set, so a caller can add its own hint (the CLI names its flag)."""

    def __init__(self, model: str, env_var: str, provider: str):
        super().__init__(f"{env_var} is not set — model '{model}' runs on {provider}")
        self.env_var = env_var


def create_provider(
    model: str,
    *,
    anthropic_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    max_concurrent_calls: int = MAX_CONCURRENT_API_CALLS,
) -> AIProvider:
    """The provider for ``model``: OpenRouter for "vendor/slug" ids
    (openai/gpt-6-luna), Claude for everything else. Only the chosen
    provider's key is required."""
    if is_openrouter_model(model):
        if not openrouter_api_key:
            raise MissingAPIKeyError(model, "OPENROUTER_API_KEY", "OpenRouter")
        return OpenRouterProvider(
            api_key=openrouter_api_key, model=model, max_concurrent_calls=max_concurrent_calls
        )
    if not anthropic_api_key:
        raise MissingAPIKeyError(model, "ANTHROPIC_API_KEY", "Anthropic")
    return ClaudeProvider(
        api_key=anthropic_api_key, model=model, max_concurrent_calls=max_concurrent_calls
    )
