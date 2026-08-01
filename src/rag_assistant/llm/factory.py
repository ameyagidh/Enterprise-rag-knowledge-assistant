"""Picks the configured LLM provider. This is the only place that branches on
`settings.llm_provider` — everything downstream depends only on `LLMProvider`.
"""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.llm.base import LLMProvider


def get_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from rag_assistant.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if settings.llm_provider == "anthropic":
        from rag_assistant.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if settings.llm_provider == "ollama":
        from rag_assistant.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
