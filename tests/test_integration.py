"""Real-provider integration tests. These make actual network calls and
therefore require a real API key -- they are skipped automatically (not
failed) when the relevant environment variable is absent, e.g. in CI or on a
fresh clone with no keys configured yet.
"""

from __future__ import annotations

import os

import pytest

from rag_assistant.assistant import EnterpriseKnowledgeAssistant
from rag_assistant.config import Settings
from rag_assistant.llm.anthropic_provider import AnthropicProvider
from rag_assistant.llm.openai_provider import OpenAIProvider


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_provider_end_to_end(settings: Settings):
    settings.llm_provider = "anthropic"
    settings.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    llm = AnthropicProvider(settings)
    assistant = EnterpriseKnowledgeAssistant(settings, llm)
    assistant.build_index()

    result = assistant.ask("What is the on-call escalation policy?")

    assert result.answer
    assert result.grounded is True


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_provider_end_to_end(settings: Settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = OpenAIProvider(settings)
    assistant = EnterpriseKnowledgeAssistant(settings, llm)
    assistant.build_index()

    result = assistant.ask("What is the on-call escalation policy?")

    assert result.answer
    assert result.grounded is True
