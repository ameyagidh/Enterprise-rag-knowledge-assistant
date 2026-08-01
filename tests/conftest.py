"""Shared test fixtures.

Everything here avoids network calls to a real LLM provider. The embedding
model and cross-encoder reranker are real (small, public HuggingFace models,
downloaded once and cached) — only the LLM generation step is faked, since
that's the part that would otherwise require a paid API key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_assistant.assistant import EnterpriseKnowledgeAssistant
from rag_assistant.config import Settings
from rag_assistant.llm.base import LLMProvider

SAMPLE_DOCS = {
    "on_call_escalation.md": (
        "# On-Call Escalation Policy\n\n"
        "SEV1 incidents page the primary on-call engineer immediately. "
        "If there is no acknowledgment within 5 minutes, escalate to the "
        "secondary on-call engineer.\n"
    ),
    "expense_policy.md": (
        "# Expense Policy\n\n"
        "Meals are reimbursed up to $75 per day while traveling for business. "
        "Submit expense reports within 30 days through Expensify.\n"
    ),
}


@pytest.fixture()
def knowledge_base_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "knowledge_base"
    docs_dir.mkdir()
    for name, content in SAMPLE_DOCS.items():
        (docs_dir / name).write_text(content)
    return docs_dir


@pytest.fixture()
def settings(tmp_path: Path, knowledge_base_dir: Path) -> Settings:
    # anthropic_api_key is set purely to satisfy Settings' fail-fast config
    # validation; no real Anthropic call happens in these tests because the
    # assistant is always constructed with a FakeLLMProvider below.
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="test-key-not-used",
        docs_dir=knowledge_base_dir,
        persist_dir=tmp_path / "chroma_store",
        manifest_path=tmp_path / "chroma_store" / "manifest.json",
        retrieve_k=5,
        rerank_top_n=2,
        chunk_size=300,
        chunk_overlap=30,
    )


class FakeLLMProvider(LLMProvider):
    """Deterministic stand-in for a real LLM: echoes back a citation to the
    first source it sees in the context block, so grounding tests can control
    exactly what the "model" says without any network access."""

    def __init__(self, cite: bool = True, source_to_cite: str | None = None):
        self.cite = cite
        self.source_to_cite = source_to_cite
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        if not self.cite:
            return "Here is an answer with no citation at all."
        source = self.source_to_cite
        if source is None:
            # Pull the first "[source: X]" tag out of the context we were given.
            marker = "[source: "
            start = user.find(marker)
            if start == -1:
                return "Here is an answer with no citation at all."
            end = user.find("]", start)
            source = user[start + len(marker) : end]
        return f"Based on the documentation, here is the answer. [source: {source}]"


@pytest.fixture()
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture()
def assistant(settings: Settings, fake_llm: FakeLLMProvider) -> EnterpriseKnowledgeAssistant:
    a = EnterpriseKnowledgeAssistant(settings, fake_llm)
    a.build_index()
    return a
