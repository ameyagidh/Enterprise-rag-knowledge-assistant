"""Regression tests for the groundedness-check bugs found in the original
prototype (`_is_grounded`):

  1. `d.metadata.get("source", "") in answer` defaults to `""`, and `"" in x`
     is always True in Python -- so a document with no `source` metadata
     silently made grounding pass unconditionally. Fixed version must treat
     "no source" as "not grounded", not "always grounded".
  2. The check matched on the *full* source path, so a model that cited just
     the filename (e.g. "runbook.md") rather than the full path (e.g.
     "./knowledge_base/runbook.md") would incorrectly fail a well-grounded
     answer. Fixed version matches on the basename.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_assistant.assistant import UNGROUNDED_FALLBACK, EnterpriseKnowledgeAssistant


def test_missing_source_metadata_is_never_grounded(assistant: EnterpriseKnowledgeAssistant):
    docs = [Document(page_content="some content", metadata={})]  # no "source" key at all
    answer = "This claim is true regardless of what the answer text says."

    sources = assistant._grounded_sources(answer, docs)

    assert sources == []


def test_citation_by_basename_is_recognized(assistant: EnterpriseKnowledgeAssistant):
    docs = [
        Document(
            page_content="SEV1 pages the primary on-call engineer.",
            metadata={"source": "./knowledge_base/on_call_escalation.md"},
        )
    ]
    # The model cites just the filename, not the full stored path.
    answer = "Per policy, SEV1 pages the primary on-call. [source: on_call_escalation.md]"

    sources = assistant._grounded_sources(answer, docs)

    assert sources == ["on_call_escalation.md"]


def test_uncited_answer_is_not_grounded(assistant: EnterpriseKnowledgeAssistant):
    docs = [
        Document(
            page_content="SEV1 pages the primary on-call engineer.",
            metadata={"source": "on_call_escalation.md"},
        )
    ]
    answer = "I'm just making this up without citing anything."

    sources = assistant._grounded_sources(answer, docs)

    assert sources == []


def test_ask_returns_fallback_when_llm_does_not_cite(assistant, fake_llm):
    fake_llm.cite = False

    result = assistant.ask("What is the on-call escalation policy?")

    assert result.grounded is False
    assert result.answer == UNGROUNDED_FALLBACK
    assert result.sources == []


def test_ask_returns_grounded_answer_when_llm_cites_retrieved_source(assistant):
    result = assistant.ask("What is the on-call escalation policy?")

    assert result.grounded is True
    assert result.sources
    assert all(s.endswith(".md") for s in result.sources)


def test_ask_with_empty_retrieval_short_circuits_without_calling_llm(assistant, fake_llm):
    # Force retrieval to return nothing, simulating an empty/irrelevant knowledge base.
    assistant._retrieve = lambda question: []  # noqa: SLF001 - intentional test override

    result = assistant.ask("Completely unrelated question about astrophysics")

    assert result.grounded is False
    assert result.answer == UNGROUNDED_FALLBACK
    # The LLM must never have been called with an empty context.
    assert fake_llm.last_user is None
