from __future__ import annotations

import pytest

from rag_assistant.assistant import EnterpriseKnowledgeAssistant


def test_ask_before_index_raises(settings, fake_llm):
    a = EnterpriseKnowledgeAssistant(settings, fake_llm)
    with pytest.raises(RuntimeError):
        a.ask("anything")


def test_build_index_raises_on_empty_knowledge_base(settings, fake_llm, tmp_path):
    empty_dir = tmp_path / "empty_kb"
    empty_dir.mkdir()
    settings.docs_dir = empty_dir
    a = EnterpriseKnowledgeAssistant(settings, fake_llm)

    with pytest.raises(RuntimeError):
        a.build_index()


def test_list_sources_returns_known_extensions(assistant):
    sources = assistant.list_sources()
    assert "on_call_escalation.md" in sources
    assert "expense_policy.md" in sources


def test_ensure_index_loads_without_rebuilding_when_unchanged(assistant, monkeypatch):
    calls = {"build": 0, "load": 0}
    monkeypatch.setattr(assistant, "build_index", lambda: calls.__setitem__("build", calls["build"] + 1) or 0)
    monkeypatch.setattr(assistant, "load_index", lambda: calls.__setitem__("load", calls["load"] + 1))

    assistant.ensure_index()

    assert calls["build"] == 0
    assert calls["load"] == 1
