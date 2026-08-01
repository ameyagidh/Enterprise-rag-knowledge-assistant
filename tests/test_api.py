"""API tests. Patches the startup dependencies (settings + LLM) so no real
network call to an LLM provider is ever made, and no real credentials are
required."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag_assistant.api.main as main_module
from tests.conftest import FakeLLMProvider


@pytest.fixture()
def client(monkeypatch, settings):
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "get_llm", lambda s: FakeLLMProvider())

    with TestClient(main_module.app) as c:
        yield c


def test_health_ok_after_successful_startup(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True


def test_documents_lists_seeded_knowledge_base(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert "on_call_escalation.md" in resp.json()["documents"]


def test_query_returns_grounded_answer(client):
    resp = client.post("/query", json={"question": "What is the on-call escalation policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["sources"]


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_ingest_force_rebuild(client):
    resp = client.post("/ingest", params={"force": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["rebuilt"] is True
    assert body["chunks_indexed"] > 0


def test_query_requires_bearer_token_when_configured(monkeypatch, settings):
    settings.api_auth_token = "secret-token"
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "get_llm", lambda s: FakeLLMProvider())

    with TestClient(main_module.app) as c:
        unauthenticated = c.post("/query", json={"question": "hi"})
        assert unauthenticated.status_code == 401

        authenticated = c.post(
            "/query",
            json={"question": "What is the on-call escalation policy?"},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert authenticated.status_code == 200
