from __future__ import annotations

from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.retrieval import vectorstore


def test_needs_rebuild_true_when_no_index(tmp_path: Path):
    docs_dir = tmp_path / "kb"
    docs_dir.mkdir()
    persist_dir = tmp_path / "chroma"
    manifest = tmp_path / "chroma" / "manifest.json"

    assert vectorstore.needs_rebuild(docs_dir, persist_dir, manifest) is True


def test_manifest_changes_when_file_added(tmp_path: Path):
    docs_dir = tmp_path / "kb"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello")

    before = vectorstore.compute_manifest(docs_dir)
    (docs_dir / "b.txt").write_text("world")
    after = vectorstore.compute_manifest(docs_dir)

    assert before != after
    assert "a.txt" in before
    assert "b.txt" in after and "b.txt" not in before


def test_index_is_idempotent_across_ensure_index_calls(settings: Settings, assistant):
    """Regression test for the original bug where build_index() was called
    unconditionally on every run, duplicating chunks in the persisted store."""
    first_count = len(assistant.vector_store.get()["ids"])

    # Nothing in docs_dir changed, so a second ensure_index() must load, not
    # rebuild -- and the chunk count must stay identical, not double.
    assistant.ensure_index()
    second_count = len(assistant.vector_store.get()["ids"])

    assert first_count == second_count
    assert first_count > 0
