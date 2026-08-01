from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents


def test_load_documents_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_load_documents_reads_markdown_and_txt(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Hello\nWorld")
    (tmp_path / "b.txt").write_text("Plain text content")

    docs = load_documents(tmp_path)

    assert len(docs) == 2
    contents = {d.page_content for d in docs}
    assert any("Hello" in c for c in contents)
    assert any("Plain text content" in c for c in contents)


def test_load_documents_skips_unsupported_binary_file(tmp_path: Path):
    (tmp_path / "good.txt").write_text("readable content")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\nnotarealpng")

    # Must not raise even though image.png can't be parsed as text/pdf/docx.
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].page_content == "readable content"


def test_split_documents_respects_chunk_size(tmp_path: Path):
    (tmp_path / "long.txt").write_text("word " * 500)
    docs = load_documents(tmp_path)

    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        # chunk_size is a soft target for RecursiveCharacterTextSplitter, not
        # a hard cutoff, so allow reasonable slack rather than an exact bound.
        assert len(chunk.page_content) <= 200
