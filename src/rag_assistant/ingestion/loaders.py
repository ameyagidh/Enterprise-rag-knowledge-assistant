"""Multi-format document loading.

The original prototype used a single `DirectoryLoader(glob=None,
loader_cls=TextLoader)`, which forces every non-dotfile through a plain-text
loader and aborts the whole ingestion run the moment it hits a PDF, image, or
any other binary file. This module instead loads each supported extension
with its own loader and `silent_errors=True`, so one bad file degrades
gracefully (logged and skipped) instead of crashing ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# extension -> (glob pattern, loader_cls import path, kwargs)
_LOADER_SPECS: list[tuple[str, dict]] = [
    ("**/*.txt", {}),
    ("**/*.md", {}),
]


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every supported document under `docs_dir`.

    Supports .txt, .md (plain text), .pdf, and .docx. Missing directory
    raises a clear error rather than an opaque loader exception. Files that
    fail to parse are logged and skipped rather than aborting the whole run.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist: {docs_dir}. "
            "Create it and add documents, or point docs_dir at an existing folder."
        )

    documents: list[Document] = []
    documents.extend(_load_text_like(docs_dir))
    documents.extend(_load_pdfs(docs_dir))
    documents.extend(_load_docx(docs_dir))

    if not documents:
        logger.warning("No documents found under %s", docs_dir)
    return documents


def _load_text_like(docs_dir: Path) -> list[Document]:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    docs: list[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(docs_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            silent_errors=True,
            show_progress=False,
        )
        try:
            docs.extend(loader.load())
        except Exception:  # noqa: BLE001 - a bad file should not abort ingestion
            logger.exception("Failed loading text-like documents matching %s", pattern)
    return docs


def _load_pdfs(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in docs_dir.rglob("*.pdf"):
        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs.extend(PyPDFLoader(str(path)).load())
        except Exception:  # noqa: BLE001
            logger.exception("Failed loading PDF %s; skipping", path)
    return docs


def _load_docx(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in docs_dir.rglob("*.docx"):
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            docs.extend(Docx2txtLoader(str(path)).load())
        except Exception:  # noqa: BLE001
            logger.exception("Failed loading DOCX %s; skipping", path)
    return docs
