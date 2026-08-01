"""Chroma vector store lifecycle management.

The original prototype called `build_index()` unconditionally on every
process start, which re-embeds and re-appends the *entire* corpus into a
*persistent* Chroma directory every single run — the store grows without
bound and retrieval fills with duplicates. `load_index()` existed but was
dead code; nothing ever called it.

This module fixes that: it fingerprints the source documents (path + size +
mtime) into a manifest file next to the persisted store. On startup, if the
manifest matches the current `docs_dir` contents, the existing store is
loaded as-is; otherwise the store is rebuilt from scratch (old data is wiped
first, so re-embedding never duplicates).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def compute_manifest(docs_dir: Path) -> dict[str, str]:
    """Fingerprint every file under docs_dir as "size:mtime_ns"."""
    docs_dir = Path(docs_dir)
    manifest: dict[str, str] = {}
    if not docs_dir.exists():
        return manifest
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            manifest[str(path.relative_to(docs_dir))] = f"{stat.st_size}:{stat.st_mtime_ns}"
    return manifest


def _read_manifest(manifest_path: Path) -> dict[str, str] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read manifest at %s; treating as stale", manifest_path)
        return None


def _write_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def needs_rebuild(docs_dir: Path, persist_dir: Path, manifest_path: Path) -> bool:
    """True if the persisted index is missing, empty, or stale vs docs_dir."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        return True
    current = compute_manifest(docs_dir)
    stored = _read_manifest(manifest_path)
    return stored != current


def build_index(chunks: list[Document], embeddings, persist_dir: Path, docs_dir: Path, manifest_path: Path):
    """Rebuild the index from scratch (idempotent: clears stale data first).

    Deliberately clears the existing Chroma *collection* in place rather than
    deleting `persist_dir` from disk. chromadb caches a client instance per
    path within a process; removing the directory out from under a client
    that's still alive (e.g. a force-rebuild against an already-running
    server) corrupts its internal SQLite state instead of failing cleanly.
    Clearing the collection through the client API avoids that entirely.
    """
    from langchain_chroma import Chroma

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    existing = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    try:
        existing.delete_collection()
    except Exception:  # noqa: BLE001 - no existing collection is the common case
        logger.debug("No existing collection to clear at %s", persist_dir, exc_info=True)

    store = Chroma.from_documents(chunks, embeddings, persist_directory=str(persist_dir))
    _write_manifest(manifest_path, compute_manifest(docs_dir))
    logger.info("Built index: %d chunks in %s", len(chunks), persist_dir)
    return store


def load_index(embeddings, persist_dir: Path):
    from langchain_chroma import Chroma

    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
