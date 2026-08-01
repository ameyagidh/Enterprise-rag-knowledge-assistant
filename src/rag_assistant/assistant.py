"""
Enterprise RAG Knowledge Assistant — core pipeline.

Answers questions from internal documentation using retrieve -> rerank ->
generate -> groundedness-check, with production fixes over the original
prototype:

  * Groundedness check no longer passes vacuously for documents missing a
    `source` metadata key (the old check did `"" in answer`, which is always
    True), and it matches on a normalized source *basename* rather than a
    full path, so citations like "on_call_escalation.md" are recognized even
    if the prompt cites just the filename.
  * The index is loaded once and only rebuilt when the knowledge base
    contents actually changed (via `rag_assistant.retrieval.vectorstore`),
    instead of unconditionally re-embedding the whole corpus on every start.
  * Empty retrieval results short-circuit to the "no grounded answer"
    response instead of asking the LLM to answer from no context.
  * The LLM call is wrapped so provider errors surface as a clear `LLMError`
    instead of crashing the process.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag_assistant.config import Settings
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents
from rag_assistant.llm.base import LLMError, LLMProvider
from rag_assistant.retrieval import vectorstore
from rag_assistant.retrieval.embeddings import get_embeddings
from rag_assistant.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question "
    "using ONLY the provided context. For every claim, cite the source file "
    "it came from using its filename in square brackets, e.g. [source: "
    "on_call_escalation.md]. If the context does not contain enough "
    "information to answer confidently, say so explicitly instead of "
    "guessing."
)


@dataclass
class AskResult:
    answer: str
    grounded: bool
    sources: list[str] = field(default_factory=list)


UNGROUNDED_FALLBACK = "I couldn't find a well-grounded answer in the knowledge base for that question."


class EnterpriseKnowledgeAssistant:
    def __init__(self, settings: Settings, llm: LLMProvider):
        self.settings = settings
        self.llm = llm
        self.embeddings = get_embeddings(settings.embedding_model)
        self.vector_store = None

    # ---------- Index lifecycle ----------
    def ensure_index(self, force_rebuild: bool = False) -> None:
        """Load the persisted index if it's up to date, else (re)build it.

        Idempotent: calling this repeatedly without knowledge-base changes
        does not re-embed anything after the first call.
        """
        if force_rebuild or vectorstore.needs_rebuild(
            self.settings.docs_dir, self.settings.persist_dir, self.settings.manifest_path
        ):
            self.build_index()
        else:
            self.load_index()

    def build_index(self) -> int:
        raw_docs = load_documents(self.settings.docs_dir)
        chunks = split_documents(raw_docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise RuntimeError(
                f"No documents to index under {self.settings.docs_dir}. "
                "Add at least one .txt/.md/.pdf/.docx file first."
            )
        self.vector_store = vectorstore.build_index(
            chunks,
            self.embeddings,
            self.settings.persist_dir,
            self.settings.docs_dir,
            self.settings.manifest_path,
        )
        logger.info("Indexed %d chunks from %d documents.", len(chunks), len(raw_docs))
        return len(chunks)

    def load_index(self) -> None:
        self.vector_store = vectorstore.load_index(self.embeddings, self.settings.persist_dir)
        logger.info("Loaded existing index from %s", self.settings.persist_dir)

    def list_sources(self) -> list[str]:
        """Distinct source filenames currently in the knowledge base directory."""
        docs_dir = self.settings.docs_dir
        if not docs_dir.exists():
            return []
        exts = {".txt", ".md", ".pdf", ".docx"}
        return sorted(
            str(p.relative_to(docs_dir))
            for p in docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

    # ---------- Retrieve -> Rerank -> Generate -> Ground-check ----------
    def _retrieve(self, question: str) -> list[Document]:
        return self.vector_store.similarity_search(question, k=self.settings.retrieve_k)

    def _rerank(self, question: str, docs: list[Document]) -> list[Document]:
        return rerank(question, docs, self.settings.reranker_model, self.settings.rerank_top_n)

    def _generate(self, question: str, docs: list[Document]) -> str:
        context = "\n\n".join(
            f"[source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
            for d in docs
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        try:
            return self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - last-resort normalization
            raise LLMError(f"Unexpected error generating an answer: {exc}") from exc

    def _grounded_sources(self, answer: str, docs: list[Document]) -> list[str]:
        """Which of the retrieved docs' sources are actually cited in the answer.

        Fixed from the original: documents without a `source` metadata key are
        excluded up front (the old check defaulted to `""`, and `"" in answer`
        is always True, silently disabling the safety net). Matching is done
        against the basename so a citation like "on_call_escalation.md" is
        recognized even though the document's `source` metadata is a full
        path such as "./knowledge_base/on_call_escalation.md".
        """
        cited: list[str] = []
        seen = set()
        for d in docs:
            source = d.metadata.get("source")
            if not source:
                continue
            basename = os.path.basename(source)
            if basename in cited or basename in seen:
                continue
            seen.add(basename)
            if basename in answer:
                cited.append(basename)
        return cited

    def ask(self, question: str) -> AskResult:
        if self.vector_store is None:
            raise RuntimeError("Call ensure_index() (or build_index()/load_index()) first.")

        candidates = self._retrieve(question)
        if not candidates:
            # Empty retrieval guard: don't ask the LLM to answer from nothing.
            return AskResult(answer=UNGROUNDED_FALLBACK, grounded=False, sources=[])

        top_docs = self._rerank(question, candidates)
        answer = self._generate(question, top_docs)
        sources = self._grounded_sources(answer, top_docs)

        if not sources:
            return AskResult(answer=UNGROUNDED_FALLBACK, grounded=False, sources=[])
        return AskResult(answer=answer, grounded=True, sources=sources)
