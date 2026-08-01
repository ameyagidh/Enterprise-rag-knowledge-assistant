"""Cross-encoder reranker wrapper."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document


@lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(question: str, docs: list[Document], model_name: str, top_n: int) -> list[Document]:
    if not docs:
        return []
    encoder = _get_cross_encoder(model_name)
    pairs = [(question, d.page_content) for d in docs]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]
