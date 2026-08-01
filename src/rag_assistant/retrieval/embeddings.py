"""Embedding model wrapper, using the maintained `langchain_huggingface`
package instead of the deprecated `langchain_community.embeddings` import.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_embeddings(model_name: str):
    """Cached so repeated calls (e.g. across API requests) don't reload weights."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)
