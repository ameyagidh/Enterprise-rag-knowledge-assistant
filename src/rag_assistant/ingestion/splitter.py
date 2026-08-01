"""Chunking. Uses the maintained `langchain_text_splitters` package rather
than the deprecated `langchain.text_splitter` import path.
"""

from __future__ import annotations

from langchain_core.documents import Document


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
