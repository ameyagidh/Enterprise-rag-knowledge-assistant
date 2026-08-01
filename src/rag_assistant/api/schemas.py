from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[str]


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_indexed: int
    rebuilt: bool


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    llm_provider: str


class DocumentsResponse(BaseModel):
    documents: list[str]
