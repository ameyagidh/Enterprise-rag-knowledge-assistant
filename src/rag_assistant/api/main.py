"""FastAPI application exposing the RAG assistant.

Endpoints:
  GET  /health     - liveness/readiness, never requires auth
  POST /ingest     - (re)build the vector index from docs_dir
  POST /query      - ask a question, get a grounded (or refused) answer
  GET  /documents  - list currently indexed source files

Auth: if `API_AUTH_TOKEN` is set, every endpoint except /health requires
`Authorization: Bearer <token>`. If unset, the API runs open (fine for local
development, not recommended for anything internet-facing).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_assistant.api.schemas import (
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from rag_assistant.assistant import EnterpriseKnowledgeAssistant
from rag_assistant.config import Settings, get_settings
from rag_assistant.llm.base import LLMError
from rag_assistant.llm.factory import get_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag_assistant.api")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    try:
        llm = get_llm(settings)
        assistant = EnterpriseKnowledgeAssistant(settings, llm)
        assistant.ensure_index()
        app.state.assistant = assistant
        app.state.startup_error = None
    except Exception as exc:  # noqa: BLE001
        # Don't crash the whole process on a bad key/missing corpus -- surface
        # the error through /health and let /query fail with a clear message,
        # so `docker compose up` doesn't just die silently.
        logger.exception("Startup failed; API will report unhealthy until fixed")
        app.state.assistant = None
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="Retrieve -> rerank -> generate -> groundedness-checked Q&A over internal docs.",
    version="1.0.0",
    lifespan=_lifespan,
)

_security = HTTPBearer(auto_error=False)


def _require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.api_auth_token:
        return  # auth disabled
    if credentials is None or credentials.credentials != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


def _get_assistant(request: Request) -> EnterpriseKnowledgeAssistant:
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:
        detail = getattr(request.app.state, "startup_error", "Assistant not initialized.")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {detail}")
    return assistant


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    assistant = getattr(request.app.state, "assistant", None)
    return HealthResponse(
        status="ok" if assistant is not None else "degraded",
        index_loaded=assistant is not None and assistant.vector_store is not None,
        llm_provider=settings.llm_provider,
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(_require_auth)])
def ingest(
    force: bool = False, assistant: EnterpriseKnowledgeAssistant = Depends(_get_assistant)
) -> IngestResponse:
    try:
        if force:
            chunks = assistant.build_index()
            rebuilt = True
        else:
            from rag_assistant.retrieval import vectorstore as vs

            rebuilt = vs.needs_rebuild(
                assistant.settings.docs_dir,
                assistant.settings.persist_dir,
                assistant.settings.manifest_path,
            )
            chunks = assistant.build_index() if rebuilt else 0
            if not rebuilt:
                assistant.load_index()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        chunks_indexed=chunks,
        documents_indexed=len(assistant.list_sources()),
        rebuilt=rebuilt,
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_auth)])
def query(
    body: QueryRequest, assistant: EnterpriseKnowledgeAssistant = Depends(_get_assistant)
) -> QueryResponse:
    try:
        result = assistant.ask(body.question)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return QueryResponse(answer=result.answer, grounded=result.grounded, sources=result.sources)


@app.get("/documents", response_model=DocumentsResponse, dependencies=[Depends(_require_auth)])
def documents(assistant: EnterpriseKnowledgeAssistant = Depends(_get_assistant)) -> DocumentsResponse:
    return DocumentsResponse(documents=assistant.list_sources())
