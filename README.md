# Enterprise RAG Knowledge Assistant

Production-ready Retrieval-Augmented Generation service that answers questions from internal documentation, with a reranking pass for precision and an explicit **groundedness check** so answers stay tied to real source text instead of drifting into the model's own guesses.

```
Docs -> Chunk -> Embed -> Chroma index
Question -> Retrieve top-k (bi-encoder) -> Rerank top-n (cross-encoder) -> Generate (cited) -> Groundedness check -> Answer
```

## What's included

- **Provider-agnostic LLM layer** — switch between OpenAI, Anthropic (Claude), or a local Ollama model with one config value, no code changes.
- **FastAPI backend** (`/health`, `/ingest`, `/query`, `/documents`) with optional bearer-token auth.
- **Streamlit chat UI** with live groundedness/citation display, reindex button, and knowledge-base browser.
- **Idempotent indexing** — re-running ingestion doesn't duplicate chunks; the index only rebuilds when the knowledge base actually changed.
- **Multi-format ingestion** — `.txt`, `.md`, `.pdf`, `.docx`; a bad file is skipped and logged, not a crash.
- **Fixed groundedness check** — the original prototype's check silently passed for any document missing `source` metadata; this version treats that as *not* grounded, and matches citations by filename rather than requiring an exact full-path match.
- **Test suite** that runs with zero API keys (a fake LLM stands in for the real ones), plus opt-in integration tests against real providers.
- **Docker + docker-compose** — one command to run the whole stack, model weights baked into the image so startup is fast.
- **CI** (GitHub Actions) — lint + unit tests on every push, no secrets required.
- A small seed knowledge base (on-call escalation, expense policy, VPN setup) so the app is immediately queryable after cloning.

## Quickstart

### Option A — Docker (recommended)

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or OPENAI_API_KEY, or switch to ollama)
docker compose up --build
```

- API: http://localhost:8000 (docs at http://localhost:8000/docs)
- UI: http://localhost:8501

### Option B — Local Python

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env and set your provider's API key

uvicorn rag_assistant.api.main:app --reload          # terminal 1
streamlit run ui/streamlit_app.py                    # terminal 2
```

## Configuration

All configuration is environment-driven (see `.env.example` for the full list) — nothing is hardcoded, and no secrets are committed to the repo.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai` \| `anthropic` \| `ollama` | `anthropic` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Claude credentials + model | model: `claude-opus-5` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI credentials + model | model: `gpt-4o-mini` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Local model endpoint | `http://localhost:11434`, `llama3` |
| `DOCS_DIR` | Knowledge base directory | `./knowledge_base` |
| `PERSIST_DIR` | Vector store location | `./chroma_store` |
| `API_AUTH_TOKEN` | If set, `/ingest`, `/query`, `/documents` require `Authorization: Bearer <token>` | unset (no auth) |

## API reference

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "index_loaded": true, "llm_provider": "anthropic"}
```

### `POST /ingest`

```bash
curl -X POST "http://localhost:8000/ingest?force=true"
```
```json
{"chunks_indexed": 9, "documents_indexed": 3, "rebuilt": true}
```

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our on-call escalation policy?"}'
```

**Grounded answer** (question the knowledge base can actually answer):

```json
{
  "answer": "SEV1 incidents page the primary on-call engineer immediately. If there is no acknowledgment within 5 minutes, escalate to the secondary on-call engineer, then to the engineering manager after 10 minutes. [source: on_call_escalation.md]",
  "grounded": true,
  "sources": ["on_call_escalation.md"]
}
```

**Ungrounded fallback** (question outside the knowledge base — this is the groundedness gate working as intended):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

```json
{
  "answer": "I couldn't find a well-grounded answer in the knowledge base for that question.",
  "grounded": false,
  "sources": []
}
```

### `GET /documents`

```bash
curl http://localhost:8000/documents
```
```json
{"documents": ["expense_policy.md", "on_call_escalation.md", "vpn_setup_guide.md"]}
```

## What the UI looks like

The Streamlit app is a standard chat interface:

- **Sidebar** — live API health, LLM provider in use, list of indexed documents, and a "🔄 Reindex knowledge base" button.
- **Main panel** — a chat input at the bottom; each assistant reply shows a caption underneath it, either `✅ Grounded — Sources: on_call_escalation.md` or `⚠️ Not grounded in the knowledge base`, so it's always visible whether an answer is backed by a real document.

```
📚 Enterprise RAG Knowledge Assistant
┌─────────────────────────────┬───────────────────────────────┐
│ Status                      │  🧑 What is our on-call        │
│ 🟢 API status: ok           │     escalation policy?         │
│ Index loaded: True          │                                 │
│ LLM provider: anthropic     │  🤖 SEV1 incidents page the     │
│                              │     primary on-call engineer   │
│ Knowledge base               │     immediately...             │
│ 3 document(s) indexed       │     ✅ Grounded — Sources:      │
│ • expense_policy.md         │        on_call_escalation.md   │
│ • on_call_escalation.md     │                                 │
│ • vpn_setup_guide.md        │  [ Ask a question... ]         │
│ [🔄 Reindex knowledge base] │                                 │
└─────────────────────────────┴───────────────────────────────┘
```

## Testing

```bash
pytest                              # unit tests only need HF model download, no LLM API key
pytest -m integration               # also exercises real OpenAI/Anthropic calls; needs keys set
pytest --cov=rag_assistant          # with coverage
```

Unit tests use a deterministic fake LLM provider so the retrieval, reranking, groundedness-check, and API layers are all fully tested without network calls to a paid provider. Integration tests are marked `@pytest.mark.integration` and skip automatically (not fail) when the corresponding API key env var isn't set.

## Architecture

```
Enterprise-rag-knowledge-assistant/
├── src/rag_assistant/
│   ├── config.py          # pydantic-settings, single source of truth for all config
│   ├── llm/                # provider-agnostic LLM layer (openai/anthropic/ollama)
│   ├── ingestion/           # multi-format loaders + chunking
│   ├── retrieval/           # embeddings, Chroma vector store lifecycle, reranker
│   ├── assistant.py         # retrieve -> rerank -> generate -> ground-check pipeline
│   └── api/                # FastAPI app
├── ui/streamlit_app.py      # chat UI
├── knowledge_base/          # seed documents (replace with your own)
├── tests/                   # unit + integration tests
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

## Extending

- **Add documents:** drop `.txt`/`.md`/`.pdf`/`.docx` files into `knowledge_base/` (or mount your own directory via `DOCS_DIR`), then `POST /ingest?force=true` or click "Reindex" in the UI.
- **Swap the vector store:** `src/rag_assistant/retrieval/vectorstore.py` is the only file that imports Chroma directly — implement the same three functions against another backend (e.g. FAISS, pgvector) to swap it.
- **Add an LLM provider:** implement `LLMProvider.generate(system, user) -> str` in a new file under `llm/`, then add a branch in `llm/factory.py`.

## License

MIT
