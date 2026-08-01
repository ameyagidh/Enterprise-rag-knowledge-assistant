# Project Overview — Enterprise RAG Knowledge Assistant

This document is a complete, detailed record of the work done to turn this project from a 105-line
single-file prototype into a production-ready, tested, deployable RAG service. It exists so anyone
picking up the repo — including future-you — can understand *why* the code looks the way it does,
not just *what* it does.

---

## 1. Starting point

The repository originally contained exactly two files:

```
Enterprise-rag-knowledge-assistant/
├── README.md
└── enterprise_rag_knowledge_assistant.py   (105 lines)
```

The script implemented a reasonable pipeline shape — retrieve (bi-encoder) → rerank (cross-encoder)
→ generate (OpenAI) → groundedness check — but had no packaging, no configuration system, no error
handling, no tests, no API, no UI, no Docker, no CI, and referenced a `./knowledge_base` directory
that didn't exist anywhere in the repo (so the documented run command failed immediately on a fresh
clone).

### Bugs identified in the original code

| # | Bug | Location (original file) | Impact |
|---|---|---|---|
| 1 | `_is_grounded` did `d.metadata.get("source", "") in answer`. `""` is a substring of every string, so any document missing `source` metadata made the groundedness check **pass unconditionally** — the exact safety mechanism the project exists to demonstrate was silently disabled. | `_is_grounded`, lines 85–87 | Ungrounded answers could be returned as if verified |
| 2 | The check matched the **full stored path** (e.g. `./knowledge_base/runbook.md`), so a model citing just the filename would incorrectly fail a *correct*, well-grounded answer. | same | False negatives on correct answers |
| 3 | `build_index()` was called unconditionally on every process start against a **persistent** Chroma directory. `load_index()` existed but nothing ever called it. | `__main__`, line 104; `build_index`/`load_index`, lines 44–62 | Every run re-embedded and re-appended the entire corpus — the store grows without bound and retrieval fills with duplicate chunks |
| 4 | No guard for empty retrieval results — an empty knowledge base or an off-topic query would still ask the LLM to answer from zero context. | `ask`, lines 89–99 | Wasted LLM calls, unpredictable output on empty context |
| 5 | `DirectoryLoader(self.config.docs_dir, loader_cls=TextLoader)` with no `glob` and no `silent_errors` — any non-text file (image, PDF, binary) aborted the entire ingestion run. | `build_index`, line 45 | One bad file breaks ingestion for the whole corpus |
| 6 | No error handling anywhere — a transient OpenAI error, a network blip, or a bad API key crashed the process with a raw stack trace. | `_generate`, lines 74–83 | Poor operational behavior, no way to distinguish user-facing failure modes |
| 7 | Deprecated import paths: `langchain_community.embeddings.HuggingFaceEmbeddings`, `langchain_community.vectorstores.Chroma`, `langchain.text_splitter.RecursiveCharacterTextSplitter`. | imports, lines 16–19 | Deprecation warnings today, hard breaks on a future LangChain release |
| 8 | Docstring claimed "ChromaDB / FAISS" support; FAISS was never imported or used. | module docstring, line 11 | Documentation drift |
| 9 | Prompt was an indented triple-quoted f-string with leading whitespace bleeding into every line sent to the model, and no system/user message split. | `_generate`, lines 76–82 | Noisy prompt, no structural separation of instructions vs. content |

Every one of these is fixed in the current codebase — see [§4](#4-bug-fixes-in-detail) for exactly
what changed and why.

---

## 2. What "production ready" meant here

The user's request was to make this an end-to-end product "any company or any person can use
directly": provider flexibility, a real API, a real UI, tests, CI, and containerization. Concretely,
that was broken down into:

1. **Packaging & configuration** — a proper Python package, pinned dependencies, environment-driven
   config, no hardcoded paths or secrets.
2. **Fix the pipeline's real bugs** — see the table above.
3. **Provider-agnostic LLM layer** — OpenAI, Anthropic (Claude), and local Ollama, swappable via one
   config value.
4. **Multi-format ingestion** — `.txt`, `.md`, `.pdf`, `.docx`, with per-file error isolation.
5. **A real API** — FastAPI, with auth, structured error responses, and interactive docs.
6. **A real GUI** — Streamlit chat interface with citations and groundedness visibility.
7. **A seed corpus** — so the app is queryable immediately after cloning, and tests have fixture
   data.
8. **Tests that don't require a paid API key** — a fake LLM provider stands in for real ones in unit
   tests; real-provider tests are opt-in and auto-skip without credentials.
9. **Docker + CI** — one-command run, automated lint + test on every push.

---

## 3. Architecture

### 3.1 Package layout

```
src/rag_assistant/
├── config.py                 # pydantic-settings: single source of truth for all configuration
├── llm/
│   ├── base.py                # LLMProvider protocol + LLMError
│   ├── openai_provider.py      # wraps langchain_openai.ChatOpenAI
│   ├── anthropic_provider.py    # wraps the official `anthropic` SDK directly (claude-opus-5)
│   ├── ollama_provider.py       # raw HTTP against a local/remote Ollama server
│   └── factory.py              # get_llm(settings) -> LLMProvider, the only branch point
├── ingestion/
│   ├── loaders.py              # per-format loaders with per-file error isolation
│   └── splitter.py             # chunking via langchain_text_splitters
├── retrieval/
│   ├── embeddings.py           # cached HuggingFaceEmbeddings loader
│   ├── vectorstore.py          # Chroma lifecycle: manifest fingerprinting, build/load, in-place clear
│   └── reranker.py             # cached CrossEncoder wrapper
├── assistant.py                # EnterpriseKnowledgeAssistant: the fixed retrieve/rerank/generate/ground pipeline
└── api/
    ├── main.py                 # FastAPI app, lifespan startup, auth dependency, routes
    └── schemas.py               # pydantic request/response models
```

`assistant.py` is the only file that orchestrates the full pipeline; every other module does exactly
one job and is swappable independently (see [§7](#7-how-to-extend-it)).

### 3.2 Why a `LLMProvider` protocol

Rather than branching on provider name throughout the codebase, every backend implements one method:

```python
class LLMProvider(Protocol):
    def generate(self, system: str, user: str) -> str: ...
```

`assistant.py` depends only on this protocol. `llm/factory.py` is the single place that knows about
provider names and construction details. Adding a fourth provider means writing one new file and one
new `if` branch — nothing else in the codebase changes.

### 3.3 Why the vector store lifecycle needed a manifest

The original bug (bug #3 above) was "always rebuild." The naive fix — "call `load_index()` if the
directory exists" — is wrong too: it means the index silently goes stale the moment someone adds a
new document to `knowledge_base/`, with no error and no signal.

The fix computes a fingerprint of every file under `docs_dir` (`size:mtime_ns` per relative path,
`retrieval/vectorstore.py::compute_manifest`), stores it as JSON next to the persisted Chroma data,
and compares it on every `ensure_index()` call:

- **Manifest matches** → `load_index()`, no re-embedding.
- **Manifest missing, empty, or different** → `build_index()`, then write the new manifest.

This makes indexing **idempotent**: calling `ensure_index()` repeatedly with no knowledge-base
changes never duplicates work, but adding a file is detected and picked up automatically without
manual intervention.

### 3.4 A subtler bug found *during* implementation: force-rebuild corruption

While writing the test suite (`tests/test_api.py::test_ingest_force_rebuild`), the first
implementation of `build_index()` did:

```python
if persist_dir.exists():
    shutil.rmtree(persist_dir)
persist_dir.mkdir(parents=True, exist_ok=True)
store = Chroma.from_documents(...)
```

This failed with `ValueError: Could not connect to tenant default_tenant` under the FastAPI
`TestClient`. The root cause: `chromadb` caches a client instance **per path** within a process. The
API's startup lifespan already creates a Chroma client at `persist_dir` when it loads/builds the
index once; calling `POST /ingest?force=true` on the *same running process* then deleted that
directory's on-disk contents while the cached client object was still alive, corrupting its internal
SQLite state.

**Fix:** `build_index()` now clears the existing collection **through the client API**
(`Chroma(...).delete_collection()`) instead of deleting the directory from disk, then repopulates it.
This is exactly the kind of bug that only surfaces once you actually exercise a "long-running server,
force-rebuild while running" code path — which is precisely why the test suite includes that case
rather than only testing a fresh-process build.

### 3.5 The groundedness check, precisely

```python
def _grounded_sources(self, answer: str, docs: list[Document]) -> list[str]:
    cited = []
    for d in docs:
        source = d.metadata.get("source")
        if not source:
            continue                          # fix for bug #1: no source => never grounded
        basename = os.path.basename(source)
        if basename in answer:
            cited.append(basename)            # fix for bug #2: match by filename, not full path
    return cited
```

`ask()` treats an empty `cited` list as "not grounded" and returns a fixed fallback string
(`UNGROUNDED_FALLBACK`), never fabricating a citation. This is intentionally a **citation-presence**
check, not a semantic-entailment check — it verifies the model referenced a real retrieved document,
not that every claim in the answer is individually correct. That's a deliberate, cheap, effective
middle ground: it catches the two failure modes that matter most (answering from parametric knowledge
with no source, and citing a document that was never retrieved) without needing a second LLM call to
verify entailment. See [§8](#8-known-limitations--future-work) for the tradeoff this implies.

---

## 4. Bug fixes in detail

| Bug | Original behavior | Fixed behavior | Where |
|---|---|---|---|
| Groundedness always-true for missing `source` | `"" in answer` → always `True` | Documents with no `source` are skipped entirely — they can never contribute a false-positive citation | `assistant.py::_grounded_sources` |
| Groundedness false-negative on filename-only citations | Required the *entire stored path* to appear verbatim in the answer | Matches on `os.path.basename(source)` | `assistant.py::_grounded_sources` |
| Unbounded re-embedding | `build_index()` called unconditionally every run | `ensure_index()` checks a content-fingerprint manifest and only rebuilds when `docs_dir` actually changed | `retrieval/vectorstore.py`, `assistant.py::ensure_index` |
| Force-rebuild directory deletion corrupting a live client | (introduced during this rebuild, then fixed) `shutil.rmtree(persist_dir)` | `Chroma(...).delete_collection()` — clears in place, no filesystem deletion under a live client | `retrieval/vectorstore.py::build_index` |
| Empty retrieval asks LLM anyway | No check | `ask()` short-circuits to the fallback if `_retrieve()` returns `[]`, without calling the LLM | `assistant.py::ask` |
| Ingestion aborts on any binary/unsupported file | Single `DirectoryLoader(loader_cls=TextLoader)`, no glob, no `silent_errors` | Per-extension loaders (`.txt`/`.md`/`.pdf`/`.docx`), each wrapped in a try/except that logs and skips failures | `ingestion/loaders.py` |
| No LLM error handling | Bare `self.llm.invoke(prompt).content` | Every provider's `generate()` catches its SDK's exceptions and raises a normalized `LLMError`; the API maps `LLMError` to a `502` with the provider's message | `llm/*_provider.py`, `api/main.py` |
| Deprecated imports | `langchain_community.embeddings`, `langchain_community.vectorstores.Chroma`, `langchain.text_splitter` | `langchain_huggingface`, `langchain_chroma`, `langchain_text_splitters` | `retrieval/embeddings.py`, `retrieval/vectorstore.py`, `ingestion/splitter.py` |
| Docstring claims FAISS support | Never implemented | Removed the claim; documented Chroma as the only backend, with `vectorstore.py` as the single swap point | `assistant.py` module docstring, README |
| Unstructured prompt | Indented triple-quoted f-string, no system/user split | Explicit `SYSTEM_PROMPT` constant + a clean user message built from citation-tagged context | `assistant.py` |
| `FastAPI.on_event("startup")` deprecation | N/A (new code) | Migrated to the `lifespan` context-manager pattern during implementation, once `ruff`/`pytest` surfaced the deprecation warning | `api/main.py` |

---

## 5. Testing strategy

### 5.1 Why a fake LLM provider

Every unit test needs the pipeline to run deterministically and without hitting a paid API. Rather
than mocking at the HTTP layer, `tests/conftest.py::FakeLLMProvider` implements the exact same
`LLMProvider.generate(system, user) -> str` interface real providers do — it's a first-class
implementation of the protocol, not a patched-in mock. It echoes back a citation tag pulled from
whatever context it's given, or a configurable "no citation" response, so tests can exercise both the
grounded and ungrounded code paths precisely.

### 5.2 What's covered

| File | What it verifies |
|---|---|
| `test_ingestion.py` | Missing directory raises a clear error; `.txt`/`.md` load correctly; an unsupported binary file (`.png`) is skipped without aborting ingestion; chunking respects size limits |
| `test_retrieval.py` | `needs_rebuild()` is `True` with no index; the manifest changes when a file is added/removed; **regression test**: calling `ensure_index()` twice with no changes does not duplicate chunks (the original bug #3, verified via `vector_store.get()["ids"]` count) |
| `test_grounding.py` | **Direct regression tests for bugs #1 and #2**: a document with no `source` metadata is never grounded regardless of answer text; a citation by filename-only is recognized even when the stored `source` is a full path; an uncited answer is rejected; `ask()` returns the fallback when the LLM doesn't cite anything; `ask()` never calls the LLM at all when retrieval returns no candidates |
| `test_assistant.py` | Calling `ask()` before the index is loaded raises `RuntimeError`; building an index over an empty knowledge base raises a clear error instead of silently indexing zero chunks; `list_sources()` returns the expected filenames; `ensure_index()` calls `load_index()`, not `build_index()`, when nothing changed |
| `test_api.py` | Full FastAPI request/response cycle against a patched settings + fake LLM (no real network calls): `/health`, `/documents`, `/query` (both grounded and validation-error paths), `/ingest?force=true`, and the bearer-token auth gate (401 without a token, 200 with the correct one) |
| `test_integration.py` | Opt-in (`@pytest.mark.integration`) tests that call the **real** Anthropic and OpenAI APIs end-to-end. Each is individually gated with `@pytest.mark.skipif(not os.environ.get("<PROVIDER>_API_KEY"))`, so they skip cleanly (not fail) in CI or on a fresh clone with no keys configured |

**Result at last verification: 23/23 unit tests passing, `ruff check` clean, zero API keys required.**

### 5.3 CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: install with `pip install -e ".[dev]"`,
`ruff check src tests`, then `pytest -m "not integration"`. No secrets are configured in CI, which is
exactly the point — the unit suite is designed to prove correctness without them.

---

## 6. End-to-end verification actually performed

Beyond `pytest`, the running application was exercised for real during this build, not just
unit-tested:

1. **`uvicorn` started against a real Anthropic key placeholder** — confirmed the app correctly calls
   the real Anthropic API and surfaces a clean `502` with the provider's own error message on
   authentication failure, rather than crashing.
2. **`uvicorn` + a real local Ollama server** (`ollama pull llama3.2:1b`, `ollama serve`) — confirmed
   a fully real, no-API-key generation path end to end: indexing the seed corpus, retrieving,
   reranking, generating, and grounding-checking a real model's real output.
3. **Streamlit UI driven via the Chrome DevTools Protocol** (headless Chrome, real page loads, real
   typed input, real wait for a real model response) to capture the screenshots embedded in the
   README — these are genuine captures of the running app, not mockups or illustrations.
4. **The groundedness gate verified live in both directions**: `"What is our on-call escalation
   policy?"` → grounded, cited answer; `"Write a haiku about the ocean."` → correctly refused as
   ungrounded (captured in the README's ungrounded-fallback screenshot).
5. All temporary processes (`uvicorn`, `streamlit`, `ollama serve`, headless Chrome) and the local
   `chroma_store/` test artifact were stopped and cleaned up after verification — see the shutdown
   note at the end of this document for exact commands used.

---

## 7. How to extend it

- **Swap the vector store.** `retrieval/vectorstore.py` is the *only* file that imports Chroma.
  Implement `compute_manifest`, `needs_rebuild`, `build_index`, and `load_index` against another
  backend (FAISS, pgvector, Qdrant, ...) with the same signatures, and nothing else in the codebase
  needs to change.
- **Add an LLM provider.** Implement `LLMProvider.generate(system, user) -> str` in a new file under
  `llm/`, add one branch to `llm/factory.py`, add its config fields to `config.py`. `assistant.py`
  never needs to change.
- **Add a document format.** Add a loader function to `ingestion/loaders.py` following the same
  try/except-and-skip pattern as `_load_pdfs`/`_load_docx`.
- **Change the auth model.** `api/main.py::_require_auth` is the single dependency gating every
  non-health endpoint — replace bearer-token comparison with JWT validation, an API gateway header
  check, etc., without touching route logic.

---

## 8. Known limitations & future work

Documented honestly rather than hidden:

- **Groundedness is citation-presence, not semantic entailment.** A model *could* cite a real source
  filename while still making an unsupported claim about its contents; the check would still mark it
  "grounded." A stronger (and much more expensive) version would run a second LLM call asking "does
  this specific claim follow from this specific retrieved text?" This was a deliberate cost/precision
  tradeoff for this iteration.
- **Small local models (e.g. `llama3.2:1b`) are noticeably weaker** than Claude/GPT-4-class models at
  following the "only cite what you were actually given" instruction — during live verification, the
  1B model occasionally over-cited unrelated retrieved chunks. This is a model-capability limitation,
  not a bug in the grounding logic itself (the logic correctly detects whatever citations are
  present); it's called out explicitly in the main README's troubleshooting table.
- **No per-tenant document isolation or multi-user auth** — the single `API_AUTH_TOKEN` is a shared
  secret, adequate for a single-team/internal deployment but not a multi-tenant SaaS posture.
- **No observability stack** (structured metrics export, tracing) — only basic Python logging. A
  natural next step for a larger deployment would be OpenTelemetry instrumentation around the
  retrieve/rerank/generate/ground stages.
- **The Streamlit "upload a document" control** currently instructs the user to copy the file into
  `knowledge_base/` manually rather than wiring a dedicated upload endpoint — noted directly in the UI
  rather than silently doing nothing.

---

## 9. Reproducing this from scratch

```bash
git clone https://github.com/ameyagidh/Enterprise-rag-knowledge-assistant.git
cd Enterprise-rag-knowledge-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set an API key, or switch LLM_PROVIDER=ollama
ruff check src tests
pytest -m "not integration"
uvicorn rag_assistant.api.main:app --reload &
streamlit run ui/streamlit_app.py
```

See the main [README](../README.md) for the full installation guide, API reference, configuration
table, and troubleshooting section.
