"""
Centralized, environment-driven configuration.

All paths are resolved relative to the *package* location by default (not the
process's current working directory), so the app behaves the same whether you
run it from the repo root, from a Docker WORKDIR, or from an installed wheel.
Override any of these via environment variables or a `.env` file — see
`.env.example` for the full list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root when running from source (src/rag_assistant/config.py -> repo root
# is three parents up). Used only to compute sane *defaults* — every default
# below can still be overridden via env vars.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider selection ---
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="anthropic",
        description="Which LLM backend to use: openai | anthropic | ollama",
    )
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")

    anthropic_api_key: str | None = Field(default=None)
    # claude-opus-5 is the current default Opus-tier model as of this writing.
    anthropic_model: str = Field(default="claude-opus-5")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")

    llm_temperature: float = Field(default=0.0)
    llm_timeout_seconds: float = Field(default=60.0)
    llm_max_retries: int = Field(default=2)

    # --- Ingestion / retrieval pipeline ---
    docs_dir: Path = Field(default=_PACKAGE_ROOT / "knowledge_base")
    persist_dir: Path = Field(default=_PACKAGE_ROOT / "chroma_store")
    manifest_path: Path = Field(default=_PACKAGE_ROOT / "chroma_store" / "manifest.json")

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    chunk_size: int = Field(default=700)
    chunk_overlap: int = Field(default=100)
    retrieve_k: int = Field(default=10)
    rerank_top_n: int = Field(default=4)

    # --- API / auth ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    # If unset, the API runs with no auth (fine for local/dev). Set it to
    # require `Authorization: Bearer <token>` on every request except /health.
    api_auth_token: str | None = Field(default=None)

    # --- UI ---
    streamlit_port: int = Field(default=8501)
    api_base_url: str = Field(default="http://localhost:8000")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> "Settings":
        # Fail fast with a clear message rather than a deep stack trace from
        # inside a third-party SDK when the app actually tries to call it.
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "llm_provider=openai but OPENAI_API_KEY is not set. "
                "Set it in your environment or .env file."
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "llm_provider=anthropic but ANTHROPIC_API_KEY is not set. "
                "Set it in your environment or .env file."
            )
        return self


def get_settings() -> Settings:
    """Construct Settings fresh each call (cheap; avoids stale cached state in tests)."""
    return Settings()
