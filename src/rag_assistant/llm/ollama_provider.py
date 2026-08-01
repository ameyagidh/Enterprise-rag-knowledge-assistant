"""Local/free provider via a running Ollama instance (no API key required)."""

from __future__ import annotations

import logging

import requests

from rag_assistant.config import Settings
from rag_assistant.llm.base import LLMError

logger = logging.getLogger(__name__)


class OllamaProvider:
    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.llm_timeout_seconds

    def generate(self, system: str, user: str) -> str:
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Ollama request failed")
            raise LLMError(
                f"Could not reach Ollama at {self._base_url} (is it running?): {exc}"
            ) from exc

        data = resp.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise LLMError("Ollama response contained no content.")
        return content
