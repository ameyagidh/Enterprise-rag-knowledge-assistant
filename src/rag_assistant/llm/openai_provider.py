"""OpenAI provider — wraps `langchain_openai.ChatOpenAI`."""

from __future__ import annotations

import logging

from rag_assistant.config import Settings
from rag_assistant.llm.base import LLMError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def generate(self, system: str, user: str) -> str:
        try:
            response = self._llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider error
            logger.exception("OpenAI generation failed")
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return response.content
