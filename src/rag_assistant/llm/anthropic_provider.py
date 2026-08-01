"""Anthropic (Claude) provider — uses the official `anthropic` Python SDK directly.

Defaults to `claude-opus-5`. Thinking is left at its model default (adaptive)
rather than being explicitly disabled or configured with a token budget, per
current Claude API guidance.
"""

from __future__ import annotations

import logging

from rag_assistant.config import Settings
from rag_assistant.llm.base import LLMError

logger = logging.getLogger(__name__)


class AnthropicProvider:
    def __init__(self, settings: Settings):
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.anthropic_model

    def generate(self, system: str, user: str) -> str:
        import anthropic

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as exc:
            logger.exception("Anthropic API returned an error status")
            raise LLMError(f"Anthropic request failed ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            logger.exception("Anthropic API connection failed")
            raise LLMError(f"Could not reach Anthropic API: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Anthropic declined to answer this request (safety refusal).")

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise LLMError("Anthropic response contained no text content.")
        return "".join(text_blocks)
