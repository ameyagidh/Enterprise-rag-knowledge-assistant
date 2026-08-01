"""Provider-agnostic LLM interface.

Every backend (OpenAI, Anthropic, Ollama, or a test double) implements this
one method. The rest of the codebase never imports a provider SDK directly —
it only depends on this protocol, which is what makes swapping providers a
one-line config change instead of a code change.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries, or is misconfigured."""


class LLMProvider(Protocol):
    def generate(self, system: str, user: str) -> str:
        """Generate a completion given a system prompt and a user prompt.

        Implementations must raise `LLMError` (not a bare provider exception)
        on failure, so callers can handle all providers uniformly.
        """
        ...
