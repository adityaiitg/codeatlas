"""LLM client abstraction using LiteLLM."""

from __future__ import annotations

import logging

from codeatlas.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Provides unified LLM completions for wiki generation and synthesis."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model

    def complete(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Call LLM via litellm with graceful failure if offline or misconfigured."""
        try:
            import litellm

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.debug(f"LLM synthesis unavailable: {e}")
            return None
