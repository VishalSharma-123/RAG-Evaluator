from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.synthetic.types import ProviderGenerationResult


class LLMProviderClient(ABC):
    """
    Base contract for provider-specific synthetic generation clients.
    """
    
    @abstractmethod
    def generate_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            metadata: dict[str, Any] | None = None,
    ) -> ProviderGenerationResult:
        """
        Generate a strict JSON response from the configured LLM Provider.
        :param system_prompt:
        :param user_prompt:
        :param metadata:
        :return:
        """
        raise NotImplementedError