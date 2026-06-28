from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.generation.base import Generator
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer


@dataclass(frozen=True)
class NemotronGenerator(Generator):
    """
    Generator interface for the approved nvidia/nemotron family.
    Runtime credentials, base URL, and provider-specific connection settings
    must be supplied through environment variables outside source control.
    """
    
    config: LLMConfig
    
    def __post_init__(self) -> None:
        if not self.config.model.startswith("nvidia/nemotron"):
            raise ValueError("Nemotron model must be 'nvidia/nemotron'")

    def generate(
            self,
            sample: EvalSample,
            context_chunks: list[Chunk],
            *,
            metadata: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        """
        Generate one grounded answer for retrieved context.
        :param sample:
        :param context_chunks:
        :param metadata:
        :return:
        """
        raise NotImplementedError(
            "NemotronGenerator.generate is not implemented yet."
        )