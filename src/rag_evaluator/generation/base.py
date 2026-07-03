from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer


class Generator(ABC):
    """
    Base interface for answer generators used in the RAG pipeline.
    """
    
    @abstractmethod
    def generate(
            self,
            sample: EvalSample,
            context_chunks: list[Chunk],
            *,
            metadata: dict[str, Any] | None = None
    ) -> GeneratedAnswer:
        """
        Generate one answer for a normalized evaluations sample using retrieved context.
        :param sample:
        :param context_chunks:
        :param metadata:
        :return: 
        """
        raise NotImplementedError