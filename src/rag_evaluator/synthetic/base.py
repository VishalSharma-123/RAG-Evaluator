from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_evaluator.schemas import Chunk, EvalSample, QuestionType


class SyntheticGenerator(ABC):
    """
    Base interface for synthetic QA generation from corpus chunks.
    Implementations convert source chunks into normalized EvalSample records
    that can be used by the evaluation pipeline like any other dataset.
    """
    
    @abstractmethod
    def generate_samples(
            self,
            chunks: list[Chunk],
            *,
            question_types: list[QuestionType] | None = None,
            max_samples: int | None = None,
            metadata: dict[str, Any] | None = None
    ) -> list[EvalSample]:
        """
        Generate normalized evaluation samples from corpus chunks.
        :param chunks:
        :param question_types:
        :param max_samples:
        :param metadata:
        :return:
        """
        raise NotImplementedError
