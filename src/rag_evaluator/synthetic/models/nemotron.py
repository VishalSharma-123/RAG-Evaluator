from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import Chunk, EvalSample, QuestionType
from rag_evaluator.synthetic.base import SyntheticGenerator
from rag_evaluator.synthetic.registry import (
    build_synthetic_provider,
    validate_model_family,
)
from rag_evaluator.synthetic.service import SyntheticGenerationService


@dataclass(frozen=True)
class NemotronSyntheticGenerator(SyntheticGenerator):
    """
    Synthetic QA generator facade for the Nemotron model family.

    This class enforces model-family policy and delegates the actual synthetic
    generation workflow to the shared provider-agnostic service layer.
    """

    config: LLMConfig

    def __post_init__(self) -> None:
        validate_model_family(
            self.config,
            allowed_prefixes=("nvidia/nemotron",),
        )

    def generate_samples(
        self,
        chunks: list[Chunk],
        *,
        question_types: list[QuestionType] | None = None,
        max_samples: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[EvalSample]:
        provider = build_synthetic_provider(self.config)
        service = SyntheticGenerationService(provider=provider)
        return service.generate_samples(
            chunks,
            question_types=question_types,
            max_samples=max_samples,
            metadata=metadata,
        )