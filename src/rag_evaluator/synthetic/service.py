from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_evaluator.schemas import Chunk, EvalSample, QuestionType
from rag_evaluator.synthetic.errors import (
    SyntheticGenerationError,
    SyntheticValidationError,
)
from rag_evaluator.synthetic.parsing import parse_synthetic_generation_response
from rag_evaluator.synthetic.prompts import (
    SYSTEM_PROMPT,
    build_synthetic_generation_prompt,
)
from rag_evaluator.synthetic.providers.base import LLMProviderClient
from rag_evaluator.synthetic.validation import validate_synthetic_samples_batch


@dataclass(frozen=True)
class SyntheticGenerationService:
    """
    Shared orchestration layer for synthetic sample generation
    """
    
    provider: LLMProviderClient
    
    def generate_samples(
            self,
            chunks: list[Chunk],
            *,
            question_types: list[QuestionType] | None = None,
            max_samples: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> list[EvalSample]:
        if not chunks:
            raise SyntheticGenerationError(
                "At least one chunk is needed."
            )
        
        requested_question_types = question_types or list(QuestionType)
        
        raw_response = self.provider.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_synthetic_generation_prompt(
                chunks,
                question_types=requested_question_types,
                max_samples=max_samples,
            ),
            metadata=metadata,
        )
        
        samples = parse_synthetic_generation_response(raw_response.content)
        
        available_chunk_ids = {chunk.chunk_id for chunk in chunks}
        available_chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }
        validated_samples = validate_synthetic_samples_batch(
            samples,
            available_chunk_ids=available_chunk_ids,
            available_chunks_by_id=available_chunks_by_id,
            allowed_question_types=set(requested_question_types),
        )
        
        return self._merge_metadata(validated_samples, metadata)
    
    def _merge_metadata(
            self,
            samples: list[EvalSample],
            metadata: dict[str, Any] | None,
    ) -> list[EvalSample]:
        if not metadata:
            return samples
        
        merged_samples: list[EvalSample] = []
        
        for sample in samples:
            merged_samples.append(
                sample.model_copy(
                    update = {
                        "metadata": {
                            **metadata,
                            **sample.metadata,
                        }
                    }
                )
            )
        
        return merged_samples
