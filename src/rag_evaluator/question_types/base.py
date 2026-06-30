from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_evaluator.schemas import Chunk, EvalSample, FailureMode, GeneratedAnswer, RetrievedChunk


@dataclass(frozen=True)
class TypeScoreSignals:
    """
    Internal per-question-type scoring and analysis signals.
    """
    grounded_in_context: bool = False
    grounded_in_reference: bool = False
    used_multiple_evidence_chunks: bool = False
    performed_comparison: bool = False
    covered_key_entities: bool = False
    abstained_correctly: bool = False
    difficulty_mismatch: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

class QuestionTypeRule(ABC):
    """
    Base extension abstract class for question-type-specific behavior.
    """
    
    @abstractmethod
    def prompt_instructions(self) -> list[str]:
        """
        Return prompt instructions for question-type-specific behavior.
        :return:
        """
        raise NotImplementedError
    
    @abstractmethod
    def validate_method(
            self,
            sample: EvalSample,
            *,
            available_chunks_by_id: dict[str, Chunk] | None = None,
    ) -> None:
        """
        Validate question-type-specific invariants for one sample.
        :param sample:
        :param available_chunks_by_id:
        :return:
        """
        raise NotImplementedError
    
    @abstractmethod
    def score_answer(
            self,
            sample: EvalSample,
            generated_answer: GeneratedAnswer | None,
            *,
            context_chunks: list[Chunk],
    ) -> TypeScoreSignals:
        """
        Compute question-type-specific scoring signals for one generated answer.
        :param sample:
        :param generated_answer:
        :param context_chunks:
        :return:
        """
        raise NotImplementedError
    
    @abstractmethod
    def classify_failures(
            self,
            sample: EvalSample,
            retrieved_chunk: list[RetrievedChunk],
            *,
            generated_answer: GeneratedAnswer | None = None,
            type_signals: TypeScoreSignals | None = None,
    ) -> list[FailureMode]:
        """
        Return question-type-specific failure modes for one evaluated sample.
        :param sample:
        :param retrieved_chunk: 
        :param generated_answer:
        :param type_signals:
        :return:
        """
        raise NotImplementedError