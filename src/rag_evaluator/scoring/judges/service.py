from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer, GenerationMetrics
from rag_evaluator.scoring.judges.base import JudgeScoringError
from rag_evaluator.scoring.judges.heuristic import HeuristicJudge
from rag_evaluator.scoring.judges.parsing import parse_judge_response
from rag_evaluator.scoring.judges.prompt import SYSTEM_PROMPT, build_judge_prompt
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.registry import build_synthetic_provider


@dataclass(frozen=True)
class LLMJudgeService:
    """
    Provider-backed judge service that scores answers with a configured LLM.
    """
    
    config: LLMConfig
    heuristic_judge: HeuristicJudge = field(default_factory=HeuristicJudge)
    
    def score(
            self,
            sample: EvalSample,
            generated_answer: GeneratedAnswer,
            *,
            context_chunks: list[Chunk],
            metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        request_metadata = metadata or {}
        provider = build_synthetic_provider(self.config)
        start_time = time.perf_counter()
        
        try:
            provider_result = provider.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_judge_prompt(
                    sample=sample,
                    generated_answer=generated_answer,
                    context_chunks=context_chunks,
                ),
                metadata=request_metadata,
            )
        except SyntheticProviderError as exc:
            raise JudgeScoringError(f"Judge request failed: {exc}") from exc
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response_metadata = self._response_metadata(provider_result.response_metadata)
        
        judge_metadata: dict[str, Any] = {
            **request_metadata,
            "provider": self.config.provider.value,
            "model": self.config.model,
            "latency_ms": latency_ms,
            "response_metadata": response_metadata,
        }
        
        if provider_result.reasoning_details is not None:
            judge_metadata["reasoning_detail"] = provider_result.reasoning_details
        if provider_result.raw_response is not None:
            judge_metadata["raw_response"] = provider_result.raw_response
        if response_metadata.get("id") is not None:
            judge_metadata["response_id"] = response_metadata["id"]

        try:
            metric = parse_judge_response(provider_result.content)
        except JudgeScoringError:
            raise
        except Exception as exc:
            raise JudgeScoringError(f"Failed to parse judge response: {exc}") from exc

        return metric

    def score_with_fallback(
        self,
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        *,
        context_chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> GenerationMetrics:
        try:
            return self.score(
                sample,
                generated_answer,
                context_chunks=context_chunks,
                metadata=metadata,
            )
        except JudgeScoringError:
            return self.heuristic_judge.score(
                sample,
                generated_answer,
                context_chunks=context_chunks,
                metadata=metadata,
            )

    def _response_metadata(self, response_metadata: dict[str, Any] | None) -> dict[str, Any]:
        return response_metadata if isinstance(response_metadata, dict) else {}
