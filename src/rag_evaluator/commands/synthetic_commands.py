from __future__ import annotations

from pathlib import Path

from rag_evaluator.application import (
    generate_synthetic_from_config,
    generate_synthetic_from_inputs,
)
from rag_evaluator.application.types import SyntheticGenerationSummary


def generate_synthetic(
    *,
    config_path: Path | None = None,
    chunks_path: Path | None,
    output_path: Path | None,
    provider: str,
    model: str,
    question_types: list[str] | None,
    max_samples: int | None,
    temperature: float,
    max_tokens: int,
    reasoning_enabled: bool,
    openai_base_url: str | None = None,
) -> SyntheticGenerationSummary:
    if config_path is not None:
        return generate_synthetic_from_config(
            config_path=config_path,
            provider=provider,
            model=model,
            question_types=question_types,
            max_samples=max_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            openai_base_url=openai_base_url,
        )

    if chunks_path is None or output_path is None:
        raise ValueError("Synthetic generation requires resolved chunks_path and output_path.")

    return generate_synthetic_from_inputs(
        chunks_path=chunks_path,
        output_path=output_path,
        provider=provider,
        model=model,
        question_types=question_types,
        max_samples=max_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_enabled=reasoning_enabled,
        llm_metadata={"base_url": openai_base_url} if openai_base_url else None,
    )
