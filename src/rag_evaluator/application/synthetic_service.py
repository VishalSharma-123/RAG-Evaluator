from __future__ import annotations

from pathlib import Path

from rag_evaluator.application.types import SyntheticGenerationSummary
from rag_evaluator.config import LLMConfig
from rag_evaluator.io import load_experiment_config, load_jsonl, write_eval_samples_jsonl
from rag_evaluator.schemas import Chunk, QuestionType
from rag_evaluator.synthetic.models.nemotron import NemotronSyntheticGenerator


def generate_synthetic_from_config(
    *,
    config_path: Path,
    provider: str,
    model: str,
    question_types: list[str] | None,
    max_samples: int | None,
    temperature: float,
    max_tokens: int,
    reasoning_enabled: bool,
    openai_base_url: str | None = None,
) -> SyntheticGenerationSummary:
    """
    Resolve synthetic generation inputs from an experiment config and execute.
    """

    experiment_config = load_experiment_config(config_path)
    synthetic_config = experiment_config.synthetic_generation
    if synthetic_config is None:
        raise ValueError(
            f"Experiment config {config_path} does not define `synthetic_generation`."
        )

    pipeline = next(
        (
            pipeline_config
            for pipeline_config in experiment_config.pipelines
            if pipeline_config.name == synthetic_config.pipeline
        ),
        None,
    )
    if pipeline is None:
        raise ValueError(
            f"Synthetic pipeline {synthetic_config.pipeline!r} was not found in {config_path}."
        )

    resolved_question_types = synthetic_config.question_types or question_types
    resolved_max_samples = (
        synthetic_config.max_samples
        if synthetic_config.max_samples is not None
        else max_samples
    )
    resolved_reasoning_enabled = bool(
        pipeline.generator.metadata.get("reasoning_enabled", reasoning_enabled)
    )
    resolved_llm_metadata = {
        **pipeline.generator.metadata,
        "reasoning_enabled": resolved_reasoning_enabled,
    }
    if openai_base_url:
        resolved_llm_metadata["base_url"] = openai_base_url

    return generate_synthetic_from_inputs(
        chunks_path=Path(synthetic_config.chunks_path),
        output_path=Path(synthetic_config.output_path),
        provider=pipeline.generator.provider.value,
        model=pipeline.generator.model,
        question_types=resolved_question_types,
        max_samples=resolved_max_samples,
        temperature=pipeline.generator.temperature,
        max_tokens=pipeline.generator.max_tokens,
        reasoning_enabled=resolved_reasoning_enabled,
        llm_metadata=resolved_llm_metadata,
        metadata={
            **synthetic_config.metadata,
            "pipeline": pipeline.name,
        },
    )


def generate_synthetic_from_inputs(
    *,
    chunks_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    question_types: list[str] | None,
    max_samples: int | None,
    temperature: float,
    max_tokens: int,
    reasoning_enabled: bool,
    llm_metadata: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> SyntheticGenerationSummary:
    """
    Generate synthetic EvalSample records from a chunk dataset.
    """

    chunk_rows = load_jsonl(chunks_path)
    chunks = [Chunk.model_validate(row) for row in chunk_rows]
    parsed_question_types = (
        [QuestionType(question_type) for question_type in question_types]
        if question_types
        else None
    )

    llm_config = LLMConfig.model_validate(
        {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {
                "reasoning_enabled": reasoning_enabled,
                **(llm_metadata or {}),
            },
        }
    )
    synthetic_generator = NemotronSyntheticGenerator(config=llm_config)
    samples = synthetic_generator.generate_samples(
        chunks,
        question_types=parsed_question_types,
        max_samples=max_samples,
        metadata={
            **(metadata or {}),
            "generator_model": model,
            "generator_provider": provider,
        },
    )

    write_eval_samples_jsonl(output_path, samples)
    return SyntheticGenerationSummary(
        chunks_path=chunks_path,
        output_path=output_path,
        provider=provider,
        model=model,
        chunk_count=len(chunks),
        sample_count=len(samples),
    )
