from __future__ import annotations

from rag_evaluator.execution.runtime import build_generator
from rag_evaluator.generation.chat_completion import ChatCompletionGenerator


def _build_pipeline(provider: str) -> object:
    from rag_evaluator.config import PipelineConfig

    return PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 2},
            "generator": {
                "provider": provider,
                "model": "meta/llama-3.1-8b-instruct:free",
            },
            "judge": {
                "provider": provider,
                "model": "meta/llama-3.1-8b-instruct:free",
            },
        }
    )


def test_build_generator_returns_generic_chat_completion_for_openrouter() -> None:
    pipeline = _build_pipeline("openrouter")

    generator = build_generator(pipeline)

    assert isinstance(generator, ChatCompletionGenerator)


def test_build_generator_returns_generic_chat_completion_for_openai() -> None:
    pipeline = _build_pipeline("openai")

    generator = build_generator(pipeline)

    assert isinstance(generator, ChatCompletionGenerator)
