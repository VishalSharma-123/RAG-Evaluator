from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_evaluator.config import (
    EmbedderProvider,
    ExperimentConfig,
    VectorStoreProvider,
)


def test_experiment_config_accepts_openrouter_store_and_embedder() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [
                {
                    "name": "tiny",
                    "source": "local_jsonl",
                    "path": "data/tiny.jsonl",
                    "split": "test",
                }
            ],
            "pipelines": [
                {
                    "name": "pipeline-1",
                    "chunker": {"type": "fixed", "chunk_size": 128},
                    "embedder": {
                        "provider": "openrouter",
                        "model": "nvidia/test-embed",
                        "metadata": {
                            "api_key_env": "OPENROUTER_API_KEY",
                            "base_url": "https://openrouter.ai/api/v1/embeddings",
                        },
                    },
                    "store": {
                        "provider": "chroma",
                        "collection_name": "unit_collection",
                    },
                    "retriever": {"type": "vector", "top_k": 3},
                }
            ],
        }
    )

    assert config.pipelines[0].embedder.provider == EmbedderProvider.OPENROUTER
    assert config.pipelines[0].store.provider == VectorStoreProvider.CHROMA


def test_pipeline_store_defaults_to_memory() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [
                {
                    "name": "tiny",
                    "source": "local_jsonl",
                    "path": "data/tiny.jsonl",
                    "split": "test",
                }
            ],
            "pipelines": [
                {
                    "name": "pipeline-1",
                    "chunker": {"type": "fixed", "chunk_size": 128},
                    "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
                    "retriever": {"type": "vector", "top_k": 3},
                }
            ],
        }
    )

    assert config.pipelines[0].store.provider == VectorStoreProvider.MEMORY


def test_embedder_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "experiment_name": "unit",
                "datasets": [
                    {
                        "name": "tiny",
                        "source": "local_jsonl",
                        "path": "data/tiny.jsonl",
                        "split": "test",
                    }
                ],
                "pipelines": [
                    {
                        "name": "pipeline-1",
                        "chunker": {"type": "fixed", "chunk_size": 128},
                        "embedder": {
                            "provider": "bge",
                            "model": "BAAI/bge-small-en-v1.5",
                            "unknown": True,
                        },
                        "retriever": {"type": "vector", "top_k": 3},
                    }
                ],
            }
        )


def test_experiment_config_accepts_synthetic_generation_section() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [
                {
                    "name": "tiny",
                    "source": "local_jsonl",
                    "path": "data/tiny.jsonl",
                    "split": "test",
                }
            ],
            "pipelines": [
                {
                    "name": "pipeline-1",
                    "chunker": {"type": "fixed", "chunk_size": 128},
                    "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
                    "retriever": {"type": "vector", "top_k": 3},
                    "generator": {
                        "provider": "openrouter",
                        "model": "nvidia/nemotron-3-super-120b-a12b:free",
                        "metadata": {"reasoning_enabled": True},
                    },
                }
            ],
            "synthetic_generation": {
                "pipeline": "pipeline-1",
                "chunks_path": "examples/chunks.jsonl",
                "output_path": "examples/synthetic_samples.jsonl",
                "question_types": ["factoid", "unanswerable"],
                "max_samples": 4,
            },
        }
    )

    assert config.synthetic_generation is not None
    assert config.synthetic_generation.pipeline == "pipeline-1"
    assert config.synthetic_generation.question_types == ["factoid", "unanswerable"]
