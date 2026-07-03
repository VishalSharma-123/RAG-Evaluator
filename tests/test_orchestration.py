from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_evaluator.config import PipelineConfig
from rag_evaluator.execution.orchestration import expand_pipeline_variants, run_experiment
from rag_evaluator.execution.types import PipelineRunOutput


def test_expand_pipeline_variants_prefers_typed_sweep_config() -> None:
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 3},
            "sweep": {
                "enabled": True,
                "name_suffix_template": "variant_{index}",
                "metadata": {"source": "typed"},
                "overrides": [
                    {
                        "retriever": {"top_k": 5},
                        "metadata": {"tier": "gold"},
                    },
                    {
                        "retriever": {"top_k": 7},
                    },
                ],
            },
            "metadata": {
                "sweep_overrides": [
                    {
                        "retriever": {"top_k": 99},
                    }
                ],
            },
        }
    )

    variants = expand_pipeline_variants(pipeline)

    assert [variant.name for variant in variants] == [
        "pipeline-1__variant_1",
        "pipeline-1__variant_2",
    ]
    assert [variant.retriever.top_k for variant in variants] == [5, 7]
    assert variants[0].metadata["source_pipeline_name"] == "pipeline-1"
    assert variants[0].metadata["sweep_index"] == 1
    assert variants[0].metadata["tier"] == "gold"
    assert variants[0].metadata["source"] == "typed"
    assert variants[0].sweep.enabled is False
    assert variants[0].sweep.overrides == []


def test_expand_pipeline_variants_falls_back_to_legacy_metadata() -> None:
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-legacy",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 3},
            "metadata": {
                "sweep_overrides": [
                    {
                        "retriever": {"top_k": 5},
                    }
                ],
            },
        }
    )

    variants = expand_pipeline_variants(pipeline)

    assert len(variants) == 1
    assert variants[0].name == "pipeline-legacy__sweep_1"
    assert variants[0].retriever.top_k == 5
    assert variants[0].metadata["source_pipeline_name"] == "pipeline-legacy"
    assert variants[0].metadata["sweep_index"] == 1


def test_run_experiment_runs_baseline_before_sweep_variants(
    make_sample,
    monkeypatch,
) -> None:
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 3},
            "sweep": {
                "enabled": True,
                "overrides": [
                    {
                        "retriever": {"top_k": 5},
                    }
                ],
            },
        }
    )
    experiment = SimpleNamespace(
        experiment_name="unit",
        pipelines=[pipeline],
    )

    calls: list[tuple[str, str]] = []

    def fake_build_pipeline_runtime(*, pipeline: PipelineConfig, documents):
        del documents
        return SimpleNamespace(
            pipeline=pipeline,
            chunks=[],
            metadata={"pipeline_name": pipeline.name},
        )

    def fake_run_pipeline_variant(*, runtime, samples):
        del samples
        calls.append((runtime.pipeline.name, runtime.metadata["execution_stage"]))
        return PipelineRunOutput(
            pipeline=runtime.pipeline,
            chunks=[],
            results=[],
            runtime_metadata=dict(runtime.metadata),
        )

    monkeypatch.setattr(
        "rag_evaluator.execution.orchestration.build_pipeline_runtime",
        fake_build_pipeline_runtime,
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.orchestration.run_pipeline_variant",
        fake_run_pipeline_variant,
    )

    output = run_experiment(
        experiment=experiment,
        samples=[make_sample()],
        documents=[],
    )

    assert calls == [
        ("pipeline-1", "baseline"),
        ("pipeline-1__sweep_1", "sweep_variant"),
    ]
    assert [run.pipeline.name for run in output.pipeline_runs] == [
        "pipeline-1",
        "pipeline-1__sweep_1",
    ]
    assert output.pipeline_runs[0].runtime_metadata["execution_stage"] == "baseline"
    assert output.pipeline_runs[1].runtime_metadata["execution_stage"] == "sweep_variant"


def test_run_experiment_stops_if_baseline_fails(
    make_sample,
    monkeypatch,
) -> None:
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 3},
            "sweep": {
                "enabled": True,
                "overrides": [
                    {
                        "retriever": {"top_k": 5},
                    }
                ],
            },
        }
    )
    experiment = SimpleNamespace(
        experiment_name="unit",
        pipelines=[pipeline],
    )

    calls: list[str] = []

    def fake_build_pipeline_runtime(*, pipeline: PipelineConfig, documents):
        del documents
        return SimpleNamespace(
            pipeline=pipeline,
            chunks=[],
            metadata={"pipeline_name": pipeline.name},
        )

    def fake_run_pipeline_variant(*, runtime, samples):
        del samples
        calls.append(runtime.pipeline.name)
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(
        "rag_evaluator.execution.orchestration.build_pipeline_runtime",
        fake_build_pipeline_runtime,
    )
    monkeypatch.setattr(
        "rag_evaluator.execution.orchestration.run_pipeline_variant",
        fake_run_pipeline_variant,
    )

    with pytest.raises(RuntimeError, match="baseline failed"):
        run_experiment(
            experiment=experiment,
            samples=[make_sample()],
            documents=[],
        )

    assert calls == ["pipeline-1"]
