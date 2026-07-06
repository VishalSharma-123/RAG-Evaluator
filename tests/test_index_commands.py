from __future__ import annotations

from pathlib import Path

from rag_evaluator.commands.index_commands import build_index
from rag_evaluator.config import ExperimentConfig
from rag_evaluator.execution.types import PipelineRuntime


def test_build_index_builds_selected_pipeline(tmp_path: Path, make_chunk, monkeypatch) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment_name: unit\npipelines: []\ndatasets: []\n", encoding="utf-8")
    pipeline = {
        "name": "pipeline-1",
        "chunker": {"type": "fixed", "chunk_size": 128},
        "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
        "store": {"provider": "memory"},
        "retriever": {"type": "bm25", "top_k": 1},
    }
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [],
            "pipelines": [pipeline],
        }
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "rag_evaluator.commands.index_commands.load_experiment_config",
        lambda path: experiment,
    )
    monkeypatch.setattr(
        "rag_evaluator.commands.index_commands.load_experiment_inputs",
        lambda loaded_experiment: type(
            "Inputs",
            (),
            {
                "experiment": loaded_experiment,
                "samples": [object(), object()],
                "documents": [object()],
            },
        )(),
    )

    def fake_build_pipeline_runtime(*, pipeline, documents):
        captured["pipeline_name"] = pipeline.name
        captured["documents"] = documents
        return PipelineRuntime(
            pipeline=pipeline,
            chunks=[make_chunk()],
            retriever=object(),
            reranker=object(),
            generator=object(),
            judge=object(),
            metadata={},
        )

    monkeypatch.setattr(
        "rag_evaluator.commands.index_commands.build_pipeline_runtime",
        fake_build_pipeline_runtime,
    )

    summary = build_index(config_path=config_path, pipeline_name="pipeline-1")

    assert captured["pipeline_name"] == "pipeline-1"
    assert summary.experiment_name == "unit"
    assert summary.document_count == 1
    assert summary.sample_count == 2
    assert summary.pipeline_indexes[0].chunk_count == 1
    assert summary.pipeline_indexes[0].retriever_type == "bm25"
