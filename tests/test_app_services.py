from __future__ import annotations

from pathlib import Path
from typing import cast

from rag_evaluator.application.experiment_inputs import (
    build_source_documents,
    load_experiment_inputs,
)
from rag_evaluator.application.experiment_service import run_experiment_from_config
from rag_evaluator.application.synthetic_service import generate_synthetic_from_config
from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.execution.types import ExperimentRunOutput, PipelineRunOutput
from rag_evaluator.schemas import EvalResult, FinalContext, GenerationMetrics, RetrievalMetrics


def test_build_source_documents_reconstructs_and_deduplicates(make_sample) -> None:
    sample = make_sample(
        sample_id="sample-1",
        reference_answer="fallback answer",
        evidence_spans=[
            {
                "document_id": "doc-1",
                "start_char": 0,
                "end_char": 5,
                "text": "alpha",
            },
            {
                "document_id": "doc-1",
                "start_char": 10,
                "end_char": 15,
                "text": "alpha",
            },
            {
                "document_id": "doc-1",
                "start_char": 20,
                "end_char": 24,
                "text": "beta",
            },
        ],
    )

    documents = build_source_documents([sample])

    assert len(documents) == 2
    assert documents[0].document_id == "doc-1"
    assert documents[0].text == "alpha\n\nbeta"
    assert documents[0].metadata == {
        "source": "evidence_span",
        "source_dataset": "unit",
    }
    assert documents[1].document_id == "sample-1"
    assert documents[1].text == "fallback answer"


def test_load_experiment_inputs_loads_samples_and_documents(
    make_sample,
    monkeypatch,
) -> None:
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [
                {
                    "name": "dataset-a",
                    "source": "local_jsonl",
                    "path": "a.jsonl",
                },
                {
                    "name": "dataset-b",
                    "source": "local_jsonl",
                    "path": "b.jsonl",
                },
            ],
            "pipelines": [],
        }
    )

    def fake_load_dataset_from_config(dataset):
        return [
            make_sample(
                sample_id=f"{dataset.name}-1",
                source_dataset=dataset.name,
                evidence_spans=[
                    {
                        "document_id": f"{dataset.name}-doc",
                        "start_char": 0,
                        "end_char": 4,
                        "text": dataset.name,
                    }
                ],
            )
        ]

    monkeypatch.setattr(
        "rag_evaluator.application.experiment_inputs.load_dataset_from_config",
        fake_load_dataset_from_config,
    )

    inputs = load_experiment_inputs(experiment)

    assert inputs.experiment is experiment
    assert len(inputs.samples) == 2
    assert [dataset.dataset_name for dataset in inputs.datasets] == ["dataset-a", "dataset-b"]
    assert [dataset.sample_count for dataset in inputs.datasets] == [1, 1]
    assert {document.document_id for document in inputs.documents} == {
        "dataset-a-1",
        "dataset-a-doc",
        "dataset-b-1",
        "dataset-b-doc",
    }


def test_run_experiment_from_config_persists_pipeline_runs(
    tmp_path: Path,
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    database_path = tmp_path / "results.duckdb"
    config_path.write_text("experiment_name: unit\npipelines: []\ndatasets: []\n", encoding="utf-8")

    sample = make_sample()
    chunk = make_chunk()
    retrieved = make_retrieved_chunk(chunk=chunk)
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1__sweep_1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 1},
        }
    )
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [
                {
                    "name": "tiny",
                    "source": "local_jsonl",
                    "path": "examples/eval_samples.jsonl",
                    "split": "test",
                }
            ],
            "pipelines": [pipeline.model_dump(mode="json")],
        }
    )
    result = EvalResult(
        run_id=pipeline.name,
        sample=sample,
        retrieved_chunks=[retrieved],
        final_context=FinalContext(
            chunks=[chunk],
            rendered_text="[1] " + chunk.text,
        ),
        generated_answer=None,
        retrieval_metrics=RetrievalMetrics(),
        generation_metrics=GenerationMetrics(),
    )

    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, database_path: Path) -> None:
            captured["database_path"] = database_path

        def write_run(self, **kwargs) -> None:
            captured["write_run"] = kwargs

    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.load_experiment_config",
        lambda path: experiment,
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.load_experiment_inputs",
        lambda loaded_experiment: type(
            "Inputs",
            (),
            {
                "experiment": loaded_experiment,
                "samples": [sample],
                "documents": [type("Doc", (), {"document_id": "doc"})()],
                "datasets": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.execute_experiment",
        lambda **kwargs: ExperimentRunOutput(
            experiment_name="unit",
            pipeline_runs=[
                PipelineRunOutput(
                    pipeline=pipeline,
                    chunks=[chunk],
                    results=[result],
                    runtime_metadata={"runtime": "ok"},
                )
            ],
        ),
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.DuckDBResultsStore",
        FakeStore,
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.build_run_id",
        lambda **kwargs: "unit__pipeline-1__20260701T000000Z",
    )

    summary = run_experiment_from_config(
        config_path=config_path,
        database_path=database_path,
    )

    assert captured["database_path"] == database_path
    write_run = captured["write_run"]
    assert write_run["pipeline"].name == "pipeline-1__sweep_1"
    assert write_run["results"] == [result]
    assert write_run["metadata"]["runtime_metadata"] == {"runtime": "ok"}


def test_run_experiment_from_config_persists_execution_stage_metadata(
    tmp_path: Path,
    make_chunk,
    make_sample,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    database_path = tmp_path / "results.duckdb"
    config_path.write_text("experiment_name: unit\npipelines: []\ndatasets: []\n", encoding="utf-8")

    sample = make_sample()
    chunk = make_chunk()
    baseline_pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 1},
        }
    )
    sweep_pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1__sweep_1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 2},
        }
    )
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [],
            "pipelines": [baseline_pipeline.model_dump(mode="json")],
        }
    )

    baseline_result = EvalResult(
        run_id=baseline_pipeline.name,
        sample=sample,
        retrieved_chunks=[],
        final_context=FinalContext(
            chunks=[chunk],
            rendered_text="[1] " + chunk.text,
        ),
        generated_answer=None,
        retrieval_metrics=RetrievalMetrics(),
        generation_metrics=GenerationMetrics(),
    )
    sweep_result = EvalResult(
        run_id=sweep_pipeline.name,
        sample=sample,
        retrieved_chunks=[],
        final_context=FinalContext(
            chunks=[chunk],
            rendered_text="[1] " + chunk.text,
        ),
        generated_answer=None,
        retrieval_metrics=RetrievalMetrics(),
        generation_metrics=GenerationMetrics(),
    )

    captured: dict[str, object] = {"write_runs": []}

    class FakeStore:
        def __init__(self, database_path: Path) -> None:
            captured["database_path"] = database_path

        def write_run(self, **kwargs) -> None:
            cast(list[dict[str, object]], captured["write_runs"]).append(kwargs)

    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.load_experiment_config",
        lambda path: experiment,
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.load_experiment_inputs",
        lambda loaded_experiment: type(
            "Inputs",
            (),
            {
                "experiment": loaded_experiment,
                "samples": [sample],
                "documents": [type("Doc", (), {"document_id": "doc"})()],
                "datasets": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.execute_experiment",
        lambda **kwargs: ExperimentRunOutput(
            experiment_name="unit",
            pipeline_runs=[
                PipelineRunOutput(
                    pipeline=baseline_pipeline,
                    chunks=[chunk],
                    results=[baseline_result],
                    runtime_metadata={"execution_stage": "baseline"},
                ),
                PipelineRunOutput(
                    pipeline=sweep_pipeline,
                    chunks=[chunk],
                    results=[sweep_result],
                    runtime_metadata={"execution_stage": "sweep_variant", "sweep_index": 1},
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.DuckDBResultsStore",
        FakeStore,
    )
    monkeypatch.setattr(
        "rag_evaluator.application.experiment_service.build_run_id",
        lambda **kwargs: "unit__pipeline-1__20260701T000000Z",
    )

    summary = run_experiment_from_config(
        config_path=config_path,
        database_path=database_path,
    )

    assert summary.pipeline_runs[0].runtime_metadata["execution_stage"] == "baseline"
    assert summary.pipeline_runs[1].runtime_metadata["execution_stage"] == "sweep_variant"
    assert [entry["pipeline"].name for entry in captured["write_runs"]] == [
        "pipeline-1",
        "pipeline-1__sweep_1",
    ]
    assert captured["write_runs"][0]["metadata"]["runtime_metadata"] == {
        "execution_stage": "baseline",
    }
    assert captured["write_runs"][1]["metadata"]["runtime_metadata"] == {
        "execution_stage": "sweep_variant",
        "sweep_index": 1,
    }
    assert summary.experiment_name == "unit"
    assert summary.pipeline_runs[0].run_id == "unit__pipeline-1__20260701T000000Z"
    assert summary.pipeline_runs[0].result_count == 1


def test_generate_synthetic_from_config_resolves_experiment_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "synthetic.jsonl"

    chunks_path.write_text(
        '{"chunk_id":"doc:chunk:0","document_id":"doc","text":"ctx"}\n',
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: synthetic_unit",
                "output_dir: runs",
                "datasets:",
                "  - name: tiny",
                "    source: local_jsonl",
                "    path: examples/eval_samples.jsonl",
                "    split: test",
                "pipelines:",
                "  - name: pipeline-1",
                "    chunker:",
                "      type: fixed",
                "      chunk_size: 128",
                "    embedder:",
                "      provider: bge",
                "      model: BAAI/bge-small-en-v1.5",
                "    retriever:",
                "      type: vector",
                "      top_k: 3",
                "    generator:",
                "      provider: openrouter",
                "      model: nvidia/nemotron-3-super-120b-a12b:free",
                "      temperature: 0.0",
                "      max_tokens: 600",
                "      metadata:",
                "        reasoning_enabled: true",
                "synthetic_generation:",
                "  pipeline: pipeline-1",
                f"  chunks_path: {chunks_path}",
                f"  output_path: {output_path}",
                "  question_types:",
                "    - factoid",
                "  max_samples: 3",
                "  metadata:",
                "    source: yaml",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_generate_synthetic_from_inputs(**kwargs):
        captured.update(kwargs)
        return type(
            "Summary",
            (),
            {
                "chunks_path": kwargs["chunks_path"],
                "output_path": kwargs["output_path"],
                "provider": kwargs["provider"],
                "model": kwargs["model"],
                "chunk_count": 1,
                "sample_count": 1,
            },
        )()

    monkeypatch.setattr(
        "rag_evaluator.application.synthetic_service.generate_synthetic_from_inputs",
        fake_generate_synthetic_from_inputs,
    )

    summary = generate_synthetic_from_config(
        config_path=config_path,
        provider="openrouter",
        model="ignored",
        question_types=None,
        max_samples=None,
        temperature=0.0,
        max_tokens=512,
        reasoning_enabled=False,
    )

    assert summary.output_path == output_path
    assert captured["chunks_path"] == chunks_path
    assert captured["provider"] == "openrouter"
    assert captured["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert captured["question_types"] == ["factoid"]
    assert captured["max_samples"] == 3
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 600
    assert captured["reasoning_enabled"] is True
    assert captured["metadata"] == {
        "source": "yaml",
        "pipeline": "pipeline-1",
    }
