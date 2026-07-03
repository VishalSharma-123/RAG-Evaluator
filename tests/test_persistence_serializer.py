from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from rag_evaluator.config import ExperimentConfig, PipelineConfig
from rag_evaluator.persistence import DuckDBResultsStore
from rag_evaluator.schemas import EvalResult, FinalContext, GeneratedAnswer


def test_duckdb_results_store_initializes_extended_schema(tmp_path: Path) -> None:
    store = DuckDBResultsStore(database_path=tmp_path / "results.duckdb")

    store.initialize()

    with duckdb.connect(str(store.database_path), read_only=True) as connection:
        run_columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info('runs')").fetchall()
        }
        answer_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info('generated_answers')").fetchall()
        }

    assert run_columns["run_status"][3] is True
    assert run_columns["run_status"][4] == "'pending'"
    assert {"final_context_json", "usage_json", "pricing_json"} <= answer_columns.keys()


def test_duckdb_results_store_upgrades_legacy_schema_in_place(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                config_hash TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                metadata_json TEXT
            );

            CREATE TABLE generated_answers (
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                answer TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency_ms INTEGER,
                cost_usd DOUBLE NOT NULL DEFAULT 0.0,
                metadata_json TEXT,
                PRIMARY KEY (run_id, sample_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO runs (
                run_id,
                experiment_name,
                pipeline_name,
                started_at,
                completed_at,
                metadata_json
            ) VALUES
                ('legacy-pending', 'legacy-experiment', 'legacy-pipeline', NULL, NULL, '{}'),
                (
                    'legacy-running',
                    'legacy-experiment',
                    'legacy-pipeline',
                    TIMESTAMP '2026-07-01 10:00:00',
                    NULL,
                    '{}'
                ),
                (
                    'legacy-completed',
                    'legacy-experiment',
                    'legacy-pipeline',
                    TIMESTAMP '2026-07-01 10:00:00',
                    TIMESTAMP '2026-07-01 10:05:00',
                    '{}'
                )
            """
        )
        connection.execute(
            """
            INSERT INTO generated_answers (
                run_id,
                sample_id,
                answer,
                model_name,
                metadata_json
            ) VALUES ('legacy-completed', 'legacy-sample', 'answer', 'legacy-model', '{}')
            """
        )

    store = DuckDBResultsStore(database_path=database_path)
    store.initialize()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        run_rows = connection.execute(
            "SELECT run_id, run_status FROM runs ORDER BY run_id"
        ).fetchall()
        answer_row = connection.execute(
            """
            SELECT final_context_json, usage_json, pricing_json
            FROM generated_answers
            WHERE run_id = 'legacy-completed' AND sample_id = 'legacy-sample'
            """
        ).fetchone()

    assert run_rows == [
        ("legacy-completed", "completed"),
        ("legacy-pending", "pending"),
        ("legacy-running", "running"),
    ]
    assert answer_row == (None, None, None)


def test_duckdb_results_store_persists_final_context_structurally(
    tmp_path: Path,
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    retrieval_metrics,
) -> None:
    database_path = tmp_path / "results.duckdb"
    pipeline = PipelineConfig.model_validate(
        {
            "name": "pipeline-1",
            "chunker": {"type": "fixed", "chunk_size": 128},
            "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
            "retriever": {"type": "vector", "top_k": 1},
        }
    )
    experiment = ExperimentConfig.model_validate(
        {
            "experiment_name": "unit",
            "datasets": [],
            "pipelines": [pipeline.model_dump(mode="json")],
        }
    )
    chunk = make_chunk(text="Final context chunk.")
    sample = make_sample()
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="Final answer",
        model_name="openrouter/test-model",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=123,
        cost_usd=0.001,
        metadata={
            "provider": "openrouter",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "pricing": {
                "source": "provider_usage",
                "mode": "direct",
            },
        },
    )

    eval_result = EvalResult(
        run_id="run-1",
        sample=sample,
        retrieved_chunks=[make_retrieved_chunk(chunk=chunk)],
        final_context=FinalContext(
            chunks=[chunk],
            rendered_text="[1] Final context chunk.",
            metadata={"source": "reranker"},
        ),
        generated_answer=generated_answer,
        retrieval_metrics=retrieval_metrics,
    )

    store = DuckDBResultsStore(database_path=database_path)
    store.write_run(
        run_id="run-1",
        experiment=experiment,
        pipeline=pipeline,
        results=[eval_result],
        metadata={
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    rows = store.fetch_run("run-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_status"] == "completed"
    final_context = json.loads(row["final_context_json"])
    answer_metadata = json.loads(row["answer_metadata_json"])
    usage_json = json.loads(row["usage_json"])
    pricing_json = json.loads(row["pricing_json"])

    assert final_context["rendered_text"] == "[1] Final context chunk."
    assert final_context["chunks"][0]["chunk_id"] == chunk.chunk_id
    assert answer_metadata["result_metadata"] == {}
    assert usage_json["prompt_tokens"] == 10
    assert pricing_json["source"] == "provider_usage"

    with duckdb.connect(str(store.database_path), read_only=True) as connection:
        run_row = connection.execute(
            "SELECT run_status, metadata_json FROM runs WHERE run_id = ?",
            ["run-1"],
        ).fetchone()

    assert run_row is not None
    assert run_row[0] == "completed"
    run_metadata = json.loads(run_row[1])
    assert run_metadata["run_status"] == "completed"
    assert run_metadata["pipeline_sweep"]["enabled"] is True
    assert run_metadata["pipeline_sweep"]["overrides"] == []
