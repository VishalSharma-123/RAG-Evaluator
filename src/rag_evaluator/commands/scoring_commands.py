from __future__ import annotations

from pathlib import Path

from rag_evaluator.application.types import ScoreRunSummary
from rag_evaluator.persistence import DuckDBResultsStore


def score_run(
    *,
    database_path: Path,
    run_id: str,
    retrieval_k: int | None = None,
) -> ScoreRunSummary:
    """
    Deterministically rescore a persisted run without re-calling LLM judges.
    """

    store = DuckDBResultsStore(database_path=database_path)
    outcome = store.rescore_run(run_id=run_id, retrieval_k=retrieval_k)
    return ScoreRunSummary(
        run_id=run_id,
        database_path=database_path,
        sample_count=outcome["sample_count"],
        metric_count=outcome["metric_count"],
        failure_label_count=outcome["failure_label_count"],
    )
