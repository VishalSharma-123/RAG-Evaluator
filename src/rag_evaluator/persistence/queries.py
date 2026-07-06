DELETE_FAILURE_LABELS_BY_RUN_ID = """
DELETE FROM failure_labels
WHERE run_id = ?
"""

DELETE_RETRIEVED_CHUNKS_BY_RUN_ID = """
DELETE FROM retrieved_chunks
WHERE run_id = ?
"""

DELETE_METRIC_SCORES_BY_RUN_ID = """
DELETE FROM metric_scores
WHERE run_id = ?
"""

DELETE_GENERATED_ANSWERS_BY_RUN_ID = """
DELETE FROM generated_answers
WHERE run_id = ?
"""

DELETE_SAMPLES_BY_RUN_ID = """
DELETE FROM samples
WHERE run_id = ?
"""

DELETE_RUNS_BY_RUN_ID = """
DELETE FROM runs
WHERE run_id = ?
"""

INSERT_RUN = """
INSERT INTO runs (
    run_id,
    experiment_name,
    pipeline_name,
    config_hash,
    started_at,
    completed_at,
    run_status,
    metadata_json
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SAMPLE = """
INSERT INTO samples (
    run_id,
    sample_id,
    question,
    question_type,
    source_dataset,
    source_split,
    reference_answer,
    is_answerable,
    metadata_json
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_RETRIEVED_CHUNK = """
INSERT INTO retrieved_chunks (
    run_id,
    sample_id,
    chunk_id,
    document_id,
    rank,
    score,
    retriever_name,
    metadata_json
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_GENERATED_ANSWER = """
INSERT INTO generated_answers (
    run_id,
    sample_id,
    answer,
    model_name,
    prompt_tokens,
    completion_tokens,
    latency_ms,
    cost_usd,
    usage_json,
    pricing_json,
    metadata_json,
    final_context_json
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_METRIC_SCORE = """
INSERT INTO metric_scores (
    run_id,
    sample_id,
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg,
    faithfulness,
    relevance,
    hallucination,
    bert_score
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_FAILURE_LABEL = """
INSERT INTO failure_labels (
    run_id,
    sample_id,
    failure_mode
)
VALUES (?, ?, ?)
"""

FETCH_RETRIEVED_CHUNKS_FOR_RUN = """
SELECT
    run_id,
    sample_id,
    chunk_id,
    document_id,
    rank,
    score,
    retriever_name,
    metadata_json
FROM retrieved_chunks
WHERE run_id = ?
ORDER BY sample_id, rank
"""

FETCH_RUN = """
SELECT
    r.run_status,
    s.run_id,
    s.sample_id,
    s.question,
    s.question_type,
    s.source_dataset,
    s.source_split,
    s.reference_answer,
    s.is_answerable,
    s.metadata_json AS sample_metadata_json,
    ga.answer,
    ga.model_name,
    ga.prompt_tokens,
    ga.completion_tokens,
    ga.latency_ms,
    ga.cost_usd,
    ga.usage_json AS usage_json,
    ga.pricing_json AS pricing_json,
    ga.metadata_json AS answer_metadata_json,
    ga.final_context_json AS final_context_json,
    ms.precision_at_k,
    ms.recall_at_k,
    ms.mrr,
    ms.ndcg,
    ms.faithfulness,
    ms.relevance,
    ms.hallucination,
    ms.bert_score,
    COALESCE(
        list(fl.failure_mode) FILTER (WHERE fl.failure_mode IS NOT NULL),
        []
    ) AS failure_modes
FROM samples AS s
LEFT JOIN runs AS r
    ON r.run_id = s.run_id
LEFT JOIN generated_answers AS ga
    ON ga.run_id = s.run_id
   AND ga.sample_id = s.sample_id
LEFT JOIN metric_scores AS ms
    ON ms.run_id = s.run_id
   AND ms.sample_id = s.sample_id
LEFT JOIN failure_labels AS fl
    ON fl.run_id = s.run_id
   AND fl.sample_id = s.sample_id
WHERE s.run_id = ?
GROUP BY
    r.run_status,
    s.run_id,
    s.sample_id,
    s.question,
    s.question_type,
    s.source_dataset,
    s.source_split,
    s.reference_answer,
    s.is_answerable,
    s.metadata_json,
    ga.answer,
    ga.model_name,
    ga.prompt_tokens,
    ga.completion_tokens,
    ga.latency_ms,
    ga.cost_usd,
    ga.usage_json,
    ga.pricing_json,
    ga.metadata_json,
    ga.final_context_json,
    ms.precision_at_k,
    ms.recall_at_k,
    ms.mrr,
    ms.ndcg,
    ms.faithfulness,
    ms.relevance,
    ms.hallucination,
    ms.bert_score
ORDER BY s.sample_id
"""
