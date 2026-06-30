LOAD_RUNS = """
SELECT
    run_id,
    experiment_name,
    pipeline_name,
    config_hash,
    started_at,
    completed_at,
    metadata_json
FROM runs
ORDER BY run_id DESC
"""

LOAD_RUN_SUMMARY = """
SELECT
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
    ga.metadata_json AS answer_metadata_json,
    ms.precision_at_k,
    ms.recall_at_k,
    ms.mrr,
    ms.ndcg,
    ms.faithfulness,
    ms.relevance,
    ms.hallucination,
    ms.bert_score
FROM samples AS s
LEFT JOIN generated_answers AS ga
    ON ga.run_id = s.run_id
   AND ga.sample_id = s.sample_id
LEFT JOIN metric_scores AS ms
    ON ms.run_id = s.run_id
   AND ms.sample_id = s.sample_id
WHERE s.run_id = ?
ORDER BY s.sample_id
"""

LOAD_FAILURE_BREAKDOWN = """
SELECT
    failure_mode,
    COUNT(*) AS failure_count
FROM failure_labels
WHERE run_id = ?
GROUP BY failure_mode
ORDER BY failure_count DESC, failure_mode
"""

LOAD_RETRIEVED_CHUNKS = """
SELECT
    sample_id,
    chunk_id,
    document_id,
    rank,
    score,
    retriever_name,
    metadata_json
FROM retrieved_chunks
WHERE run_id = ?
  AND sample_id = ?
ORDER BY rank
"""
