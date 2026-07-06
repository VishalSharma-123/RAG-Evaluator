LOAD_RUNS = """
SELECT
    run_id,
    experiment_name,
    pipeline_name,
    config_hash,
    started_at,
    completed_at,
    run_status,
    metadata_json
FROM runs
ORDER BY run_id DESC
"""

LOAD_RUN_COMPARISON = """
SELECT
    r.run_id,
    r.experiment_name,
    r.pipeline_name,
    r.config_hash,
    r.started_at,
    r.completed_at,
    r.run_status,
    COUNT(DISTINCT s.sample_id) AS sample_count,
    AVG(ms.precision_at_k) AS avg_precision_at_k,
    AVG(ms.recall_at_k) AS avg_recall_at_k,
    AVG(ms.mrr) AS avg_mrr,
    AVG(ms.ndcg) AS avg_ndcg,
    AVG(ms.faithfulness) AS avg_faithfulness,
    AVG(ms.relevance) AS avg_relevance,
    AVG(ms.hallucination) AS avg_hallucination,
    AVG(ms.bert_score) AS avg_bert_score,
    SUM(COALESCE(ga.cost_usd, 0.0)) AS total_cost_usd,
    AVG(ga.latency_ms) AS avg_latency_ms,
    SUM(COALESCE(ga.prompt_tokens, 0)) AS prompt_tokens,
    SUM(COALESCE(ga.completion_tokens, 0)) AS completion_tokens
FROM runs AS r
LEFT JOIN samples AS s
    ON s.run_id = r.run_id
LEFT JOIN generated_answers AS ga
    ON ga.run_id = s.run_id
   AND ga.sample_id = s.sample_id
LEFT JOIN metric_scores AS ms
    ON ms.run_id = s.run_id
   AND ms.sample_id = s.sample_id
WHERE r.run_id IN ({run_id_placeholders})
GROUP BY
    r.run_id,
    r.experiment_name,
    r.pipeline_name,
    r.config_hash,
    r.started_at,
    r.completed_at,
    r.run_status
ORDER BY r.run_id DESC
"""

LOAD_RUN_SUMMARY = """
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

LOAD_QUESTION_TYPE_BREAKDOWN = """
SELECT
    s.run_id,
    s.question_type,
    COUNT(*) AS sample_count,
    AVG(ms.precision_at_k) AS avg_precision_at_k,
    AVG(ms.recall_at_k) AS avg_recall_at_k,
    AVG(ms.mrr) AS avg_mrr,
    AVG(ms.ndcg) AS avg_ndcg,
    AVG(ms.faithfulness) AS avg_faithfulness,
    AVG(ms.relevance) AS avg_relevance,
    AVG(ms.hallucination) AS avg_hallucination,
    AVG(ms.bert_score) AS avg_bert_score
FROM samples AS s
LEFT JOIN metric_scores AS ms
    ON ms.run_id = s.run_id
   AND ms.sample_id = s.sample_id
WHERE s.run_id IN ({run_id_placeholders})
GROUP BY s.run_id, s.question_type
ORDER BY s.run_id, s.question_type
"""

LOAD_FAILURE_BREAKDOWN = """
SELECT
    run_id,
    failure_mode,
    COUNT(*) AS failure_count
FROM failure_labels
WHERE run_id IN ({run_id_placeholders})
GROUP BY run_id, failure_mode
ORDER BY run_id, failure_count DESC, failure_mode
"""

LOAD_COST_LATENCY_COMPARISON = """
SELECT
    s.run_id,
    COUNT(*) AS sample_count,
    SUM(COALESCE(ga.cost_usd, 0.0)) AS total_cost_usd,
    AVG(ga.cost_usd) AS avg_cost_usd,
    AVG(ga.latency_ms) AS avg_latency_ms,
    MIN(ga.latency_ms) AS min_latency_ms,
    MAX(ga.latency_ms) AS max_latency_ms,
    SUM(COALESCE(ga.prompt_tokens, 0)) AS prompt_tokens,
    SUM(COALESCE(ga.completion_tokens, 0)) AS completion_tokens
FROM samples AS s
LEFT JOIN generated_answers AS ga
    ON ga.run_id = s.run_id
   AND ga.sample_id = s.sample_id
WHERE s.run_id IN ({run_id_placeholders})
GROUP BY s.run_id
ORDER BY s.run_id
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
