CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    config_hash TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    question TEXT NOT NULL,
    question_type TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    source_split TEXT,
    reference_answer TEXT,
    is_answerable BOOLEAN NOT NULL,
    metadata_json TEXT,
    PRIMARY KEY (run_id, sample_id)
);

CREATE TABLE IF NOT EXISTS retrieved_chunks (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score DOUBLE NOT NULL,
    retriever_name TEXT NOT NULL,
    metadata_json TEXT,
    PRIMARY KEY (run_id, sample_id, rank)
);

CREATE TABLE IF NOT EXISTS generated_answers (
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

CREATE TABLE IF NOT EXISTS metric_scores (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    precision_at_k DOUBLE,
    recall_at_k DOUBLE,
    mrr DOUBLE,
    ndcg DOUBLE,
    faithfulness DOUBLE,
    relevance DOUBLE,
    hallucination DOUBLE,
    bert_score DOUBLE,
    PRIMARY KEY (run_id, sample_id)
);

CREATE TABLE IF NOT EXISTS failure_labels (
    run_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    failure_mode TEXT NOT NULL,
    PRIMARY KEY (run_id, sample_id, failure_mode)
);

CREATE INDEX IF NOT EXISTS idx_samples_run_id
    ON samples (run_id);

CREATE INDEX IF NOT EXISTS idx_samples_question_type
    ON samples (question_type);

CREATE INDEX IF NOT EXISTS idx_generated_answers_run_id
    ON generated_answers (run_id);

CREATE INDEX IF NOT EXISTS idx_metric_scores_run_id
    ON metric_scores (run_id);

CREATE INDEX IF NOT EXISTS idx_failure_labels_run_id
    ON failure_labels (run_id);

CREATE INDEX IF NOT EXISTS idx_failure_labels_failure_mode
    ON failure_labels (failure_mode);

CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_run_id
    ON retrieved_chunks (run_id);

CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_sample_id
    ON retrieved_chunks (sample_id);