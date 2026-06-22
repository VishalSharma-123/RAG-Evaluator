# RAG Evaluator

A config-driven framework for stress-testing RAG pipelines across retrieval quality, generation faithfulness, hallucination rate, cost, and latency.

The framework is designed around one rule: report retrieval and generation quality separately, and always break results down by evaluation stratum such as question type, dataset, and pipeline configuration.

## Architecture

```text
experiment.yaml
      |
      v
Ingestion -> Retrieval Index -> Pipeline Runner -> Scorer Engine -> DuckDB -> Streamlit
     ^                                ^
     |                                |
Datasets / Corpus ------------ EvalSample
```

## Core Layers

1. Ingestion
   - Chunkers: fixed, sentence, semantic, late chunking.
   - Embedders: OpenAI, BGE, Cohere.
   - Indexes: vector store plus BM25.

2. Dataset and Synthetic QA
   - Normalize all datasets into `EvalSample`.
   - Generate `(question, answer, evidence_chunk_id)` triples from corpus chunks.
   - Supported question types: factoid, multi-hop, abstractive, adversarial, comparative, unanswerable.

3. Pipeline Runner
   - Read `experiment.yaml`.
   - Run controlled sweeps over chunker, embedder, retriever, reranker, and generator variants.
   - Verify one small end-to-end run before launching full sweeps.

4. Scorer Engine
   - Retrieval metrics: Precision@k, Recall@k, MRR, NDCG.
   - Generation metrics: faithfulness, relevance, hallucination, BERTScore.
   - Failure modes: `RETRIEVAL_MISS`, `RETRIEVAL_RANK`, `CONTEXT_IGNORED`, `HALLUCINATION`, `PARTIAL_ANSWER`, `UNANSWERABLE_FAIL`.

5. Results Store and Dashboard
   - Store runs, configs, samples, retrieved chunks, generated answers, metrics, failures, costs, and latency in DuckDB.
   - Compare runs and inspect per-question failures in Streamlit.

## First Implementation Phase

Phase 0 establishes the project foundation:

1. Package scaffold.
2. Shared Pydantic schemas.
3. YAML and JSONL loading helpers.
4. Schema validation tests.

Recommended file order:

1. `pyproject.toml`
2. `README.md`
3. `src/rag_evaluator/__init__.py`
4. `src/rag_evaluator/schemas.py`
5. `src/rag_evaluator/config.py`
6. `src/rag_evaluator/io.py`
7. `tests/test_schemas.py`
8. `tests/test_config.py`

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Optional retrieval and LLM dependencies can be installed when those layers are implemented:

```bash
python -m pip install -e ".[retrieval,llm]"
```

## Metric Warnings

- Do not mix dataset schemas before normalizing them to `EvalSample`.
- Do not conflate retrieval failures with generation failures.
- MRR rewards rank-1 hits heavily; pair it with Recall@k and NDCG when evidence coverage matters.
- LLM-as-judge prompts must use concrete rubrics and strict JSON output.
- Aggregate-only reporting is insufficient for RAG evaluation because it hides stratum-specific failure modes.
