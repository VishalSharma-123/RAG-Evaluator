from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_evaluator.config import PipelineConfig
from rag_evaluator.reranking.cohere import CohereReranker
from rag_evaluator.reranking.cross_encoder import CrossEncoderReranker
from rag_evaluator.reranking.factory import PassThroughReranker, build_reranker
from rag_evaluator.reranking.openrouter import OpenRouterReranker


def _build_pipeline(**overrides: object) -> PipelineConfig:
    payload: dict[str, object] = {
        "name": "pipeline-1",
        "chunker": {"type": "fixed", "chunk_size": 128},
        "embedder": {"provider": "bge", "model": "BAAI/bge-small-en-v1.5"},
        "retriever": {"type": "vector", "top_k": 3},
    }
    payload.update(overrides)
    return PipelineConfig.model_validate(payload)


def test_build_reranker_returns_pass_through_for_none() -> None:
    pipeline = _build_pipeline()

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, PassThroughReranker)
    assert reranker.configured_type == "none"
    assert reranker.implementation_name == "pass_through"
    assert reranker.implemented is False


def test_pass_through_reranker_trims_and_renumbers(make_chunk, make_retrieved_chunk) -> None:
    pipeline = _build_pipeline()
    reranker = build_reranker(pipeline)

    chunks = [
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:1"), rank=7, score=0.7),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:2"), rank=8, score=0.6),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:3"), rank=9, score=0.5),
    ]

    reranked = reranker.rerank(None, chunks, top_k=2)

    assert [item.chunk.chunk_id for item in reranked] == ["doc:chunk:1", "doc:chunk:2"]
    assert [item.rank for item in reranked] == [1, 2]


def test_build_reranker_builds_cross_encoder_reranker(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

        def predict(self, sentences, **kwargs: object):
            captured["sentences"] = list(sentences)
            captured["predict_kwargs"] = kwargs
            return [0.2, 0.9, 0.5]

    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)

    pipeline = _build_pipeline(
        reranker={
            "type": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 2,
            "metadata": {
                "batch_size": 8,
                "device": "cpu",
                "show_progress_bar": True,
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunks = [
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:1")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:2")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:3")),
    ]
    reranked = reranker.rerank(make_sample(), chunks, top_k=2)

    assert captured["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["predict_kwargs"]["show_progress_bar"] is True
    assert captured["predict_kwargs"]["batch_size"] == 8
    assert [item.chunk.chunk_id for item in reranked] == [
        "doc:chunk:2",
        "doc:chunk:3",
    ]
    assert [item.rank for item in reranked] == [1, 2]
    assert captured["sentences"] == [
        ("What is RAG?", "Retrieval augmented generation uses retrieved context."),
        ("What is RAG?", "Retrieval augmented generation uses retrieved context."),
        ("What is RAG?", "Retrieval augmented generation uses retrieved context."),
    ]


def test_build_reranker_reads_cross_encoder_model_from_metadata(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

        def predict(self, sentences, **kwargs: object):
            return [0.1 for _ in sentences]

    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)

    pipeline = _build_pipeline(
        reranker={
            "type": "cross_encoder",
            "metadata": {
                "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    sample = make_sample()
    reranker.rerank(
        sample,
        [make_retrieved_chunk(chunk=make_chunk(text="retrieved text"))],
        top_k=1,
    )

    assert captured["model_name"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"


def test_build_reranker_builds_cohere_reranker(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClientV2:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def rerank(self, *, model: str, query: str, documents: list[str], top_n: int):
            captured["rerank_kwargs"] = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=2, relevance_score=0.9),
                    SimpleNamespace(index=1, relevance_score=0.8),
                    SimpleNamespace(index=0, relevance_score=0.4),
                ]
            )

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr("cohere.ClientV2", FakeClientV2)

    pipeline = _build_pipeline(
        reranker={
            "type": "cohere",
            "model": "rerank-v3.5",
            "top_k": 2,
            "metadata": {
                "api_key_env": "COHERE_API_KEY",
                "base_url": "https://example.invalid",
                "client_name": "unit-tests",
                "timeout": 12.5,
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, CohereReranker)
    assert reranker.model_name == "rerank-v3.5"

    chunks = [
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:1")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:2")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:3")),
    ]
    reranked = reranker.rerank(make_sample(), chunks, top_k=2)

    assert captured["client_kwargs"]["base_url"] == "https://example.invalid"
    assert captured["client_kwargs"]["client_name"] == "unit-tests"
    assert captured["client_kwargs"]["timeout"] == 12.5
    assert captured["rerank_kwargs"] == {
        "model": "rerank-v3.5",
        "query": "What is RAG?",
        "documents": [
            "Retrieval augmented generation uses retrieved context.",
            "Retrieval augmented generation uses retrieved context.",
            "Retrieval augmented generation uses retrieved context.",
            ],
            "top_n": 2,
        }
    assert [item.chunk.chunk_id for item in reranked] == [
        "doc:chunk:3",
        "doc:chunk:2",
    ]
    assert [item.rank for item in reranked] == [1, 2]


def test_build_reranker_reads_cohere_model_from_metadata(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClientV2:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def rerank(self, *, model: str, query: str, documents: list[str], top_n: int):
            captured["rerank_kwargs"] = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
            return SimpleNamespace(results=[])

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setattr("cohere.ClientV2", FakeClientV2)

    pipeline = _build_pipeline(
        reranker={
            "type": "cohere",
            "metadata": {
                "model": "rerank-english-v3.0",
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, CohereReranker)
    assert reranker.model_name == "rerank-english-v3.0"

    sample = make_sample()
    reranker.rerank(
        sample,
        [make_retrieved_chunk(chunk=make_chunk(text="retrieved text"))],
        top_k=1,
    )

    assert captured["client_kwargs"]["api_key"] == "test-key"


def test_build_reranker_builds_openrouter_reranker(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.85},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "rag-evaluator-tests")
    monkeypatch.setattr("requests.post", fake_post)

    pipeline = _build_pipeline(
        reranker={
            "type": "openrouter",
            "model": "openrouter/rerank-test",
            "top_k": 2,
            "metadata": {
                "timeout_seconds": 12.5,
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, OpenRouterReranker)
    assert reranker.model_name == "openrouter/rerank-test"

    chunks = [
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:1")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:2")),
        make_retrieved_chunk(chunk=make_chunk(chunk_id="doc:chunk:3")),
    ]
    reranked = reranker.rerank(make_sample(), chunks, top_k=2)

    assert captured["url"] == "https://openrouter.ai/api/v1/rerank"
    assert captured["kwargs"]["timeout"] == 12.5
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer unit-test-key"
    assert captured["kwargs"]["headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["kwargs"]["headers"]["X-Title"] == "rag-evaluator-tests"
    assert captured["kwargs"]["json"] == {
        "model": "openrouter/rerank-test",
        "query": "What is RAG?",
        "documents": [
            "Retrieval augmented generation uses retrieved context.",
            "Retrieval augmented generation uses retrieved context.",
            "Retrieval augmented generation uses retrieved context.",
        ],
        "top_n": 2,
    }
    assert [item.chunk.chunk_id for item in reranked] == [
        "doc:chunk:3",
        "doc:chunk:2",
    ]
    assert [item.rank for item in reranked] == [1, 2]


def test_build_reranker_reads_openrouter_model_from_metadata(
    make_chunk,
    make_retrieved_chunk,
    make_sample,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    pipeline = _build_pipeline(
        reranker={
            "type": "openrouter",
            "metadata": {
                "model": "openrouter/rerank-compact",
                "api_key": "inline-key",
            },
        }
    )

    reranker = build_reranker(pipeline)

    assert isinstance(reranker, OpenRouterReranker)
    assert reranker.model_name == "openrouter/rerank-compact"

    reranker.rerank(
        make_sample(),
        [make_retrieved_chunk(chunk=make_chunk(text="retrieved text"))],
        top_k=1,
    )

    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer inline-key"


@pytest.mark.parametrize("reranker_type", ["cross_encoder", "cohere", "openrouter"])
def test_build_reranker_rejects_missing_model(
    reranker_type: str,
) -> None:
    pipeline = _build_pipeline(
        reranker={
            "type": reranker_type,
            "top_k": 2,
        }
    )

    with pytest.raises(ValueError, match="model name"):
        build_reranker(pipeline)
