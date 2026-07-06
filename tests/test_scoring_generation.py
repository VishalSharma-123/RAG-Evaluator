from __future__ import annotations

from rag_evaluator.schemas import GeneratedAnswer
from rag_evaluator.scoring.generation import score_generation, score_generation_batch


def test_score_generation_returns_none_for_missing_answer(make_sample) -> None:
    sample = make_sample()

    assert score_generation(sample, None) is None


def test_score_generation_returns_zero_metrics_for_blank_answer(make_sample) -> None:
    sample = make_sample()
    generated_answer = GeneratedAnswer(
        sample_id=sample.sample_id,
        answer="   ",
        model_name="unit-test",
    )

    metrics = score_generation(
        sample,
        generated_answer,
    )

    assert metrics is not None
    assert metrics.faithfulness == 0.0
    assert metrics.relevance == 0.0
    assert metrics.hallucination == 1.0
    assert metrics.bert_score == 0.0


def test_score_generation_batch_returns_metrics_by_sample_id(make_sample) -> None:
    sample = make_sample()

    batch = score_generation_batch([sample], {sample.sample_id: None})

    assert sample.sample_id in batch
    assert batch[sample.sample_id] is None
