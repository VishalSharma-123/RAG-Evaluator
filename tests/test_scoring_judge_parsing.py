from __future__ import annotations

import pytest

from rag_evaluator.scoring.judges.base import JudgeScoringError
from rag_evaluator.scoring.judges.parsing import parse_judge_response


def test_parse_judge_response_parses_metrics() -> None:
    metrics = parse_judge_response(
        """
        {
          "faithfulness": 0.9,
          "relevance": 0.8,
          "hallucination": 0.1,
          "bert_score": null
        }
        """
    )

    assert metrics.faithfulness == 0.9
    assert metrics.relevance == 0.8
    assert metrics.hallucination == 0.1
    assert metrics.bert_score is None


@pytest.mark.parametrize(
    ("raw_response", "match"),
    [
        ("not json", "not valid JSON"),
        ("[]", "valid JSON object"),
        ('{"faithfulness": 1.2, "relevance": 0.8, "hallucination": 0.1}', "between 0 and 1"),
        ('{"relevance": 0.8, "hallucination": 0.1}', "missing `faithfulness`"),
    ],
)
def test_parse_judge_response_rejects_invalid_payloads(raw_response: str, match: str) -> None:
    with pytest.raises(JudgeScoringError, match=match):
        parse_judge_response(raw_response)
