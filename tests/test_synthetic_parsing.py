from __future__ import annotations

import pytest

from rag_evaluator.synthetic.errors import SyntheticParsingError
from rag_evaluator.synthetic.parsing import parse_synthetic_generation_response


def test_parse_synthetic_generation_response_applies_defaults() -> None:
    samples = parse_synthetic_generation_response(
        """
        {
          "samples": [
            {
              "sample_id": "s1",
              "question": "What is the capital of France?",
              "reference_answer": "Paris",
              "question_type": "factoid",
              "evidence_chunk_ids": ["doc:chunk:0"],
              "is_answerable": true
            }
          ]
        }
        """
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.source_dataset == "synthetic"
    assert sample.source_split == "generated"
    assert sample.source_id == "s1"
    assert sample.answer_aliases == []
    assert sample.choices == []
    assert sample.metadata == {}


def test_parse_synthetic_generation_response_rejects_invalid_json() -> None:
    with pytest.raises(SyntheticParsingError, match="not valid JSON"):
        parse_synthetic_generation_response("not-json")


def test_parse_synthetic_generation_response_requires_samples_list() -> None:
    with pytest.raises(SyntheticParsingError, match="`samples` list"):
        parse_synthetic_generation_response("{}")


def test_parse_synthetic_generation_response_rejects_invalid_sample_shape() -> None:
    with pytest.raises(SyntheticParsingError, match="Invalid synthetic sample"):
        parse_synthetic_generation_response(
            """
            {
              "samples": [
                {
                  "sample_id": "s1"
                }
              ]
            }
            """
        )
