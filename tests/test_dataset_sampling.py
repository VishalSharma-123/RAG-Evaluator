from __future__ import annotations

import pytest

from rag_evaluator.datasets.sampling import stratified_sample
from rag_evaluator.schemas import QuestionType


def test_stratified_sample_limits_per_group(make_sample) -> None:
    samples = [
        make_sample(sample_id="a", source_dataset="ds1", question_type=QuestionType.FACTOID),
        make_sample(sample_id="b", source_dataset="ds1", question_type=QuestionType.FACTOID),
        make_sample(sample_id="c", source_dataset="ds2", question_type=QuestionType.MULTI_HOP),
        make_sample(sample_id="d", source_dataset="ds2", question_type=QuestionType.MULTI_HOP),
    ]

    selected = stratified_sample(samples, per_group=1, seed=7)

    assert len(selected) == 2
    assert {sample.source_dataset for sample in selected} == {"ds1", "ds2"}


def test_stratified_sample_rejects_invalid_key(make_sample) -> None:
    with pytest.raises(ValueError, match="Unknown stratification key"):
        stratified_sample([make_sample()], per_group=1, keys=("missing_key",))
