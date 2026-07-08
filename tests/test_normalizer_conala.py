from __future__ import annotations

from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.normalizers.code.conala import ConalaNormalizer


def test_conala_sample_id_includes_index_when_source_id_repeats() -> None:
    normalizer = ConalaNormalizer(
        DatasetConfig(
            name="conala",
            source=DatasetSource.HUGGINGFACE,
            dataset_name="neulab/conala",
            split="train",
        )
    )
    record = {
        "question_id": 41067960,
        "intent": "sort a list",
        "snippet": "items.sort()",
    }

    first = normalizer.normalize_record(record, index=0, split="train")
    second = normalizer.normalize_record(record, index=1, split="train")

    assert first.source_id == "41067960"
    assert first.sample_id == "conala:train:41067960:0"
    assert second.sample_id == "conala:train:41067960:1"
