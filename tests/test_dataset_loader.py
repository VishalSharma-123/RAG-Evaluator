from __future__ import annotations

import pytest

from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.loader import (
    load_dataset_from_config,
    load_dataset_samples,
    validate_unique_sample_ids,
)


def test_load_dataset_from_config_uses_built_adapter(monkeypatch, make_sample) -> None:
    expected = [make_sample()]

    class FakeAdapter:
        def load(self):
            return expected

    monkeypatch.setattr(
        "rag_evaluator.datasets.loader.build_dataset_adapter",
        lambda config: FakeAdapter(),
    )

    config = DatasetConfig(
        name="unit",
        source=DatasetSource.LOCAL_JSONL,
        path="data/unit.jsonl",
        split="test",
    )

    assert load_dataset_from_config(config) == expected


def test_load_dataset_samples_resolves_and_loads(monkeypatch, make_sample) -> None:
    config = DatasetConfig(
        name="unit",
        source=DatasetSource.LOCAL_JSONL,
        path="data/unit.jsonl",
        split="test",
    )
    expected = [make_sample()]

    monkeypatch.setattr(
        "rag_evaluator.datasets.loader.resolve_dataset_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "rag_evaluator.datasets.loader.load_dataset_from_config",
        lambda _config: expected,
    )

    assert load_dataset_samples("unit") == expected


def test_validate_unique_sample_ids_rejects_duplicates(make_sample) -> None:
    samples = [
        make_sample(sample_id="duplicate"),
        make_sample(sample_id="unique"),
        make_sample(sample_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_unique_sample_ids(samples, dataset_name="unit")
