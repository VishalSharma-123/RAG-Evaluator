from __future__ import annotations

import pytest

from rag_evaluator.datasets.catalog import load_dataset_catalog, resolve_dataset_config


def test_load_dataset_catalog_reads_real_config() -> None:
    catalog = load_dataset_catalog("configs/datasets.yaml")

    assert "natural_questions" in catalog.datasets


def test_resolve_dataset_config_uses_expected_defaults() -> None:
    config = resolve_dataset_config("natural_questions", catalog_path="configs/datasets.yaml")

    assert config.name == "natural_questions"
    assert config.split == "validation"
    assert config.metadata["normalizer"] == "natural_questions"


def test_resolve_dataset_config_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset ref"):
        resolve_dataset_config("missing_dataset", catalog_path="configs/datasets.yaml")
