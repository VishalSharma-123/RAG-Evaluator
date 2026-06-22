from __future__ import annotations

from pathlib import Path

from rag_evaluator.datasets.adapters import DatasetAdapter
from rag_evaluator.datasets.catalog import resolve_dataset_config
from rag_evaluator.datasets.config import DatasetConfig
from rag_evaluator.datasets.registry import build_dataset_adapter
from rag_evaluator.schemas import EvalSample


def load_dataset_samples(
    dataset_ref: str,
    *,
    catalog_path: str | Path = "configs/datasets.yaml",
    split: str | None = None,
    sample_limit: int | None = None,
) -> list[EvalSample]:
    """
    Resolve a dataset ref, build its adapter, and load EvalSample records.
    :param dataset_ref:
    :param catalog_path:
    :param split:
    :param sample_limit:
    :return:
    """
    config = resolve_dataset_config(
        dataset_ref,
        catalog_path=catalog_path,
        split=split,
        sample_limit=sample_limit,
    )
    return load_dataset_from_config(config)


def load_dataset_from_config(config: DatasetConfig) -> list[EvalSample]:
    """
    Build an adapter from a resolved config and load samples.
    :param config:
    :return:
    """

    adapter: DatasetAdapter = build_dataset_adapter(config)
    return adapter.load()
