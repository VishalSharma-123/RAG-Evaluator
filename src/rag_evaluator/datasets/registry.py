from __future__ import annotations

from rag_evaluator.datasets.adapters import (
    DatasetAdapter,
    GitHubDatasetAdapter,
    HuggingFaceDatasetAdapter,
    LocalJSONLDatasetAdapter,
    RagasTestsetAdapter,
)
from rag_evaluator.datasets.config import DatasetConfig, DatasetSource


def build_dataset_adapter(config: DatasetConfig) -> DatasetAdapter:
    """
    Build the dataset adapter for a resolved dataset config.
    :param config:
    :return:
    """

    if config.source == DatasetSource.LOCAL_JSONL:
        return LocalJSONLDatasetAdapter(config)

    if config.source == DatasetSource.HUGGINGFACE:
        return HuggingFaceDatasetAdapter(config)

    if config.source == DatasetSource.GITHUB:
        return GitHubDatasetAdapter(config)

    if config.source == DatasetSource.RAGAS:
        return RagasTestsetAdapter(config)

    raise ValueError(f"Unsupported dataset source: {config.source}")
