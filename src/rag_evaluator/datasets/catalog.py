from __future__ import annotations

import typing
from pathlib import Path

from rag_evaluator.datasets.config import DatasetCatalog, DatasetConfig, DatasetSource
from rag_evaluator.io import load_yaml

DEFAULT_DATASET_CATALOG_PATH = Path("configs/datasets.yaml")

def load_dataset_catalog(path: str | Path = DEFAULT_DATASET_CATALOG_PATH) -> DatasetCatalog:
    """
    Load and validate the dataset catalog YAML.
    """

    data = load_yaml(path)
    return DatasetCatalog.model_validate(data)


def resolve_dataset_config(
    dataset_ref: str,
    *,
    catalog: DatasetCatalog | None = None,
    catalog_path: str | Path = DEFAULT_DATASET_CATALOG_PATH,
    split: str | None = None,
    sample_limit: int | None = None,
    metadata: dict[str, typing.Any] | None = None,
) -> DatasetConfig:
    """
    Resolve a catalog dataset key into a DatasetConfig.
    :param dataset_ref:
    :param catalog:
    :param catalog_path:
    :param split:
    :param sample_limit:
    :param metadata:
    :return:
    """
    
    active_catalog = catalog or load_dataset_catalog(catalog_path)

    if dataset_ref not in active_catalog.datasets:
        known_refs = ", ".join(sorted(active_catalog.datasets))
        raise ValueError(f"Unknown dataset ref '{dataset_ref}'. Available datasets: {known_refs}")

    entry = active_catalog.datasets[dataset_ref]
    resolved_split = split or entry.default_split
    resolved_metadata: dict[str, typing.Any] = {
        "dataset_ref": dataset_ref,
        "display_name": entry.display_name,
        "url": entry.url,
        "domain": entry.domain,
        "normalizer": entry.normalizer,
        "local_normalized_path": entry.local_normalized_path,
        **entry.metadata,
    }

    if metadata:
        resolved_metadata.update(metadata)

    return DatasetConfig(
        name=dataset_ref,
        source=entry.source,
        path=entry.local_normalized_path if entry.source == DatasetSource.LOCAL_JSONL else None,
        dataset_name=entry.dataset_name,
        dataset_config=entry.dataset_config,
        split=resolved_split,
        question_type=entry.question_types[0] if len(entry.question_types) == 1 else None,
        domain=entry.domain,
        sample_limit=sample_limit,
        metadata=resolved_metadata,
    )
