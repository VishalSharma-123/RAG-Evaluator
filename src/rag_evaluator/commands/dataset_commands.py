from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from rag_evaluator.datasets import load_dataset_catalog, resolve_dataset_config
from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.loader import load_dataset_from_config
from rag_evaluator.datasets.writer import write_normalized_dataset
from rag_evaluator.io import export_json_schema, load_eval_samples_jsonl, load_experiment_config


def validate_dataset(path: Path) -> int:
    samples = load_eval_samples_jsonl(path)
    print(f"Validated {len(samples)} EvalSample record from {path}")
    return 0


def validate_config(path: Path) -> int:
    config = load_experiment_config(path)
    print(f"Validated experiment config: {config.experiment_name}")
    print(f"Datasets: {len(config.datasets)}")
    print(f"Pipelines: {len(config.pipelines)}")
    return 0


def export_schema_command(model: type[BaseModel] | None, output_path: Path, model_name: str) -> int:
    export_json_schema(model, output_path)
    print(f"Exported {model_name} JSON schema to {output_path}")
    return 0


def list_datasets(catalog_path: Path) -> int:
    catalog = load_dataset_catalog(catalog_path)

    for dataset_ref, entry in sorted(catalog.datasets.items()):
        print(f"{dataset_ref}: {entry.display_name} [{entry.source}]")

    return 0


def normalize_dataset(
    dataset_ref: str,
    *,
    catalog_path: Path,
    split: str | None,
    sample_limit: int | None,
    output_path: Path | None,
) -> int:
    config = resolve_dataset_config(
        dataset_ref,
        catalog_path=catalog_path,
        split=split,
        sample_limit=sample_limit,
    )
    samples = load_dataset_from_config(config)

    destination = output_path
    if destination is None:
        destination = Path(config.metadata["local_normalized_path"])

    manifest_path = destination.parent / "manifest.yaml"
    write_normalized_dataset(
        samples,
        output_path=destination,
        manifest_path=manifest_path,
        manifest={
            "dataset_ref": dataset_ref,
            "dataset_name": config.dataset_name,
            "dataset_config": config.dataset_config,
            "split": config.split,
            "source": config.source.value,
            "domain": config.domain,
            "sample_limit": config.sample_limit,
        },
    )

    print(f"Normalized {len(samples)} samples for {dataset_ref}")
    print(f"Wrote dataset to {destination}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def smoke_normalize_sources(
    *,
    catalog_path: Path,
    sample_limit: int,
    output_dir: Path | None,
) -> int:
    os.environ.setdefault("USE_TORCH", "0")
    catalog = load_dataset_catalog(catalog_path)
    example_path = Path("examples/eval_samples.jsonl")

    configs: list[tuple[str, DatasetConfig]] = [
        (
            "local_jsonl",
            DatasetConfig(
                name="example_local",
                source=DatasetSource.LOCAL_JSONL,
                path=str(example_path),
                split="example",
                sample_limit=sample_limit,
            ),
        ),
        (
            "huggingface",
            resolve_dataset_config(
                "natural_questions",
                catalog=catalog,
                sample_limit=sample_limit,
            ),
        ),
        (
            "github",
            DatasetConfig(
                name="rgb_smoke",
                source=DatasetSource.GITHUB,
                path=str(example_path),
                split="test",
                sample_limit=sample_limit,
                metadata={
                    "normalizer": "rgb",
                    "local_normalized_path": str(example_path),
                },
            ),
        ),
        (
            "ragas",
            DatasetConfig(
                name="ragas_smoke",
                source=DatasetSource.RAGAS,
                path=str(example_path),
                split="generated",
                sample_limit=sample_limit,
                metadata={
                    "normalizer": "ragas",
                    "local_normalized_path": str(example_path),
                },
            ),
        ),
    ]

    failures: list[tuple[str, str]] = []
    for label, config in configs:
        try:
            samples = load_dataset_from_config(config)
            print(f"{label}: normalized {len(samples)} samples")

            if output_dir is not None:
                destination = output_dir / f"{label}.jsonl"
                write_normalized_dataset(samples, output_path=destination)
                print(f"{label}: wrote {destination}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((label, message))
            print(f"{label}: FAILED - {message}")

    if failures:
        print("Smoke normalization completed with failures:")
        for label, message in failures:
            print(f"- {label}: {message}")
        return 1

    print("Smoke normalization completed successfully for all source types.")
    return 0
