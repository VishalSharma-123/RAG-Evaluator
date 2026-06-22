from __future__ import annotations

import argparse
import os
from pathlib import Path

from pydantic import BaseModel

from rag_evaluator.config import ExperimentConfig
from rag_evaluator.datasets import load_dataset_catalog, resolve_dataset_config
from rag_evaluator.datasets.config import DatasetConfig, DatasetSource
from rag_evaluator.datasets.loader import load_dataset_from_config
from rag_evaluator.datasets.writer import write_normalized_dataset
from rag_evaluator.ingestion.chunkers import SourceDocument, build_chunker
from rag_evaluator.io import (
    export_json_schema,
    load_eval_samples_jsonl,
    load_experiment_config,
    write_jsonl,
)
from rag_evaluator.schemas import (
    Chunk,
    EvalResult,
    EvalSample,
    GeneratedAnswer,
    RetrievedChunk,
)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "eval-sample": EvalSample,
    "chunk": Chunk,
    "retrieved-chunk": RetrievedChunk,
    "generated-answer": GeneratedAnswer,
    "eval-result": EvalResult,
    "experiment-config": ExperimentConfig,
}

def validate_dataset(path: Path) -> int:
    sample = load_eval_samples_jsonl(path)
    print(f"Validated {len(sample)} EvalSample record from {path}")
    return 0
    
def validate_config(path: Path) -> int:
    config = load_experiment_config(path)
    print(f"Validated experiment config: {config.experiment_name}")
    print(f"Datasets: {len(config.datasets)}")
    print(f"Pipelines: {len(config.pipelines)}")
    return 0

def export_schema(model_name: str, output_path: Path) -> int:
    model = SCHEMA_MODELS.get(model_name)
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


def chunk_text(
        *,
        chunker_type: str,
        chunk_size: int | None,
        chunk_overlap: int,
        text: str | None,
        input_path: Path | None,
        document_id: str,
        output_path: Path | None,
) -> int:
    if text is None and input_path is None:
        raise ValueError("Provide either --text or --input.")

    if text is not None and input_path is not None:
        raise ValueError("Use only one of --text or --input.")

    content = text
    if input_path is not None:
        content = input_path.read_text(encoding="utf-8")

    assert content is not None

    chunker = build_chunker(
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document = SourceDocument(
        document_id=document_id,
        text=content,
        metadata={
            "source_path": str(input_path) if input_path is not None else None,
        },
    )
    chunks = chunker.chunk([document])

    if output_path is not None:
        write_jsonl(output_path, chunks)
        print(f"Wrote {len(chunks)} chunks to {output_path}")
        return 0

    print(f"Chunked document into {len(chunks)} chunks using {chunker_type}")
    for chunk in chunks:
        print(f"--- {chunk.chunk_id} [{chunk.start_char}, {chunk.end_char}] ---")
        print(chunk.text)
        print()

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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-evaluator",
        description="CLI helpers for RAG Evaluator framework",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a normalized EvalSample JSONL dataset.",
    )
    dataset_parser.add_argument(
        "path",
        type = Path,
        help = "Path to a JSONL file containing one EvalSample per line."
    )
    
    config_parser = subparsers.add_parser(
        "validate-config",
        help = "Validate an experiment YAML config file."
    )
    config_parser.add_argument(
        "path",
        type = Path,
        help = "Path to a YAML file containing an experiment config."
    )
    
    schema_parser = subparsers.add_parser(
        "export-schema",
        help = "Export a JSON Schema for a public model"
    )
    schema_parser.add_argument(
        "model",
        choices = sorted(SCHEMA_MODELS),
        help = "Schema model to export"
    )
    schema_parser.add_argument(
        "output_path",
        type = Path,
        help = "Destination path to export schema"
    )

    list_parser = subparsers.add_parser(
        "list-datasets",
        help="List datasets from the dataset catalog.",
    )
    list_parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )

    normalize_parser = subparsers.add_parser(
        "normalize-dataset",
        help="Load a dataset from the catalog and write normalized EvalSample JSONL.",
    )
    normalize_parser.add_argument(
        "dataset_ref",
        help="Dataset reference key from configs/datasets.yaml.",
    )
    normalize_parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    normalize_parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split override.",
    )
    normalize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit.",
    )
    normalize_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path override.",
    )

    chunk_parser = subparsers.add_parser(
        "chunk-text",
        help="Chunk a text file or inline text and print or save Chunk records.",
    )
    chunk_parser.add_argument(
        "--type",
        dest="chunker_type",
        choices=["fixed", "sentence", "semantic", "late"],
        required=True,
        help="Chunking strategy to use.",
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size.",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Chunk overlap. For sentence/semantic this is sentence overlap; for late this is context sentences.",
    )
    source_group = chunk_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--text",
        type=str,
        default=None,
        help="Inline text to chunk.",
    )
    source_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a UTF-8 text file to chunk.",
    )
    chunk_parser.add_argument(
        "--document-id",
        type=str,
        default="document",
        help="Document ID to attach to produced chunks.",
    )
    chunk_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path for produced chunks.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-normalize-sources",
        help="Run a small normalization pass for each supported dataset source type.",
    )
    smoke_parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    smoke_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of samples to normalize per source type.",
    )
    smoke_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write normalized JSONL outputs for each source type.",
    )
    
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command == "validate-dataset":
        return validate_dataset(args.path)
    
    if args.command == "validate-config":
        return validate_config(args.path)
    
    if args.command == "export-schema":
        return export_schema(args.model, args.output_path)

    if args.command == "list-datasets":
        return list_datasets(args.catalog)

    if args.command == "normalize-dataset":
        return normalize_dataset(
            args.dataset_ref,
            catalog_path=args.catalog,
            split=args.split,
            sample_limit=args.limit,
            output_path=args.output,
        )

    if args.command == "chunk-text":
        return chunk_text(
            chunker_type=args.chunker_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            text=args.text,
            input_path=args.input,
            document_id=args.document_id,
            output_path=args.output,
        )

    if args.command == "smoke-normalize-sources":
        return smoke_normalize_sources(
            catalog_path=args.catalog,
            sample_limit=args.limit,
            output_dir=args.output_dir,
        )
    
    parser.error(f"Unknown command: {args.command}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
