from __future__ import annotations

import argparse
from pathlib import Path

from rag_evaluator.commands.handlers import (
    handle_chunk_text,
    handle_export_schema,
    handle_generate_synthetic,
    handle_list_datasets,
    handle_normalize_dataset,
    handle_run_experiment,
    handle_smoke_normalize_sources,
    handle_validate_config,
    handle_validate_dataset,
)
from rag_evaluator.commands.schema import SCHEMA_MODELS
from rag_evaluator.schemas import QuestionType

Subparsers = argparse._SubParsersAction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-evaluator",
        description="CLI helpers for RAG Evaluator framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_validate_dataset(subparsers)
    _register_validate_config(subparsers)
    _register_export_schema(subparsers)
    _register_list_datasets(subparsers)
    _register_normalize_dataset(subparsers)
    _register_chunk_text(subparsers)
    _register_smoke_normalize_sources(subparsers)
    _register_generate_synthetic(subparsers)
    _register_run_experiment(subparsers)
    return parser


def _register_validate_dataset(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a normalized EvalSample JSONL dataset.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a JSONL file containing one EvalSample per line.",
    )
    parser.set_defaults(handler=handle_validate_dataset)


def _register_validate_config(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "validate-config",
        help="Validate an experiment YAML config file.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a YAML file containing an experiment config.",
    )
    parser.set_defaults(handler=handle_validate_config)


def _register_export_schema(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "export-schema",
        help="Export a JSON Schema for a public model",
    )
    parser.add_argument(
        "model",
        choices=sorted(SCHEMA_MODELS),
        help="Schema model to export",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination path to export schema",
    )
    parser.set_defaults(handler=handle_export_schema)


def _register_list_datasets(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "list-datasets",
        help="List datasets from the dataset catalog.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    parser.set_defaults(handler=handle_list_datasets)


def _register_normalize_dataset(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "normalize-dataset",
        help="Load a dataset from the catalog and write normalized EvalSample JSONL.",
    )
    parser.add_argument(
        "dataset_ref",
        help="Dataset reference key from configs/datasets.yaml.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path override.",
    )
    parser.set_defaults(handler=handle_normalize_dataset)


def _register_chunk_text(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "chunk-text",
        help="Chunk a text file or inline text and print or save Chunk records.",
    )
    parser.add_argument(
        "--type",
        dest="chunker_type",
        choices=["fixed", "sentence", "semantic", "late"],
        required=True,
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help=(
            "Chunk overlap. For sentence/semantic this is sentence overlap; "
            "for late this is context sentences."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument(
        "--document-id",
        type=str,
        default="document",
        help="Document ID to attach to produced chunks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path for produced chunks.",
    )
    parser.set_defaults(handler=handle_chunk_text)


def _register_smoke_normalize_sources(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "smoke-normalize-sources",
        help="Run a small normalization pass for each supported dataset source type.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to the dataset catalog YAML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of samples to normalize per source type.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write normalized JSONL outputs for each source type.",
    )
    parser.set_defaults(handler=handle_smoke_normalize_sources)


def _register_generate_synthetic(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "generate-synthetic",
        help="Generate synthetic EvalSample JSONL from chunk JSONL using Nemotron.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment YAML with a `synthetic_generation` section.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to a JSONL file containing Chunk records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path for generated EvalSample records.",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai"],
        default="openrouter",
        help="LLM provider used to call the configured model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/nemotron-3-super-120b-a12b:free",
        help="Synthetic generation model name.",
    )
    parser.add_argument(
        "--question-type",
        dest="question_types",
        action="append",
        choices=[question_type.value for question_type in QuestionType],
        default=None,
        help="Allowed question type. Repeat to allow multiple types.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of synthetic samples to request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature for synthetic generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for synthetic generation.",
    )
    parser.add_argument(
        "--reasoning-enabled",
        action="store_true",
        help="Enable provider-side reasoning when supported.",
    )
    parser.set_defaults(
        handler=handle_generate_synthetic,
        command_parser=parser,
    )


def _register_run_experiment(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        "run-experiment",
        help="Run all configured pipelines for an experiment and persist results.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="DuckDB path used to persist experiment results.",
    )
    parser.set_defaults(handler=handle_run_experiment)
