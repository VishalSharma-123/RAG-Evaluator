from __future__ import annotations

import argparse

from rag_evaluator.commands.chunk_commands import build_chunks, write_chunks
from rag_evaluator.commands.dataset_commands import (
    export_schema_command,
    list_datasets,
    normalize_dataset,
    smoke_normalize_sources,
    validate_config,
    validate_dataset,
)
from rag_evaluator.commands.experiment_commands import run_experiment
from rag_evaluator.commands.renderers import (
    render_chunk_output,
    render_experiment_summary,
    render_synthetic_summary,
)
from rag_evaluator.commands.schema import SCHEMA_MODELS
from rag_evaluator.commands.synthetic_commands import generate_synthetic


def handle_validate_dataset(args: argparse.Namespace) -> int:
    return validate_dataset(args.path)


def handle_validate_config(args: argparse.Namespace) -> int:
    return validate_config(args.path)


def handle_export_schema(args: argparse.Namespace) -> int:
    return export_schema_command(SCHEMA_MODELS.get(args.model), args.output_path, args.model)


def handle_list_datasets(args: argparse.Namespace) -> int:
    return list_datasets(args.catalog)


def handle_normalize_dataset(args: argparse.Namespace) -> int:
    return normalize_dataset(
        args.dataset_ref,
        catalog_path=args.catalog,
        split=args.split,
        sample_limit=args.limit,
        output_path=args.output,
    )


def handle_chunk_text(args: argparse.Namespace) -> int:
    chunks = build_chunks(
        chunker_type=args.chunker_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        text=args.text,
        input_path=args.input,
        document_id=args.document_id,
    )

    if args.output is not None:
        return write_chunks(args.output, chunks)

    render_chunk_output(chunks, args.chunker_type)
    return 0


def handle_smoke_normalize_sources(args: argparse.Namespace) -> int:
    return smoke_normalize_sources(
        catalog_path=args.catalog,
        sample_limit=args.limit,
        output_dir=args.output_dir,
    )


def handle_generate_synthetic(args: argparse.Namespace) -> int:
    if args.config is None and (args.chunks is None or args.output is None):
        args.command_parser.error(
            "generate-synthetic requires either --config or both --chunks and --output."
        )

    summary = generate_synthetic(
        config_path=args.config,
        chunks_path=args.chunks,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        question_types=args.question_types,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_enabled=args.reasoning_enabled,
    )
    render_synthetic_summary(summary)
    return 0


def handle_run_experiment(args: argparse.Namespace) -> int:
    summary = run_experiment(
        config_path=args.config,
        database_path=args.database_path,
    )
    render_experiment_summary(summary)
    return 0
