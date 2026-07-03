from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from rag_evaluator.config import ExperimentConfig, PipelineConfig, SweepConfig
from rag_evaluator.execution.runner import run_pipeline_variant
from rag_evaluator.execution.runtime import build_pipeline_runtime
from rag_evaluator.execution.types import ExperimentRunOutput, PipelineRunOutput
from rag_evaluator.ingestion.chunkers import SourceDocument
from rag_evaluator.schemas import EvalSample


def run_experiment(
    *,
    experiment: ExperimentConfig,
    samples: list[EvalSample],
    documents: list[SourceDocument],
) -> ExperimentRunOutput:
    pipeline_runs: list[PipelineRunOutput] = []

    for pipeline in experiment.pipelines:
        pipeline_runs.append(
            _run_pipeline_variant(
                pipeline=pipeline,
                samples=samples,
                documents=documents,
                execution_stage="baseline",
            )
        )

        sweep_config = _resolve_sweep_config(pipeline)
        if sweep_config is None:
            continue

        for pipeline_variant in expand_pipeline_variants(pipeline):
            pipeline_runs.append(
                _run_pipeline_variant(
                    pipeline=pipeline_variant,
                    samples=samples,
                    documents=documents,
                    execution_stage="sweep_variant",
                )
            )

    return ExperimentRunOutput(
        experiment_name=experiment.experiment_name,
        pipeline_runs=pipeline_runs,
    )


def _run_pipeline_variant(
    *,
    pipeline: PipelineConfig,
    samples: list[EvalSample],
    documents: list[SourceDocument],
    execution_stage: str,
) -> PipelineRunOutput:
    runtime = build_pipeline_runtime(
        pipeline=pipeline,
        documents=documents,
    )
    runtime_metadata = {
        **runtime.metadata,
        "execution_stage": execution_stage,
    }
    try:
        runtime = replace(runtime, metadata=runtime_metadata)
    except TypeError:
        runtime_payload = dict(vars(runtime))
        runtime_payload["metadata"] = runtime_metadata
        runtime = SimpleNamespace(**runtime_payload)
    run_output = run_pipeline_variant(
        runtime=runtime,
        samples=samples,
    )
    return PipelineRunOutput(
        pipeline=run_output.pipeline,
        chunks=run_output.chunks,
        results=run_output.results,
        runtime_metadata={
            **run_output.runtime_metadata,
            "execution_stage": execution_stage,
        },
    )


def expand_pipeline_variants(pipeline: PipelineConfig) -> list[PipelineConfig]:
    sweep_config = _resolve_sweep_config(pipeline)
    if sweep_config is None:
        return [pipeline]

    base_payload = pipeline.model_dump(mode="json")
    variants: list[PipelineConfig] = []

    for index, override in enumerate(sweep_config.overrides, start=1):
        if not isinstance(override, dict):
            raise ValueError("Each sweep override must be a dictionary.")

        variant_payload = _deep_merge_dicts(base_payload, override)
        variant_payload["metadata"] = {
            **base_payload.get("metadata", {}),
            **sweep_config.metadata,
            **override.get("metadata", {}),
            "source_pipeline_name": pipeline.name,
            "sweep_index": index,
        }

        name_suffix = _resolve_name_suffix(
            pipeline,
            sweep_config,
            override,
            index=index,
        )
        variant_payload["name"] = f"{pipeline.name}__{name_suffix}"
        variant_sweep_payload = sweep_config.model_dump(mode="json")
        variant_sweep_payload["enabled"] = False
        variant_sweep_payload["overrides"] = []
        variant_sweep_payload["metadata"] = sweep_config.metadata
        variant_payload["sweep"] = variant_sweep_payload
        variants.append(PipelineConfig.model_validate(variant_payload))

    return variants


def _resolve_sweep_config(pipeline: PipelineConfig) -> SweepConfig | None:
    if pipeline.sweep.enabled and pipeline.sweep.overrides:
        return pipeline.sweep

    legacy_overrides = pipeline.metadata.get("sweep_overrides")
    if legacy_overrides is None:
        return None

    if not isinstance(legacy_overrides, list):
        raise ValueError(
            "pipeline.metadata.sweep_overrides must be a list of override objects."
        )

    if not legacy_overrides:
        return None

    return SweepConfig.model_validate(
        {
            "enabled": True,
            "overrides": legacy_overrides,
            "metadata": pipeline.metadata.get("sweep_metadata", {}),
        }
    )


def _resolve_name_suffix(
    pipeline: PipelineConfig,
    sweep_config: SweepConfig,
    override: dict[str, Any],
    *,
    index: int,
) -> str:
    if isinstance(override.get("name_suffix"), str) and override["name_suffix"].strip():
        return override["name_suffix"]

    if sweep_config.name_suffix_template:
        try:
            return sweep_config.name_suffix_template.format(
                index=index,
                pipeline_name=pipeline.name,
                name=pipeline.name,
            )
        except KeyError as exc:
            raise ValueError(
                f"Invalid name_suffix_template for pipeline {pipeline.name!r}: {exc}"
            ) from exc

    return f"sweep_{index}"


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged
