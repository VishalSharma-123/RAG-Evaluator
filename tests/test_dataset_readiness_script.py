from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def test_readiness_dry_run_reports_missing_local_only_dataset(tmp_path: Path) -> None:
    module = _load_script_module()
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        """
version: 1
datasets:
  local_missing:
    display_name: Local Missing
    source: local_jsonl
    dataset_name: null
    dataset_config: null
    default_split: test
    url: https://example.invalid/local
    question_types: [factoid]
    domain: test
    normalizer: ragas
    local_normalized_path: missing/local_missing.jsonl
""",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        catalog=catalog_path,
        dataset=None,
        limit=10,
        output_dir=tmp_path / "runs",
        normalized_dir=tmp_path / "normalized",
        database_path=tmp_path / "results.duckdb",
        readiness_config=None,
        dry_run=True,
        continue_on_error=True,
        require_api_key=True,
        embedder_model="embedder",
        generator_model="generator",
        judge_model="judge",
        max_tokens=256,
        top_k=5,
        log_level="INFO",
    )

    exit_code = module.run_readiness(args)

    assert exit_code == 0
    manifest_text = (tmp_path / "runs" / "manifest.json").read_text(encoding="utf-8")
    assert '"dataset_ref": "local_missing"' in manifest_text
    assert '"status": "blocked_missing_local_file"' in manifest_text


def test_build_experiment_config_uses_bounded_local_dataset_and_smoke_pipeline(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    args = argparse.Namespace(
        output_dir=tmp_path / "runs",
        normalized_dir=tmp_path / "normalized",
        database_path=tmp_path / "results.duckdb",
        readiness_config=None,
        limit=10,
        embedder_model="test-embedder",
        generator_model="test-generator",
        judge_model="test-judge",
        max_tokens=128,
        top_k=3,
        log_level="INFO",
    )
    args = module.apply_readiness_config(args)

    config = module.build_experiment_config(
        dataset_ref="squad_v2",
        normalized_path=tmp_path / "squad_v2.jsonl",
        args=args,
    )

    assert config["datasets"][0]["source"] == "local_jsonl"
    assert config["datasets"][0]["sample_limit"] == 10
    assert config["pipelines"][0]["sweep"]["enabled"] is False
    assert config["pipelines"][0]["embedder"]["provider"] == "openai"
    assert config["pipelines"][0]["generator"]["max_tokens"] == 128
    assert config["pipelines"][0]["retriever"]["top_k"] == 3
    assert module.DEFAULT_SAMPLE_LIMIT == 10


def test_readiness_yaml_config_controls_openai_models_and_urls(tmp_path: Path) -> None:
    module = _load_script_module()
    config_path = tmp_path / "readiness.yaml"
    config_path.write_text(
        """
run:
  sample_limit: 10
  output_dir: custom/runs
  normalized_dir: custom/normalized
  database_path: custom/results.duckdb
pipeline:
  top_k: 7
  embedder:
    provider: openai
    model: custom-embed
    batch_size: 16
    api_key_env: CUSTOM_EMBED_KEY
    base_url: https://embed.example.test/v1
  generator:
    provider: openai
    model: custom-generate
    temperature: 0.1
    max_tokens: 333
    api_key_env: CUSTOM_GENERATE_KEY
    base_url: https://generate.example.test/v1
  judge:
    provider: openai
    model: custom-judge
    temperature: 0.0
    max_tokens: 111
    api_key_env: CUSTOM_JUDGE_KEY
    base_url: https://judge.example.test/v1
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        readiness_config=config_path,
        output_dir=module.DEFAULT_OUTPUT_DIR,
        normalized_dir=module.DEFAULT_NORMALIZED_DIR,
        database_path=module.DEFAULT_DATABASE_PATH,
        limit=module.DEFAULT_SAMPLE_LIMIT,
        embedder_model="ignored-embedder",
        generator_model="ignored-generator",
        judge_model="ignored-judge",
        max_tokens=256,
        top_k=5,
        log_level="INFO",
    )

    args = module.apply_readiness_config(args)
    config = module.build_experiment_config(
        dataset_ref="squad_v2",
        normalized_path=tmp_path / "squad_v2.jsonl",
        args=args,
    )
    pipeline = config["pipelines"][0]

    assert args.output_dir == Path("custom/runs")
    assert args.normalized_dir == Path("custom/normalized")
    assert args.database_path == Path("custom/results.duckdb")
    assert pipeline["retriever"]["top_k"] == 7
    assert pipeline["embedder"]["model"] == "custom-embed"
    assert pipeline["embedder"]["batch_size"] == 16
    assert pipeline["embedder"]["metadata"]["base_url"] == "https://embed.example.test/v1"
    assert pipeline["generator"]["model"] == "custom-generate"
    assert pipeline["generator"]["max_tokens"] == 333
    assert pipeline["generator"]["metadata"]["api_key_env"] == "CUSTOM_GENERATE_KEY"
    assert pipeline["generator"]["metadata"]["base_url"] == "https://generate.example.test/v1"
    assert pipeline["judge"]["model"] == "custom-judge"
    assert pipeline["judge"]["max_tokens"] == 111
    assert pipeline["judge"]["metadata"]["api_key_env"] == "CUSTOM_JUDGE_KEY"
    assert pipeline["judge"]["metadata"]["base_url"] == "https://judge.example.test/v1"


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "run_dataset_readiness_256.py"
    spec = importlib.util.spec_from_file_location("run_dataset_readiness_256", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to import {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
