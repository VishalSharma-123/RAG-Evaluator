from __future__ import annotations

from pathlib import Path

from rag_evaluator.cli import build_parser, generate_synthetic
from rag_evaluator.io import load_eval_samples_jsonl, write_jsonl
from rag_evaluator.schemas import QuestionType


def test_build_parser_supports_generate_synthetic() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "generate-synthetic",
            "--chunks",
            "chunks.jsonl",
            "--output",
            "synthetic.jsonl",
            "--question-type",
            "factoid",
            "--question-type",
            "unanswerable",
        ]
    )

    assert args.command == "generate-synthetic"
    assert args.provider == "openrouter"
    assert args.question_types == ["factoid", "unanswerable"]


def test_generate_synthetic_writes_samples(
    tmp_path: Path,
    make_chunk,
    make_sample,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "synthetic.jsonl"
    write_jsonl(chunks_path, [make_chunk()])

    captured: dict[str, object] = {}

    class FakeNemotronSyntheticGenerator:
        def __init__(self, *, config) -> None:
            captured["config"] = config

        def generate_samples(
            self,
            chunks,
            *,
            question_types=None,
            max_samples=None,
            metadata=None,
        ):
            captured["chunks"] = chunks
            captured["question_types"] = question_types
            captured["max_samples"] = max_samples
            captured["metadata"] = metadata
            return [make_sample(source_dataset="synthetic", source_split="generated")]

    monkeypatch.setattr(
        "rag_evaluator.cli.NemotronSyntheticGenerator",
        FakeNemotronSyntheticGenerator,
    )

    exit_code = generate_synthetic(
        chunks_path=chunks_path,
        output_path=output_path,
        provider="openrouter",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        question_types=["factoid"],
        max_samples=2,
        temperature=0.0,
        max_tokens=512,
        reasoning_enabled=True,
    )

    assert exit_code == 0
    assert output_path.exists()
    assert captured["question_types"] == [QuestionType.FACTOID]
    assert captured["max_samples"] == 2
    assert captured["metadata"] == {
        "generator_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "generator_provider": "openrouter",
    }

    samples = load_eval_samples_jsonl(output_path)
    assert len(samples) == 1
    assert samples[0].source_dataset == "synthetic"


def test_generate_synthetic_uses_experiment_config(
    tmp_path: Path,
    make_sample,
    monkeypatch,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "synthetic.jsonl"

    chunks_path.write_text(
        '{"chunk_id":"doc:chunk:0","document_id":"doc","text":"ctx"}\n',
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: synthetic_unit",
                "output_dir: runs",
                "datasets:",
                "  - name: tiny",
                "    source: local_jsonl",
                "    path: examples/eval_samples.jsonl",
                "    split: test",
                "pipelines:",
                "  - name: pipeline-1",
                "    chunker:",
                "      type: fixed",
                "      chunk_size: 128",
                "    embedder:",
                "      provider: bge",
                "      model: BAAI/bge-small-en-v1.5",
                "    retriever:",
                "      type: vector",
                "      top_k: 3",
                "    generator:",
                "      provider: openrouter",
                "      model: nvidia/nemotron-3-super-120b-a12b:free",
                "      temperature: 0.0",
                "      max_tokens: 600",
                "      metadata:",
                "        reasoning_enabled: true",
                "synthetic_generation:",
                "  pipeline: pipeline-1",
                f"  chunks_path: {chunks_path}",
                f"  output_path: {output_path}",
                "  question_types:",
                "    - factoid",
                "  max_samples: 3",
                "  metadata:",
                "    source: yaml",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeNemotronSyntheticGenerator:
        def __init__(self, *, config) -> None:
            captured["config"] = config

        def generate_samples(
            self,
            chunks,
            *,
            question_types=None,
            max_samples=None,
            metadata=None,
        ):
            captured["question_types"] = question_types
            captured["max_samples"] = max_samples
            captured["metadata"] = metadata
            return [make_sample(source_dataset="synthetic", source_split="generated")]

    monkeypatch.setattr(
        "rag_evaluator.cli.NemotronSyntheticGenerator",
        FakeNemotronSyntheticGenerator,
    )

    exit_code = generate_synthetic(
        config_path=config_path,
        chunks_path=None,
        output_path=None,
        provider="openrouter",
        model="ignored",
        question_types=None,
        max_samples=None,
        temperature=0.0,
        max_tokens=512,
        reasoning_enabled=False,
    )

    assert exit_code == 0
    assert output_path.exists()
    assert captured["question_types"] == [QuestionType.FACTOID]
    assert captured["max_samples"] == 3
    assert captured["metadata"] == {
        "source": "yaml",
        "pipeline": "pipeline-1",
        "generator_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "generator_provider": "openrouter",
    }
