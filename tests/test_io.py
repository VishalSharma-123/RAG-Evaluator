from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_evaluator.io import (
    load_eval_samples_jsonl,
    load_jsonl,
    load_yaml,
    resolve_env_vars,
    write_jsonl,
)


def test_load_yaml_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a YAML object"):
        load_yaml(path)


def test_write_and_load_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"value": 1}, {"value": 2}])

    assert load_jsonl(path) == [{"value": 1}, {"value": 2}]


def test_load_eval_samples_jsonl_validates_records(tmp_path: Path, make_sample) -> None:
    path = tmp_path / "samples.jsonl"
    sample = make_sample().model_dump(mode="json")
    invalid = {"sample_id": "bad"}
    path.write_text(
        json.dumps(sample) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid EvalSample"):
        load_eval_samples_jsonl(path)


def test_resolve_env_vars_replaces_nested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIT_TOKEN", "secret")

    resolved = resolve_env_vars(
        {
            "token": "${UNIT_TOKEN}",
            "nested": ["${UNIT_TOKEN}", {"inner": "${UNIT_TOKEN}"}],
        }
    )

    assert resolved == {
        "token": "secret",
        "nested": ["secret", {"inner": "secret"}],
    }
