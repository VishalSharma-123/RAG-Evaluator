from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from rag_evaluator.schemas import EvalSample

ModelT = TypeVar("ModelT", bound=BaseModel)

def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file into dictionary
    """

    yaml_path = Path(path)

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        message = f"Expected a YAML object at {yaml_path}, got {type(data).__name__}"
        raise ValueError(message)

    return data

def load_experiment_config(path: str | Path) -> ExperimentConfig:  # noqa: F821
    """
    Load and validate an experiment YAML file
    """
    from rag_evaluator.config import ExperimentConfig

    data = load_yaml(path)
    data = resolve_env_vars(data)

    return ExperimentConfig.model_validate(data)

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file as a list of dictionaries
    """

    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON on line {line_number} in {jsonl_path}: {exc.msg}."
                raise ValueError(message) from exc

            if not isinstance(row, dict):
                message = f"Invalid JSON on line {line_number} in {jsonl_path}."
                raise ValueError(message)

            rows.append(row)

    return rows

def load_eval_samples_jsonl(path: str | Path) -> list[EvalSample]:
    """
    Load normalized EvalSample records from a JSONL file.
    """

    rows = load_jsonl(path)
    samples: list[EvalSample] = []

    for index, row in enumerate(rows, start=1):
        try:
            samples.append(EvalSample.model_validate(row))
        except ValidationError as exc:
            message = f"Invalid EvalSample at record {index} in {path}."
            raise ValueError(message) from exc

    return samples

def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    """
    Write pydantic models or dictionaries to a JSONL file.
    :param path:
    :param records:
    :return:
    """

    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            if isinstance(record, BaseModel):
                data = record.model_dump(mode="json")
            else:
                data = record

            file.write(json.dumps(data, ensure_ascii=False))
            file.write("\n")

def write_eval_samples_jsonl(path: str | Path, samples: Iterable[EvalSample]) -> None:
    """
    Write normalized EvalSample records to a JSONL file.
    """
    write_jsonl(path, samples)

def export_json_schema(model: type[ModelT], path: str | Path) -> None:
    """
    Export a Pydantic model JSON Schema to a disk.
    :param model:
    :param path:
    :return:
    """

    schema_path = Path(path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    schema = model.model_json_schema()

    with schema_path.open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2, ensure_ascii=False)
        file.write("\n")

def resolve_env_vars(value: Any) -> Any:
    """
    Resolve ${ENV_VAR} placeholder in nested config objects
    """

    if isinstance(value, dict):
        return {key: resolve_env_vars(item) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]

    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.getenv(env_var)

        if resolved is None:
            message = f"Environment variable {env_var} is not set."
            raise ValueError(message)

        return resolved

    return value
