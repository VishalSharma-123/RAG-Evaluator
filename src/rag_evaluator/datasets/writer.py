from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from rag_evaluator.io import write_eval_samples_jsonl
from rag_evaluator.schemas import EvalSample


def write_normalized_dataset(
    samples: Iterable[EvalSample],
    *,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    """
    Write normalized EvalSample records and an optional manifest.
    :param samples:
    :param output_path:
    :param manifest_path:
    :param manifest:
    :return:
    """

    output = Path(output_path)
    sample_list = list(samples)
    write_eval_samples_jsonl(output, sample_list)

    if manifest_path is not None:
        write_dataset_manifest(
            manifest_path,
            {
                "normalized_path": str(output),
                "sample_count": len(sample_list),
                "sha256": sha256_file(output),
                **(manifest or {}),
            },
        )


def write_dataset_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """
    Write dataset normalization metadata as YAML.
    :param path:
    :param manifest:
    :return:
    """

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=True)


def sha256_file(path: str | Path) -> str:
    """
    Compute a SHA-256 checksum for a file.
    :param path:
    :return:
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()
