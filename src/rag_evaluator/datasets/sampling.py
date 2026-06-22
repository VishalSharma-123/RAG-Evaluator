from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rag_evaluator.schemas import EvalSample


def stratified_sample(
        samples: Iterable[EvalSample],
        *,
        per_group: int,
        keys: tuple[str, ...] = ("source_dataset", "question_type"),
        seed: int = 42,
) -> list[EvalSample]:
    """
    Sample up to N EvalSample records per stratum.
    """
    
    if per_group < 1:
        raise ValueError(f"per_group {per_group} must be >= 1")

    grouped: dict[tuple[Any, ...], list[EvalSample]] = defaultdict(list)

    for sample in samples:
        group_key = tuple(_get_sample_value(sample, key) for key in keys)
        grouped[group_key].append(sample)

    rng = random.Random(seed)
    selected: list[EvalSample] = []

    for group_key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        bucket = grouped[group_key]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_group])

    return selected

def _get_sample_value(sample: EvalSample, key: str) -> Any:
    """
    Read a stratum value from an EvalSample field or metadata.
    """
    
    if hasattr(sample, key):
        return getattr(sample, key)
    
    if key in sample.metadata:
        return sample.metadata[key]
    
    raise ValueError(f"Unknown stratification key: {key}")
