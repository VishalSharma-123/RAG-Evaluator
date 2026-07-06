from __future__ import annotations

import json
from typing import Any

from rag_evaluator.schemas import GenerationMetrics
from rag_evaluator.scoring.judges.base import JudgeScoringError


def parse_judge_response(raw_response: str) -> GenerationMetrics:
    """
    Parse a judge response into GenerationMetrics
    :param raw_response:
    :return:
    """
    payload = _load_json_object(raw_response)

    faithfulness = _load_metric(payload, "faithfulness")
    relevance = _load_metric(payload, "relevance")
    hallucination = _load_metric(payload, "hallucination")
    bert_score = _load_optional_metric(payload, "bert_score")

    return GenerationMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        hallucination=hallucination,
        bert_score=bert_score,
    )

def _load_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if not text:
        raise JudgeScoringError("Judge response is empty.")
    
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeScoringError("Judge response is not valid JSON.") from exc
    
    if not isinstance(payload, dict):
        raise JudgeScoringError("Judge response must be a valid JSON object.")

    return payload

def _load_metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise JudgeScoringError(f"Judge response is missing `{key}`.")

    metric = _coerce_float(value, key)
    _validate_unit_interval(metric, key)
    return metric

def _load_optional_metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None

    metric = _coerce_float(value, key)
    _validate_unit_interval(metric, key)
    return metric

def _coerce_float(value: Any, key: str) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError) as exc:
        raise JudgeScoringError(f"Judge response field `{key}` must be numeric.") from exc

    if metric != metric or metric in (float("inf"), float("-inf")):
        raise JudgeScoringError(f"Judge response field `{key}` must be finite.")

    return metric

def _validate_unit_interval(value: float, key: str) -> None:
    if value < 0.0 or value > 1.0:
        raise JudgeScoringError(f"Judge response field `{key}` must be between 0 and 1.")
