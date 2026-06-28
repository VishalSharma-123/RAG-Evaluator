from __future__ import annotations

import json
from typing import Any

from rag_evaluator.schemas import EvalSample, GeneratedAnswer


class GenerationOutputError(ValueError):
    """
    Raised when a generator response cannot be parsed into the expected shape.
    """
    
def parse_generation_response(
        raw_response: str,
        *,
        sample: EvalSample,
        model_name: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
) -> GeneratedAnswer:
    """
    Parse a raw model resposne into a GeneratedAnswer
    :param raw_response:
    :param sample:
    :param model_name:
    :param prompt_tokens:
    :param completion_tokens:
    :param latency_ms:
    :param cost_usd:
    :param metadata:
    :return:
    """
    payload = _load_json_object(raw_response)
    answer = payload.get("answer")
    
    if not isinstance(answer, str):
        raise GenerationOutputError("Generator response must contain a string `answer` field.")
    
    normalized_answer = answer.strip()
    if not normalized_answer:
        raise GenerationOutputError("Generator response returned empty `answer` field.")
    
    return GeneratedAnswer(
        sample_id=sample.sample_id,
        answer=normalized_answer,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        metadata=metadata or {},
    )

def _load_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if not text:
        raise GenerationOutputError("Generator response is empty.")
    
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationOutputError("Generator response is not valid JSON.") from exc
    
    if not isinstance(payload, dict):
        raise GenerationOutputError("Generator response must be a JSON object.")
    
    return payload
