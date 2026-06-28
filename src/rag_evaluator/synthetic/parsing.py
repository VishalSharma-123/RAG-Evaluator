from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from rag_evaluator.schemas import EvalSample
from rag_evaluator.synthetic.errors import SyntheticParsingError


def parse_synthetic_generation_response(raq_response: str) -> list[EvalSample]:
    """
    Parse a strict JSON synthetic generation response into EvalSample records.
    :param raq_response:
    :return:
    """
    payload = _load_json_object(raq_response)
    raw_samples = payload.get('samples')
    
    if not isinstance(raw_samples, list):
        raise SyntheticParsingError(
            "Synthetic generation response must contain a `samples` list."
        )
    
    samples: list[EvalSample] = []
    
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise SyntheticParsingError(
                f"Synthetic sample at index {index} must be a JSON object."
            )
        
        sample_payload = _normalize_sample_payload(raw_sample)
        
        try:
            sample = EvalSample.model_validate(sample_payload)
        except ValidationError as exc:
            raise SyntheticParsingError(
                f"Invalid synthetic sample at index {index}: {exc}"
            ) from exc
        
        samples.append(sample)
    
    return samples

def _normalize_sample_payload(raw_sample: dict[str, Any]) -> dict[str, Any]:
    sample_payload = dict(raw_sample)

    sample_payload.setdefault("answer_aliases", [])
    sample_payload.setdefault("choices", [])
    sample_payload.setdefault("source_dataset", "synthetic")
    sample_payload.setdefault("source_split", "generated")
    sample_payload.setdefault("source_id", sample_payload.get("sample_id"))
    sample_payload.setdefault("evidence_spans", [])
    sample_payload.setdefault("metadata", {})

    return sample_payload

def _load_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if not text:
        raise SyntheticParsingError("Synthetic generator response is empty.")
    
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SyntheticParsingError(
            "Synthetic generator response is not valid JSON."
        ) from exc
    
    if not isinstance(payload, dict):
        raise SyntheticParsingError(
            "Synthetic generator response must be a JSON object."
        )
    
    return payload