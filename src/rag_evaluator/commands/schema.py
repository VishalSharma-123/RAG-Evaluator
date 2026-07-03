from __future__ import annotations

from pydantic import BaseModel

from rag_evaluator.config import ExperimentConfig
from rag_evaluator.schemas import (
    Chunk,
    EvalResult,
    EvalSample,
    GeneratedAnswer,
    RetrievedChunk,
)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "eval-sample": EvalSample,
    "chunk": Chunk,
    "retrieved-chunk": RetrievedChunk,
    "generated-answer": GeneratedAnswer,
    "eval-result": EvalResult,
    "experiment-config": ExperimentConfig,
}

__all__ = ["SCHEMA_MODELS"]
