from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from rag_evaluator.schemas import QuestionType


class DatasetSource(StrEnum):
    """
    Supported dataset source types.
    """

    LOCAL_JSONL = "local_jsonl"
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    RAGAS = "ragas"


class DatasetConfig(BaseModel):
    """
    Resolved dataset configuration used by adapters.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: DatasetSource = DatasetSource.LOCAL_JSONL
    path: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    split: str = "validation"
    question_type: QuestionType | None = None
    domain: str | None = None
    sample_limit: PositiveInt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CatalogDataset(BaseModel):
    """
    One dataset entry from configs/datasets.yaml.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str
    source: DatasetSource
    dataset_name: str | None = None
    dataset_config: str | None = None
    default_split: str
    url: str
    question_types: list[QuestionType] = Field(default_factory=list)
    domain: str
    normalizer: str
    local_normalized_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCatalog(BaseModel):
    """
    Dataset catalog loaded from YAML.
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    description: str | None = None
    datasets: dict[str, CatalogDataset]
