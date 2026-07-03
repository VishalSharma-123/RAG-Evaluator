from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from rag_evaluator.datasets.config import DatasetConfig


class ChunkerType(StrEnum):
    """
    Supported chunking strategies.
    """
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    LATE = "late"

class EmbedderProvider(StrEnum):
    """
    Supported embedding providers.
    """
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    BGE = "bge"
    COHERE = "cohere"

class VectorStoreProvider(StrEnum):
    """
    Supported vector storage.
    """
    MEMORY = "memory"
    CHROMA = "chroma"

class RetrieverType(StrEnum):
    """
    Supported retrieval strategies.
    """
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"

class RerankerType(StrEnum):
    """
    Supported reranking strategies.
    """
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    OPENROUTER = "openrouter"

class LLMProvider(StrEnum):
    """
    Supported LLM API providers
    """
    OPENROUTER = "openrouter"
    OPENAI = "openai"

class ApiFormat(StrEnum):
    """
    Supported LLM API wire formats.
    """
    OPENAI_COMPATIBLE = "openai_compatible"

class ChunkerConfig(BaseModel):
    """
    Configuration for one chunking strategy
    """
    model_config = ConfigDict(extra="forbid")

    type: ChunkerType
    chunk_size: PositiveInt | None = None
    chunk_overlap: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class EmbedderConfig(BaseModel):
    """
    Configuration for one embedding model.
    """
    model_config = ConfigDict(extra="forbid")

    provider: EmbedderProvider
    model: str | None = None
    batch_size: PositiveInt = 32
    metadata: dict[str, Any] = Field(default_factory=dict)

class VectorStoreConfig(BaseModel):
    """
    Configuration for one vector store.
    """
    model_config = ConfigDict(extra="forbid")
    
    provider: VectorStoreProvider = VectorStoreProvider.MEMORY
    collection_name: str | None = None
    persist_directory: str = "storage/chroma"
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrieverConfig(BaseModel):
    """
    Configuration for one retriever model.
    """
    model_config = ConfigDict(extra="forbid")

    type: RetrieverType
    top_k: PositiveInt = 10
    metadata: dict[str, Any] = Field(default_factory=dict)

class RerankerConfig(BaseModel):
    """
    Configuration for one reranker model.
    """
    model_config = ConfigDict(extra="forbid")

    type: RerankerType = RerankerType.NONE
    model: str | None = None
    top_k: PositiveInt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SweepConfig(BaseModel):
    """
    Explicit sweep controls for pipeline variant expansion.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overrides: list[dict[str, Any]] = Field(default_factory=list)
    name_suffix_template: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """
    Configuration for an OpenRouter-hosted LLM.
    """
    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = LLMProvider.OPENROUTER
    api_format: ApiFormat = ApiFormat.OPENAI_COMPATIBLE
    model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    temperature: NonNegativeFloat = 0.0
    max_tokens: PositiveInt = 1024
    cost_usd: NonNegativeFloat = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

class SyntheticGenerationConfig(BaseModel):
    """
    Configuration for generating synthetic EvalSample datasets from chunks.
    """
    model_config = ConfigDict(extra="forbid")

    pipeline: str
    chunks_path: str
    output_path: str
    question_types: list[str] = Field(default_factory=list)
    max_samples: PositiveInt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSettingsConfig(BaseModel):
    """
    Top-level run settings for an experiment execution.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "runs"
    database_path: str | None = None
    run_name: str | None = None
    overwrite: bool = False
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """
    One Complete RAG pipeline configuration.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    chunker: ChunkerConfig
    embedder: EmbedderConfig
    store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    generator: LLMConfig = Field(default_factory=LLMConfig)
    judge: LLMConfig = Field(default_factory=LLMConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

class ExperimentConfig(BaseModel):
    """
    Top-level experiment configuration parsed from experiment.yaml
    """
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    datasets: list[DatasetConfig]
    pipelines: list[PipelineConfig]
    synthetic_generation: SyntheticGenerationConfig | None = None
    output_dir: str = "runs"
    run_settings: RunSettingsConfig = Field(default_factory=RunSettingsConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_run_settings(self) -> ExperimentConfig:
        if "run_settings" not in self.model_fields_set:
            self.run_settings = self.run_settings.model_copy(
                update={
                    "output_dir": self.output_dir,
                }
            )
        elif "output_dir" not in self.model_fields_set:
            self.output_dir = self.run_settings.output_dir
        else:
            self.output_dir = self.run_settings.output_dir

        return self
