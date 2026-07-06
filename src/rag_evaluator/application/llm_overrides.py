from __future__ import annotations

from rag_evaluator.config import EmbedderProvider, ExperimentConfig, LLMProvider


def apply_openai_base_url(
    experiment: ExperimentConfig,
    openai_base_url: str | None,
) -> ExperimentConfig:
    """
    Inject an OpenAI-compatible base URL into OpenAI-backed config blocks.
    """

    if not openai_base_url:
        return experiment

    updated_pipelines = []
    for pipeline in experiment.pipelines:
        pipeline_updates: dict[str, object] = {}

        if pipeline.embedder.provider == EmbedderProvider.OPENAI:
            pipeline_updates["embedder"] = pipeline.embedder.model_copy(
                update={
                    "metadata": {
                        **pipeline.embedder.metadata,
                        "base_url": openai_base_url,
                    }
                }
            )

        if pipeline.generator.provider == LLMProvider.OPENAI:
            pipeline_updates["generator"] = pipeline.generator.model_copy(
                update={
                    "metadata": {
                        **pipeline.generator.metadata,
                        "base_url": openai_base_url,
                    }
                }
            )

        if pipeline.judge.provider == LLMProvider.OPENAI:
            pipeline_updates["judge"] = pipeline.judge.model_copy(
                update={
                    "metadata": {
                        **pipeline.judge.metadata,
                        "base_url": openai_base_url,
                    }
                }
            )

        if pipeline_updates:
            updated_pipelines.append(pipeline.model_copy(update=pipeline_updates))
        else:
            updated_pipelines.append(pipeline)

    return experiment.model_copy(update={"pipelines": updated_pipelines})
