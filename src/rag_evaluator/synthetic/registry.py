from __future__ import annotations

from rag_evaluator.config import LLMConfig, LLMProvider
from rag_evaluator.synthetic.errors import SyntheticGenerationError
from rag_evaluator.synthetic.providers.base import LLMProviderClient
from rag_evaluator.synthetic.providers.openai import OpenAIProviderClient
from rag_evaluator.synthetic.providers.openrouter import OpenRouterProviderClient


def build_synthetic_provider(config: LLMConfig) -> LLMProviderClient:
    """
    Build the provider client used by synthetic generation
    :param config:
    :return:
    """
    
    provider = config.provider
    
    if provider == LLMProvider.OPENROUTER:
        return OpenRouterProviderClient(config=config)
    
    if provider == LLMProvider.OPENAI:
        return OpenAIProviderClient(config=config)
    
    raise SyntheticGenerationError(
        f"Unsupported synthetic generation provider: {provider}"
    )

def validate_model_family(
        config: LLMConfig,
        *,
        allowed_prefixes: tuple[str, ...] | None = None,
) -> None:
    """
    Validate that the configured model belongs to one of the allowed families.
    :param config:
    :param allowed_prefixes:
    :return:
    """
    
    if not allowed_prefixes:
        return
    
    model_name = config.model
    if any(model_name.startswith(prefix) for prefix in allowed_prefixes):
        return
    
    supported = ", ".join(allowed_prefixes)
    raise SyntheticGenerationError(
        f"Unsupported model family provider: {model_name!r}. Expected one of: {supported}."
    )
