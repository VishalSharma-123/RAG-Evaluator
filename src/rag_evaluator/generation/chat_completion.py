from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rag_evaluator.config import LLMConfig
from rag_evaluator.generation.base import Generator
from rag_evaluator.generation.parsing import parse_generation_response
from rag_evaluator.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.registry import build_synthetic_provider


@dataclass(frozen=True)
class ChatCompletionGenerator(Generator):
    """
    Provider-agnostic chat-completions generator.

    This uses the shared synthetic provider registry so OpenRouter and OpenAI
    both work through the same execution path.
    """

    config: LLMConfig

    def generate(
        self,
        sample: EvalSample,
        context_chunks: list[Chunk],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        request_metadata = metadata or {}
        provider = build_synthetic_provider(self.config)
        start_time = time.perf_counter()

        try:
            provider_result = provider.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_generation_prompt(sample, context_chunks),
                metadata=request_metadata,
            )
        except SyntheticProviderError as exc:
            raise RuntimeError(f"Chat completion request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        usage = self._usage_dict(provider_result.usage)
        prompt_tokens = self._int_usage_value(usage, "prompt_tokens")
        completion_tokens = self._int_usage_value(usage, "completion_tokens")
        total_tokens = self._int_usage_value(usage, "total_tokens")
        cost_usd, pricing_metadata = self._calculate_cost_usd(
            usage=usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_metadata=request_metadata,
        )

        response_metadata = self._response_metadata(provider_result.response_metadata)
        answer_metadata: dict[str, Any] = {
            **request_metadata,
            "provider": self.config.provider.value,
            "model": self.config.model,
            "usage": usage,
            "pricing": pricing_metadata,
        }
        if response_metadata:
            answer_metadata["response_metadata"] = response_metadata
            response_id = response_metadata.get("id")
            if response_id is not None:
                answer_metadata["response_id"] = response_id
        if provider_result.reasoning_details is not None:
            answer_metadata["reasoning_details"] = provider_result.reasoning_details
        if provider_result.raw_response is not None:
            answer_metadata["raw_response"] = provider_result.raw_response

        return parse_generation_response(
            provider_result.content,
            sample=sample,
            model_name=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata=answer_metadata,
        )

    def _calculate_cost_usd(
        self,
        *,
        usage: dict[str, Any],
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        request_metadata: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        direct_cost = self._coerce_cost(
            usage.get("cost_usd")
            or usage.get("total_cost")
            or usage.get("cost")
        )
        if direct_cost is not None:
            return direct_cost, {
                "source": "provider_usage",
                "mode": "direct",
            }

        combined_metadata = {
            **self.config.metadata,
            **request_metadata,
        }

        prompt_rate = self._coerce_cost(
            combined_metadata.get("prompt_cost_per_1k_tokens_usd")
        )
        completion_rate = self._coerce_cost(
            combined_metadata.get("completion_cost_per_1k_tokens_usd")
        )
        unified_rate = self._coerce_cost(
            combined_metadata.get("cost_per_1k_tokens_usd")
            or combined_metadata.get("price_per_1k_tokens_usd")
        )

        if unified_rate is not None:
            if prompt_rate is None:
                prompt_rate = unified_rate
            if completion_rate is None:
                completion_rate = unified_rate

        if prompt_rate is not None or completion_rate is not None:
            resolved_prompt_tokens = prompt_tokens or 0
            resolved_completion_tokens = completion_tokens or 0
            if (
                resolved_prompt_tokens == 0
                and resolved_completion_tokens == 0
                and total_tokens is not None
                and unified_rate is not None
            ):
                cost_usd = (total_tokens / 1000.0) * unified_rate
            else:
                cost_usd = (
                    (resolved_prompt_tokens / 1000.0) * (prompt_rate or 0.0)
                    + (resolved_completion_tokens / 1000.0) * (completion_rate or 0.0)
                )

            return round(cost_usd, 12), {
                "source": "metadata_rates",
                "prompt_rate_per_1k": prompt_rate,
                "completion_rate_per_1k": completion_rate,
                "unified_rate_per_1k": unified_rate,
            }

        fixed_cost = self._coerce_cost(combined_metadata.get("fixed_cost_usd"))
        if fixed_cost is None:
            fixed_cost = self._coerce_cost(self.config.cost_usd)
        if fixed_cost is not None and fixed_cost > 0:
            return fixed_cost, {
                "source": "fixed_cost",
            }

        return 0.0, {
            "source": "unavailable",
        }

    def _usage_dict(self, usage: dict[str, Any] | None) -> dict[str, Any]:
        return usage if isinstance(usage, dict) else {}

    def _response_metadata(self, response_metadata: dict[str, Any] | None) -> dict[str, Any]:
        return response_metadata if isinstance(response_metadata, dict) else {}

    def _coerce_cost(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int_usage_value(self, usage: dict[str, Any], key: str) -> int | None:
        value = usage.get(key)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None
