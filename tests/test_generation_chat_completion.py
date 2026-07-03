from __future__ import annotations

from typing import Any

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.generation.chat_completion import ChatCompletionGenerator
from rag_evaluator.generation.nemotron import NemotronGenerator
from rag_evaluator.generation.parsing import GenerationOutputError
from rag_evaluator.synthetic.types import ProviderGenerationResult


class FakeProvider:
    def __init__(self, result: ProviderGenerationResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderGenerationResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": metadata,
            }
        )
        return self._result


def test_chat_completion_generator_parses_response_and_records_telemetry(
    make_sample,
    make_chunk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    perf_values = iter([200.0, 200.25])

    def fake_perf_counter() -> float:
        return next(perf_values)

    fake_provider = FakeProvider(
        ProviderGenerationResult(
            content='{"answer": "Paris"}',
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            response_metadata={"provider": "openrouter", "id": "chatcmpl-1"},
        )
    )

    def fake_build_provider(config) -> FakeProvider:
        captured["config"] = config
        return fake_provider

    monkeypatch.setattr(
        "rag_evaluator.generation.chat_completion.time.perf_counter",
        fake_perf_counter,
    )
    monkeypatch.setattr(
        "rag_evaluator.generation.chat_completion.build_synthetic_provider",
        fake_build_provider,
    )

    generator = ChatCompletionGenerator(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "meta/llama-3.1-8b-instruct:free",
                "temperature": 0.0,
                "max_tokens": 256,
                "metadata": {
                    "prompt_cost_per_1k_tokens_usd": 2.0,
                    "completion_cost_per_1k_tokens_usd": 4.0,
                },
            }
        )
    )

    answer = generator.generate(
        make_sample(question="What is the capital of France?"),
        [make_chunk(text="Paris is the capital of France.")],
        metadata={"pipeline_name": "unit"},
    )

    assert captured["config"].model == "meta/llama-3.1-8b-instruct:free"
    assert len(fake_provider.calls) == 1
    assert answer.answer == "Paris"
    assert answer.model_name == "meta/llama-3.1-8b-instruct:free"
    assert answer.prompt_tokens == 100
    assert answer.completion_tokens == 50
    assert answer.latency_ms == 250
    assert answer.cost_usd == pytest.approx(0.4)
    assert answer.metadata["provider"] == "openrouter"
    assert answer.metadata["response_id"] == "chatcmpl-1"
    assert answer.metadata["pricing"]["source"] == "metadata_rates"


def test_chat_completion_generator_requires_api_key(
    make_sample,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHAT_COMPLETION_TEST_API_KEY", raising=False)

    generator = ChatCompletionGenerator(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "meta/llama-3.1-8b-instruct:free",
                "metadata": {
                    "api_key_env": "CHAT_COMPLETION_TEST_API_KEY",
                },
            }
        )
    )

    with pytest.raises(RuntimeError, match="Missing OpenRouter API Key"):
        generator.generate(
            make_sample(),
            context_chunks=[],
        )


def test_chat_completion_generator_rejects_malformed_response(
    make_sample,
    make_chunk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = FakeProvider(
        ProviderGenerationResult(
            content="Paris",
            response_metadata={"provider": "openrouter", "id": "chatcmpl-2"},
        )
    )
    monkeypatch.setattr(
        "rag_evaluator.generation.chat_completion.build_synthetic_provider",
        lambda config: fake_provider,
    )

    generator = ChatCompletionGenerator(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "meta/llama-3.1-8b-instruct:free",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    with pytest.raises(GenerationOutputError, match="valid JSON"):
        generator.generate(
            make_sample(),
            [make_chunk()],
        )


def test_nemotron_generator_remains_a_compatibility_alias(
    make_sample,
    make_chunk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = FakeProvider(
        ProviderGenerationResult(
            content='{"answer": "Paris"}',
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )
    monkeypatch.setattr(
        "rag_evaluator.generation.chat_completion.build_synthetic_provider",
        lambda config: fake_provider,
    )

    generator = NemotronGenerator(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "meta/llama-3.1-8b-instruct:free",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    answer = generator.generate(make_sample(), [make_chunk()])

    assert answer.answer == "Paris"
    assert answer.prompt_tokens == 10
