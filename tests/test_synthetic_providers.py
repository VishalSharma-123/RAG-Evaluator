from __future__ import annotations

from typing import Any

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.providers.openai import OpenAIProviderClient
from rag_evaluator.synthetic.providers.openrouter import OpenRouterProviderClient


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.text = "dummy"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_openai_provider_client_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            {
                "id": "resp-1",
                "choices": [
                    {
                        "message": {
                            "content": '{"samples": []}',
                        }
                    }
                ],
                "usage": {"total_tokens": 10},
            }
        )

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openai.requests.post", fake_post)

    client = OpenAIProviderClient(
        config=LLMConfig.model_validate(
            {
                "provider": "openai",
                "model": "gpt-test",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    result = client.generate_json(system_prompt="sys", user_prompt="user")

    assert result.content == '{"samples": []}'
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_openrouter_provider_client_includes_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            {
                "id": "resp-2",
                "choices": [
                    {
                        "message": {
                            "content": '{"samples": []}',
                            "reasoning_details": {"steps": 1},
                        }
                    }
                ],
                "usage": {"total_tokens": 11},
            }
        )

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openrouter.requests.post", fake_post)

    client = OpenRouterProviderClient(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-test",
                "metadata": {"api_key": "secret", "reasoning_enabled": True},
            }
        )
    )

    result = client.generate_json(system_prompt="sys", user_prompt="user")

    assert result.content == '{"samples": []}'
    assert result.reasoning_details == {"steps": 1}
    assert captured["json"]["reasoning"] == {"enabled": True}


def test_openai_provider_client_rejects_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        return DummyResponse({"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openai.requests.post", fake_post)

    client = OpenAIProviderClient(
        config=LLMConfig.model_validate(
            {
                "provider": "openai",
                "model": "gpt-test",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    with pytest.raises(SyntheticProviderError, match="non-empty message content"):
        client.generate_json(system_prompt="sys", user_prompt="user")


def test_openrouter_provider_client_continues_when_only_reasoning_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        DummyResponse(
            {
                "id": "resp-1",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_details": [{"type": "reasoning.text", "text": "We"}],
                        }
                    }
                ],
            }
        ),
        DummyResponse(
            {
                "id": "resp-2",
                "choices": [
                    {
                        "message": {
                            "content": '{"samples": []}',
                            "reasoning_details": [{"type": "reasoning.text", "text": "We"}],
                        }
                    }
                ],
                "usage": {"total_tokens": 20},
            }
        ),
    ]
    captured_payloads: list[dict[str, Any]] = []

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> DummyResponse:
        captured_payloads.append(json)
        return responses.pop(0)

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openrouter.requests.post", fake_post)

    client = OpenRouterProviderClient(
        config=LLMConfig.model_validate(
            {
                "provider": "openrouter",
                "model": "nvidia/nemotron-test",
                "metadata": {"api_key": "secret"},
            }
        )
    )

    result = client.generate_json(system_prompt="sys", user_prompt="user")

    assert result.content == '{"samples": []}'
    assert len(captured_payloads) == 2
    assert captured_payloads[1]["messages"][-1]["content"].startswith("Continue from your prior reasoning")
