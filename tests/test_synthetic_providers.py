from __future__ import annotations

from typing import Any

import pytest

from rag_evaluator.config import LLMConfig
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.providers.openai import OpenAIProviderClient
from rag_evaluator.synthetic.providers.openrouter import OpenRouterProviderClient


class DummyChoiceMessage:
    def __init__(self, content: Any, reasoning_details: Any | None = None) -> None:
        self.content = content
        self.reasoning_details = reasoning_details

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content}
        if self.reasoning_details is not None:
            payload["reasoning_details"] = self.reasoning_details
        return payload


class DummyChoice:
    def __init__(self, message: DummyChoiceMessage | None) -> None:
        self.message = message


class DummyUsage:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.id = payload.get("id")
        self.usage = DummyUsage(payload.get("usage", {})) if isinstance(payload.get("usage"), dict) else None
        choices = payload.get("choices", [])
        self.choices = [
            DummyChoice(
                DummyChoiceMessage(
                    choice.get("message", {}).get("content"),
                    choice.get("message", {}).get("reasoning_details"),
                )
            )
            for choice in choices
            if isinstance(choice, dict)
        ]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class DummyChatCompletions:
    def __init__(self, response: DummyResponse, captured: dict[str, Any]) -> None:
        self._response = response
        self._captured = captured

    def create(self, **kwargs: Any) -> DummyResponse:
        self._captured["create_kwargs"] = kwargs
        return self._response


class DummyClient:
    def __init__(self, *, api_key: str, organization: str | None, project: str | None, base_url: str | None, timeout: float, max_retries: int, default_headers: dict[str, str]) -> None:
        self.kwargs = {
            "api_key": api_key,
            "organization": organization,
            "project": project,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
        }
        self.chat = type("Chat", (), {})()
        self.chat.completions = None  # set later


def test_openai_provider_client_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    response = DummyResponse(
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

    def fake_openai(**kwargs: Any) -> DummyClient:
        captured["client_kwargs"] = kwargs
        client = DummyClient(**kwargs)
        client.chat.completions = DummyChatCompletions(response, captured)
        return client

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openai.OpenAI", fake_openai)

    client = OpenAIProviderClient(
        config=LLMConfig.model_validate(
            {
                "provider": "openai",
                "model": "gpt-test",
                "metadata": {"api_key": "secret", "base_url": "https://example.invalid/v1"},
            }
        )
    )

    result = client.generate_json(system_prompt="sys", user_prompt="user")

    assert result.content == '{"samples": []}'
    assert captured["client_kwargs"]["api_key"] == "secret"
    assert captured["client_kwargs"]["base_url"] == "https://example.invalid/v1"
    assert captured["create_kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["create_kwargs"]["messages"][0]["role"] == "system"


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
    response = DummyResponse({"choices": [{"message": {"content": ""}}]})

    def fake_openai(**kwargs: Any) -> DummyClient:
        client = DummyClient(**kwargs)
        client.chat.completions = DummyChatCompletions(response, {})
        return client

    monkeypatch.setattr("rag_evaluator.synthetic.providers.openai.OpenAI", fake_openai)

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
