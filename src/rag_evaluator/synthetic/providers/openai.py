from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from rag_evaluator.config import LLMConfig
from rag_evaluator.synthetic.errors import SyntheticProviderError
from rag_evaluator.synthetic.providers.base import LLMProviderClient
from rag_evaluator.synthetic.types import ProviderGenerationResult


@dataclass(frozen=True)
class OpenAIProviderClient(LLMProviderClient):
    """
    Provider client for OpenAI-hosted models.
    """

    config: LLMConfig

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderGenerationResult:
        request_metadata = metadata or {}
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            response = requests.post(
                self._base_url(),
                headers=self._headers(request_metadata),
                json=payload,
                timeout=self._timeout_seconds(),
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            raise SyntheticProviderError(
                f"OpenAI request failed with HTTP error: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise SyntheticProviderError(f"OpenAI request failed: {exc}") from exc

        try:
            response_payload = response.json()
        except json.JSONDecodeError as exc:
            raise SyntheticProviderError("OpenAI response was not valid JSON.") from exc

        message = self._extract_message(response_payload)
        content = self._extract_content(message)
        if not content.strip():
            raise SyntheticProviderError(
                "OpenAI response did not contain a non-empty message content."
            )

        usage = response_payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else {}

        return ProviderGenerationResult(
            content=content,
            raw_response=response_payload,
            usage=usage_payload,
            response_metadata={
                "provider": "openai",
                "model": self.config.model,
                "id": response_payload.get("id"),
            },
        )

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        extra_body = self.config.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        return payload

    def _extract_message(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SyntheticProviderError("OpenAI response did not contain any choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise SyntheticProviderError(
                "OpenAI response choice had an invalid shape."
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise SyntheticProviderError(
                "OpenAI response did not contain a message object."
            )

        return message

    def _extract_content(self, message: dict[str, Any]) -> str:
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            return "".join(text_parts)

        raise SyntheticProviderError(
            "OpenAI response content had an unsupported shape."
        )

    def _base_url(self) -> str:
        base_url = (
            self.config.metadata.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1/chat/completions"
        )
        return str(base_url)

    def _api_key(self) -> str:
        api_key = self.config.metadata.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SyntheticProviderError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or config.metadata['api_key']."
            )
        return str(api_key)

    def _timeout_seconds(self) -> float:
        raw_timeout = self.config.metadata.get("timeout_seconds", 60)
        try:
            return float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise SyntheticProviderError(
                f"Invalid timeout_seconds value: {raw_timeout!r}."
            ) from exc

    def _headers(self, metadata: dict[str, Any]) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

        organization = (
            metadata.get("openai_organization")
            or self.config.metadata.get("openai_organization")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        project = (
            metadata.get("openai_project")
            or self.config.metadata.get("openai_project")
            or os.getenv("OPENAI_PROJECT")
        )

        if organization:
            headers["OpenAI-Organization"] = str(organization)
        if project:
            headers["OpenAI-Project"] = str(project)

        return headers
