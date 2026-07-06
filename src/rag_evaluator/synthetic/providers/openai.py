from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import APIError, APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

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
        client = self._client(request_metadata)

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
                extra_body=self._extra_body(),
                timeout=self._timeout_seconds(),
            )
        except (AuthenticationError, BadRequestError, RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
            raise SyntheticProviderError(f"OpenAI request failed: {exc}") from exc
        except Exception as exc:
            raise SyntheticProviderError(f"OpenAI request failed: {exc}") from exc

        response_payload = self._response_payload(response)
        message = self._extract_message(response)
        content = self._extract_content(message)
        if not content.strip():
            raise SyntheticProviderError(
                "OpenAI response did not contain a non-empty message content."
            )

        usage_payload = self._usage_payload(response)

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

    def _client(self, metadata: dict[str, Any]) -> OpenAI:
        return OpenAI(
            api_key=self._api_key(),
            organization=self._organization(metadata),
            project=self._project(metadata),
            base_url=self._base_url(),
            timeout=self._timeout_seconds(),
            max_retries=int(self.config.metadata.get("max_retries", 2)),
            default_headers=self._default_headers(metadata),
        )

    def _extract_message(self, response: Any) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if not choices:
            raise SyntheticProviderError("OpenAI response did not contain any choices.")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise SyntheticProviderError("OpenAI response did not contain a message object.")

        if isinstance(message, dict):
            return message

        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                return dumped

        content = getattr(message, "content", None)
        reasoning_details = getattr(message, "reasoning_details", None)
        response_message: dict[str, Any] = {}
        if content is not None:
            response_message["content"] = content
        if reasoning_details is not None:
            response_message["reasoning_details"] = reasoning_details
        return response_message

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

    def _response_payload(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
            if isinstance(payload, dict):
                return payload
        if isinstance(response, dict):
            return response
        return {}

    def _extra_body(self) -> dict[str, Any] | None:
        extra_body = self.config.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            return extra_body
        return None

    def _usage_payload(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            payload = usage.model_dump()
            if isinstance(payload, dict):
                return payload
        if isinstance(usage, dict):
            return usage
        return {}

    def _base_url(self) -> str:
        base_url = (
            self.config.metadata.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return str(base_url)

    def _api_key(self) -> str:
        api_key = self.config.metadata.get("api_key")
        if api_key:
            return str(api_key)

        api_key_env = str(self.config.metadata.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise SyntheticProviderError(
                f"Missing OpenAI API key. Set {api_key_env} or config.metadata['api_key']."
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

    def _default_headers(self, metadata: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {}
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

    def _organization(self, metadata: dict[str, Any]) -> str | None:
        organization = (
            metadata.get("openai_organization")
            or self.config.metadata.get("openai_organization")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        return str(organization) if organization else None

    def _project(self, metadata: dict[str, Any]) -> str | None:
        project = (
            metadata.get("openai_project")
            or self.config.metadata.get("openai_project")
            or os.getenv("OPENAI_PROJECT")
        )
        return str(project) if project else None
