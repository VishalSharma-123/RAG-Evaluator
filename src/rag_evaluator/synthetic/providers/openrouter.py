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
class OpenRouterProviderClient(LLMProviderClient):
    """
    Provider client for OpenRouter-hosted models.
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
            metadata=request_metadata,
        )

        response_payload = self._request_completion(payload, request_metadata)
        message = self._extract_message(response_payload)
        content = self._extract_content(message)

        if not content.strip():
            response_payload = self._continue_for_final_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                request_metadata=request_metadata,
                original_message=message,
            )
            message = self._extract_message(response_payload)
            content = self._extract_content(message)

        if not content.strip():
            raise SyntheticProviderError(
                "OpenRouter response did not contain a non-empty message content."
            )

        usage = response_payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else {}

        return ProviderGenerationResult(
            content=content,
            raw_response=response_payload,
            reasoning_details=message.get("reasoning_details"),
            usage=usage_payload,
            response_metadata={
                "provider": "openrouter",
                "model": self.config.model,
                "id": response_payload.get("id"),
            }
        )

    def _request_completion(
            self,
            payload: dict[str, Any],
            metadata: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                self._base_url(),
                headers=self._headers(metadata),
                json=payload,
                timeout=self._timeout_seconds()
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            raise SyntheticProviderError(
                f"OpenRouter request failed with HTTP error: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise SyntheticProviderError(
                f"OpenRouter request failed: {exc}"
            ) from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SyntheticProviderError(
                "OpenRouter response was not valid JSON."
            ) from exc
    
    def _build_payload(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            metadata: dict[str, Any],
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
        
        reasoning_enabled = metadata.get(
            "reasoning_enabled",
            self.config.metadata.get("reasoning_enabled", False),
        )
        if reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        
        extra_body = self.config.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        
        return payload
    
    def _extract_message(
            self,
            response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SyntheticProviderError(
                "OpenRouter response did not contain any choices"
            )
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise SyntheticProviderError(
                "OpenRouter response choice had an  invalid shape."
            )
        
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise SyntheticProviderError(
                "OpenRouter response did not contain any message"
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

        return ""

    def _continue_for_final_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            request_metadata: dict[str, Any],
            original_message: dict[str, Any],
    ) -> dict[str, Any]:
        reasoning_details = original_message.get("reasoning_details")
        if not reasoning_details:
            return {
                "choices": [
                    {
                        "message": original_message,
                    }
                ]
            }

        continuation_payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": original_message.get("content") or "",
                    "reasoning_details": reasoning_details,
                },
                {
                    "role": "user",
                    "content": (
                        "Continue from your prior reasoning and return only the final "
                        "JSON object that satisfies the original instructions. "
                        "Do not include markdown fences or extra explanation."
                    ),
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        extra_body = self.config.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            continuation_payload.update(extra_body)

        return self._request_completion(continuation_payload, request_metadata)
    
    def _base_url(self) -> str:
        base_url = (
            self.config.metadata.get("base_url")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1/chat/completions"
        )
        return str(base_url)
    
    def _api_key(self):
        api_key = self.config.metadata.get("api_key")
        if api_key:
            return str(api_key)

        api_key_env = str(self.config.metadata.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise SyntheticProviderError(
                f"Missing OpenRouter API Key. Set {api_key_env} or config.metadata['api_key']."
            )
        return str(api_key)
    
    def _timeout_seconds(self) -> float:
        raw_timeout = self.config.metadata.get("timeout_seconds", 60)
        
        try:
            return float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise SyntheticProviderError(
                f"Invalid timeout_seconds value: {raw_timeout}"
            ) from exc
    
    def _headers(self, metadata: dict[str, Any]) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        
        http_referer = (
            metadata.get("http_referer")
            or self.config.metadata.get("http_referer")
            or os.getenv("OPENROUTER_HTTP_REFERER")
        )
        app_name = (
            metadata.get("app_name")
            or self.config.metadata.get("app_name")
            or os.getenv("OPENROUTER_APP_NAME")
        )
        
        if http_referer:
            headers["HTTP-Referer"] = str(http_referer)
        if app_name:
            headers["X-Title"] = str(app_name)
        
        return headers
