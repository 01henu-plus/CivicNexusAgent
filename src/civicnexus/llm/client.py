from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Callable
import httpx
from pydantic import BaseModel


class LLMError(RuntimeError):
    """The provider did not return a usable structured answer."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 12.0,
        max_retries: int = 1,
        max_tokens: int = 256,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def complete(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        if (
            not self.api_key
            or not self.model
            or not self.endpoint.startswith(("http://", "https://"))
        ):
            raise LLMError("LLM provider is not configured.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content)
                return schema.model_validate(json.loads(content))
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
        raise LLMError(f"LLM structured response failed: {last_error}") from last_error


class FakeLLMClient(LLMClient):
    """Deterministic client used by tests and the offline demo mode."""

    def __init__(
        self, responder: Callable[[str, str, type[BaseModel]], dict[str, Any]] | None = None
    ) -> None:
        self.responder = responder
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append({"system": system, "user": user})
        if self.responder:
            return schema.model_validate(self.responder(system, user, schema))
        return schema.model_validate({})
