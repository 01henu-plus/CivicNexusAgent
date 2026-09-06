"""Small, replaceable language-model adapters."""

from civicnexus.llm.client import LLMClient, LLMError, OpenAICompatibleClient, FakeLLMClient

__all__ = ["LLMClient", "LLMError", "OpenAICompatibleClient", "FakeLLMClient"]
