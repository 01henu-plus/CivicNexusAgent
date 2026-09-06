from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from civicnexus.domain.context import CompiledContext
from civicnexus.domain.results import AgentResult


class Agent(ABC):
    """Contract for an agent that reads a context and returns a structured result."""

    name: ClassVar[str]

    @abstractmethod
    def run(self, context: CompiledContext) -> AgentResult:
        raise NotImplementedError
