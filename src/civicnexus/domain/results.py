from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from civicnexus.domain.enums import TaskStatus


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class AgentProposal(BaseModel):
    agent_name: str
    accepted: bool = True
    patch: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    next_status: TaskStatus | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    message_zh: str | None = None
    reason: str = ""
    local_result: dict[str, Any] = Field(default_factory=dict)
    llm_result: dict[str, Any] = Field(default_factory=dict)
    llm_error: str | None = None


class AgentResult(AgentProposal):
    """Backward-compatible name for the proposal returned to the Harness."""
