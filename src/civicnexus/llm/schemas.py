from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMProposal(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    next_status: str | None = None
    message_zh: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tool_requests: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
