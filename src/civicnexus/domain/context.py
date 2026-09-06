from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from civicnexus.domain.enums import TaskStatus


class TaskContext(BaseModel):
    task_id: UUID = Field(default_factory=uuid4)
    user_id: str = "anonymous"
    session_id: str = "default"
    user_message: str
    status: TaskStatus = TaskStatus.RECEIVED
    messages: list["ChatMessage"] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    routing: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    checkpoint: str | None = None
    checkpoint_version: int = 0
    last_event_seq: int = 0
    review_round: int = 0
    skill: dict[str, Any] = Field(default_factory=dict)
    pending_tools: list[dict[str, Any]] = Field(default_factory=list)
    created_case_id: str | None = None
    reply: str = ""
    summary: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    memory_hits: list[dict[str, Any]] = Field(default_factory=list)
    context_stats: dict[str, Any] = Field(default_factory=dict)
    rendered_context: str = ""
    snapshot_id: str | None = None
    recent_event_ids: list[str] = Field(default_factory=list)
    summary_ids: list[str] = Field(default_factory=list)
    active_fact_ids: list[str] = Field(default_factory=list)
    pending_profile_fact: dict[str, Any] = Field(default_factory=dict)

    def append_message(self, role: Literal["user", "assistant"], content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        if role == "user":
            self.user_message = content


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompiledContext(BaseModel):
    """Read-only context assembled by the Harness for one Agent step."""

    model_config = {"frozen": True}
    task_id: UUID
    user_id: str
    session_id: str
    status: TaskStatus
    latest_message: str
    recent_messages: tuple[ChatMessage, ...] = ()
    active_facts: dict[str, Any] = Field(default_factory=dict)
    memories: tuple[dict[str, Any], ...] = ()
    summary: str = ""
    missing_fields: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    routing: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    skill: dict[str, Any] = Field(default_factory=dict)
    context_stats: dict[str, Any] = Field(default_factory=dict)
