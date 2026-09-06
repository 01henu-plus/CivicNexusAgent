"""Small, serialisable models shared by the context and memory ports.
The models deliberately contain no database-specific types.  A later SQLite or
Redis adapter can persist the same objects without changing the runtime code.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware timestamp (kept as a function for Pydantic)."""
    return datetime.now(timezone.utc)


class MemoryLayer(StrEnum):
    """The intentionally small memory hierarchy used by the demo runtime."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class EventRecord(BaseModel):
    """One append-only task event.
    ``sequence`` is scoped to a task, making replay and checkpoint debugging
    deterministic even when event IDs are random UUIDs.
    """

    event_id: UUID = Field(default_factory=uuid4)
    task_id: str
    sequence: int = Field(ge=1, alias="seq")
    event_type: str = Field(min_length=1)
    actor: str = Field(default="runtime", min_length=1, alias="actor_type")
    payload: dict[str, Any] = Field(default_factory=dict, alias="payload_redacted")
    state_before: str | None = None
    state_after: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    model_config = {"populate_by_name": True}

    # SQL uses the shorter names; properties keep the memory port pleasant
    # to use without duplicating fields in the serialised representation.
    @property
    def seq(self) -> int:
        return self.sequence

    @property
    def actor_type(self) -> str:
        return self.actor

    @property
    def payload_redacted(self) -> dict[str, Any]:
        return self.payload

    @property
    def payload_hash(self) -> str:
        import hashlib

        return hashlib.sha256(
            json.dumps(self.payload, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()


class Fact(BaseModel):
    """A current, structured fact in a task or session ledger."""

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    scope_id: str
    key: str = Field(min_length=1)
    value: Any
    source: str = Field(default="runtime", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    revision: int = Field(default=1, ge=1)
    status: str = "confirmed"
    source_event_ids: list[str] = Field(default_factory=list)
    confirmed_event_id: str | None = None
    supersedes_fact_id: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: str = "normal"
    expires_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def version(self) -> int:
        return self.revision


class FactLedgerSnapshot(BaseModel):
    """Serializable view of all facts at a point in time."""

    scope_id: str
    version: int = Field(default=0, ge=0)
    facts: dict[str, Fact] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryItem(BaseModel):
    """A piece of layered memory.
    ``scope_id`` is mandatory by design: callers must explicitly choose a
    task/session scope instead of accidentally sharing private user context.
    """

    memory_id: UUID = Field(default_factory=uuid4)
    scope_id: str
    layer: MemoryLayer
    key: str = Field(min_length=1)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    # Search results may carry a transient lexical score.  Stored memories
    # keep the default zero value.
    score: float = 0.0


class ContextStats(BaseModel):
    """Observable context-budget metrics emitted with every snapshot."""

    original_chars: int = Field(default=0, ge=0)
    compacted_chars: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    reduction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_units: int = Field(default=0, ge=0)
    compiled_units: int = Field(default=0, ge=0)
    compression_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    retained_fact_count: int = Field(default=0, ge=0)
    source_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    summary_version: str = "local-v1"
    preserved_fields: list[str] = Field(default_factory=list)


class ContextSnapshot(BaseModel):
    """A compact, JSON-safe checkpoint for resuming an agent run."""

    snapshot_id: UUID = Field(default_factory=uuid4)
    task_id: str
    version: int = Field(default=1, ge=1)
    # ``source_seq`` and the ID lists mirror the persistence adapter's
    # append-only/event-sourced representation.  They are optional at the
    # application boundary and remain useful for a purely in-memory run.
    source_seq: int = Field(default=0, ge=0)
    recent_event_ids: list[str] = Field(default_factory=list)
    summary_ids: list[str] = Field(default_factory=list)
    active_fact_ids: list[str] = Field(default_factory=list)
    status: str = "RECEIVED"
    summary: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    recent_messages: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    rendered_context: str = ""
    unit_count: int = Field(default=0, ge=0)
    compression_stats: dict[str, Any] = Field(default_factory=dict)
    stats: ContextStats = Field(default_factory=ContextStats)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def raw_units(self) -> int:
        return int(self.compression_stats.get("raw_units", self.stats.raw_units))

    @property
    def compiled_units(self) -> int:
        return int(self.compression_stats.get("compiled_units", self.stats.compiled_units))

    @property
    def compression_ratio(self) -> float:
        return float(self.compression_stats.get("compression_ratio", self.stats.compression_ratio))


__all__ = [
    "ContextSnapshot",
    "ContextStats",
    "EventRecord",
    "Fact",
    "FactLedgerSnapshot",
    "MemoryItem",
    "MemoryLayer",
    "utc_now",
]
