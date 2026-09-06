"""Storage ports used by the runtime.
Only protocols live here.  They keep orchestration independent from SQLite,
Redis, or any other persistence choice and make the in-memory implementations
useful as deterministic test doubles.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID
from civicnexus.memory.models import (
    ContextSnapshot,
    EventRecord,
    Fact,
    FactLedgerSnapshot,
    MemoryItem,
    MemoryLayer,
)

TaskId = str | UUID


@runtime_checkable
class EventLog(Protocol):
    """Append-only event log for one or more tasks."""

    def append(
        self,
        task_id: TaskId,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        actor: str = "runtime",
        state_before: str | None = None,
        state_after: str | None = None,
    ) -> EventRecord: ...
    def list(self, task_id: TaskId, *, after_sequence: int = 0) -> list[EventRecord]: ...


@runtime_checkable
class FactLedger(Protocol):
    """Current structured facts, scoped to a task or session."""

    def set(
        self,
        scope_id: TaskId,
        key: str,
        value: Any,
        *,
        source: str = "runtime",
        confidence: float = 1.0,
        status: str | None = None,
        source_event_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Fact: ...
    def get(self, scope_id: TaskId, key: str) -> Fact | None: ...
    def all(self, scope_id: TaskId) -> dict[str, Fact]: ...
    def delete(self, scope_id: TaskId, key: str) -> bool: ...
    def snapshot(self, scope_id: TaskId) -> FactLedgerSnapshot: ...
    def restore(self, snapshot: FactLedgerSnapshot) -> None: ...


@runtime_checkable
class LayeredMemory(Protocol):
    """Recallable working, episodic, and semantic memory."""

    def put(
        self,
        scope_id: TaskId,
        layer: MemoryLayer,
        key: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> MemoryItem: ...
    def get(
        self,
        scope_id: TaskId,
        key: str,
        *,
        layer: MemoryLayer | None = None,
    ) -> MemoryItem | None: ...
    def search(
        self,
        scope_id: TaskId,
        query: str = "",
        *,
        layer: MemoryLayer | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]: ...
    def delete(
        self,
        scope_id: TaskId,
        key: str,
        *,
        layer: MemoryLayer | None = None,
    ) -> bool: ...


@runtime_checkable
class SnapshotStore(Protocol):
    """Checkpoint port for compact context snapshots."""

    def save(self, snapshot: ContextSnapshot) -> ContextSnapshot: ...
    def load(self, task_id: TaskId) -> ContextSnapshot | None: ...
    def delete(self, task_id: TaskId) -> bool: ...


__all__ = [
    "EventLog",
    "FactLedger",
    "LayeredMemory",
    "SnapshotStore",
    "TaskId",
]
