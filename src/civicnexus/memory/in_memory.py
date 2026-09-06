"""Deterministic in-memory adapters for the memory protocols.
These adapters are intentionally small.  They are suitable for a local demo
and for unit tests; production code can replace each port independently.
"""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from re import findall
from re import sub
from threading import RLock
from typing import Any
from uuid import uuid4
from civicnexus.memory.models import (
    ContextSnapshot,
    EventRecord,
    Fact,
    FactLedgerSnapshot,
    MemoryItem,
    MemoryLayer,
)
from civicnexus.memory.protocols import TaskId


def _scope(value: TaskId) -> str:
    return str(value)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expired(fact: Fact) -> bool:
    expires = fact.expires_at
    return bool(expires and expires <= _now())


class InMemoryEventLog:
    """Append-only event log with per-task monotonically increasing sequence."""

    def __init__(self) -> None:
        self._events: dict[str, list[EventRecord]] = {}
        self._lock = RLock()

    def append(
        self,
        task_id: TaskId,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        actor: str = "runtime",
        state_before: str | None = None,
        state_after: str | None = None,
    ) -> EventRecord:
        if not event_type or not event_type.strip():
            raise ValueError("event_type must not be empty")
        if not actor or not actor.strip():
            raise ValueError("actor must not be empty")
        task_key = _scope(task_id)
        with self._lock:
            bucket = self._events.setdefault(task_key, [])
            record = EventRecord(
                task_id=task_key,
                sequence=len(bucket) + 1,
                event_type=event_type.strip(),
                actor=actor.strip(),
                payload=_redact(dict(payload or {})),
                state_before=state_before,
                state_after=state_after,
            )
            bucket.append(record)
            return record.model_copy(deep=True)

    def list(self, task_id: TaskId, *, after_sequence: int = 0) -> list[EventRecord]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        with self._lock:
            return [
                event.model_copy(deep=True)
                for event in self._events.get(_scope(task_id), [])[after_sequence:]
            ]

    read = list

    def clear(self, task_id: TaskId | None = None) -> None:
        with self._lock:
            if task_id is None:
                self._events.clear()
            else:
                self._events.pop(_scope(task_id), None)


class InMemoryFactLedger:
    """Versioned, conflict-aware facts while keeping only active values in ``all``."""

    def __init__(self) -> None:
        self._facts: dict[str, dict[str, Fact]] = {}
        self._history: dict[str, list[Fact]] = {}
        self._versions: dict[str, int] = {}
        self._lock = RLock()

    def set(
        self,
        scope_id: TaskId,
        key: str,
        value: Any,
        *,
        source: str = "runtime",
        confidence: float = 1.0,
        status: str | None = None,
        source_event_ids: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        sensitivity: str = "normal",
        expires_at: datetime | None = None,
    ) -> Fact:
        if not key or not key.strip():
            raise ValueError("fact key must not be empty")
        if not source or not source.strip():
            raise ValueError("fact source must not be empty")
        scope_key = _scope(scope_id)
        with self._lock:
            bucket = self._facts.setdefault(scope_key, {})
            item_key = key.strip()
            previous = bucket.get(item_key)
            source = source.strip()
            chosen_status = status or (
                "candidate" if source in {"llm", "inference", "context"} else "confirmed"
            )
            priority = {
                "correction": 5,
                "user": 4,
                "tool": 3,
                "rule": 2,
                "context": 1,
                "llm": 0,
                "inference": 0,
            }
            if previous and priority.get(source, 0) < priority.get(previous.source, 0):
                rejected = Fact(
                    fact_id=str(uuid4()),
                    scope_id=scope_key,
                    key=item_key,
                    value=deepcopy(value),
                    source=source,
                    confidence=confidence,
                    revision=previous.revision + 1,
                    status="rejected",
                    source_event_ids=list(source_event_ids or []),
                    metadata=deepcopy(dict(metadata or {})),
                    supersedes_fact_id=previous.fact_id,
                    importance=importance,
                    sensitivity=sensitivity,
                    expires_at=expires_at,
                    updated_at=_now(),
                )
                self._history.setdefault(scope_key, []).append(rejected.model_copy(deep=True))
                self._versions[scope_key] = self._versions.get(scope_key, 0) + 1
                return previous.model_copy(deep=True)
            if (
                previous
                and previous.value == value
                and (previous.source == source)
                and (previous.status == chosen_status)
            ):
                return previous.model_copy(deep=True)
            revision = previous.revision + 1 if previous else 1
            fact = Fact(
                fact_id=str(uuid4()),
                scope_id=scope_key,
                key=item_key,
                value=deepcopy(value),
                source=source,
                confidence=confidence,
                revision=revision,
                status=chosen_status,
                source_event_ids=list(source_event_ids or []),
                metadata=deepcopy(dict(metadata or {})),
                supersedes_fact_id=previous.fact_id if previous else None,
                importance=importance,
                sensitivity=sensitivity,
                expires_at=expires_at,
                updated_at=_now(),
            )
            if previous:
                previous.status = "superseded"
                self._history.setdefault(scope_key, []).append(previous.model_copy(deep=True))
            bucket[fact.key] = fact
            self._history.setdefault(scope_key, []).append(fact.model_copy(deep=True))
            self._versions[scope_key] = self._versions.get(scope_key, 0) + 1
            return fact.model_copy(deep=True)

    def update(
        self,
        scope_id: TaskId,
        values: Mapping[str, Any],
        *,
        source: str = "runtime",
        confidence: float = 1.0,
        status: str | None = None,
    ) -> dict[str, Fact]:
        """Set several facts and return their resulting records."""
        return {
            key: self.set(scope_id, key, value, source=source, confidence=confidence, status=status)
            for key, value in values.items()
        }

    def get(self, scope_id: TaskId, key: str) -> Fact | None:
        with self._lock:
            fact = self._facts.get(_scope(scope_id), {}).get(key)
            if fact and fact.status in {"candidate", "confirmed"} and (not _expired(fact)):
                return fact.model_copy(deep=True)
            return None

    def all(self, scope_id: TaskId, *, include_inactive: bool = False) -> dict[str, Fact]:
        with self._lock:
            values = {
                key: fact.model_copy(deep=True)
                for key, fact in self._facts.get(_scope(scope_id), {}).items()
                if include_inactive
                or (fact.status in {"candidate", "confirmed"} and (not _expired(fact)))
            }
            return values

    values = all

    def history(self, scope_id: TaskId, key: str | None = None) -> list[Fact]:
        with self._lock:
            values = self._history.get(_scope(scope_id), [])
            return [item.model_copy(deep=True) for item in values if key is None or item.key == key]

    def delete(self, scope_id: TaskId, key: str) -> bool:
        scope_key = _scope(scope_id)
        with self._lock:
            bucket = self._facts.get(scope_key)
            if (
                not bucket
                or key not in bucket
                or bucket[key].status not in {"candidate", "confirmed"}
            ):
                return False
            bucket[key].status = "rejected"
            self._history.setdefault(scope_key, []).append(bucket[key].model_copy(deep=True))
            self._versions[scope_key] = self._versions.get(scope_key, 0) + 1
            return True

    def snapshot(self, scope_id: TaskId) -> FactLedgerSnapshot:
        scope_key = _scope(scope_id)
        return FactLedgerSnapshot(
            scope_id=scope_key, version=self._versions.get(scope_key, 0), facts=self.all(scope_key)
        )

    def restore(self, snapshot: FactLedgerSnapshot) -> None:
        scope_key = _scope(snapshot.scope_id)
        with self._lock:
            for fact in snapshot.facts.values():
                if fact.scope_id != scope_key:
                    raise ValueError("fact scope does not match ledger snapshot")
            self._facts[scope_key] = {
                key: fact.model_copy(deep=True) for key, fact in snapshot.facts.items()
            }
            self._history[scope_key] = [
                fact.model_copy(deep=True) for fact in snapshot.facts.values()
            ]
            self._versions[scope_key] = snapshot.version


class InMemoryLayeredMemory:
    """Small lexical memory store separated by scope and memory layer."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, MemoryLayer, str], MemoryItem] = {}
        self._lock = RLock()

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
    ) -> MemoryItem:
        if not key or not key.strip():
            raise ValueError("memory key must not be empty")
        if not isinstance(content, str):
            raise TypeError("memory content must be a string")
        scope_key = _scope(scope_id)
        layer_value = MemoryLayer(layer)
        item_key = key.strip()
        storage_key = (scope_key, layer_value, item_key)
        metadata = dict(metadata or {})
        if expires_at is not None:
            metadata.setdefault("expires_at", expires_at.isoformat())
        elif "expires_at" not in metadata and str(scope_key).split(":", 1)[0] in {
            "task",
            "session",
        }:
            metadata["expires_at"] = (
                _now() + timedelta(days=1 if str(scope_key).startswith("task:") else 30)
            ).isoformat()
        with self._lock:
            previous = self._items.get(storage_key)
            item = MemoryItem(
                memory_id=previous.memory_id if previous else uuid4(),
                scope_id=scope_key,
                layer=layer_value,
                key=item_key,
                content=content,
                metadata=deepcopy(metadata),
                importance=importance,
                created_at=previous.created_at if previous else _now(),
                updated_at=_now(),
            )
            self._items[storage_key] = item
            return item.model_copy(deep=True)

    remember = put

    def get(
        self, scope_id: TaskId, key: str, *, layer: MemoryLayer | None = None
    ) -> MemoryItem | None:
        scope_key = _scope(scope_id)
        with self._lock:
            if layer is not None:
                item = self._items.get((scope_key, MemoryLayer(layer), key))
                return item.model_copy(deep=True) if item and not _memory_expired(item) else None
            candidates = [
                item
                for (item_scope, _, item_key), item in self._items.items()
                if item_scope == scope_key and item_key == key
            ]
            if not candidates:
                return None
            candidates = [item for item in candidates if not _memory_expired(item)]
            if not candidates:
                return None
            item = max(candidates, key=lambda value: value.updated_at)
            return item.model_copy(deep=True)

    def search(
        self, scope_id: TaskId, query: str = "", *, layer: MemoryLayer | None = None, limit: int = 5
    ) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scope_key = _scope(scope_id)
        selected_layer = MemoryLayer(layer) if layer is not None else None
        terms = _query_terms(query)
        with self._lock:
            candidates = [
                item
                for (item_scope, item_layer, _), item in self._items.items()
                if item_scope == scope_key
                and (selected_layer is None or item_layer == selected_layer)
                and (not _memory_expired(item))
            ]
            ranked: list[tuple[float, MemoryItem]] = []
            for item in candidates:
                haystack = f"{item.key} {item.content}".lower()
                score = float(sum((1 for term in terms if term in haystack))) if terms else 0.0
                if terms and score == 0:
                    continue
                ranked.append((score, item))
            ranked.sort(
                key=lambda pair: (
                    -pair[0],
                    -pair[1].importance,
                    -pair[1].updated_at.timestamp(),
                    str(pair[1].memory_id),
                )
            )
            results: list[MemoryItem] = []
            for score, item in ranked[:limit]:
                copy = item.model_copy(deep=True)
                copy.score = score
                results.append(copy)
            return results

    recall = search

    def delete(self, scope_id: TaskId, key: str, *, layer: MemoryLayer | None = None) -> bool:
        scope_key = _scope(scope_id)
        with self._lock:
            if layer is not None:
                return self._items.pop((scope_key, MemoryLayer(layer), key), None) is not None
            keys = [
                storage_key
                for storage_key in self._items
                if storage_key[0] == scope_key and storage_key[2] == key
            ]
            for storage_key in keys:
                del self._items[storage_key]
            return bool(keys)

    def clear(self, scope_id: TaskId | None = None) -> None:
        with self._lock:
            if scope_id is None:
                self._items.clear()
                return
            scope_key = _scope(scope_id)
            for storage_key in [key for key in self._items if key[0] == scope_key]:
                del self._items[storage_key]


def _query_terms(query: str) -> set[str]:
    if not query or not query.strip():
        return set()
    normalized = query.lower().strip()
    terms = set(findall("[a-z0-9_]+|[\\u4e00-\\u9fff]", normalized))
    if len(normalized) > 1:
        terms.add(normalized)
    return terms


def _redact(value: Any) -> dict[str, Any]:
    hidden = {"token", "api_key", "apikey", "password", "secret"}

    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: "[REDACTED]" if str(key).lower() in hidden else clean(val)
                for key, val in item.items()
            }
        if isinstance(item, list):
            return [clean(val) for val in item]
        if isinstance(item, str):
            return sub(
                "\\b1\\d{10}\\b",
                "<phone>",
                sub("[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}", "<email>", item),
            )
        return item

    return clean(value)


def _memory_expired(item: MemoryItem) -> bool:
    value = item.metadata.get("expires_at")
    if not value:
        return False
    try:
        from datetime import datetime

        expiry = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        return expiry <= _now()
    except (TypeError, ValueError):
        return False


class InMemorySnapshotStore:
    """Latest-snapshot store with a simple optimistic-version guard."""

    def __init__(self) -> None:
        self._snapshots: dict[str, ContextSnapshot] = {}
        self._lock = RLock()

    def save(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        task_key = _scope(snapshot.task_id)
        with self._lock:
            previous = self._snapshots.get(task_key)
            if previous and snapshot.version < previous.version:
                raise ValueError("cannot overwrite a newer context snapshot")
            self._snapshots[task_key] = snapshot.model_copy(deep=True)
            return snapshot.model_copy(deep=True)

    def load(self, task_id: TaskId) -> ContextSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(_scope(task_id))
            return snapshot.model_copy(deep=True) if snapshot else None

    latest = load

    def delete(self, task_id: TaskId) -> bool:
        with self._lock:
            return self._snapshots.pop(_scope(task_id), None) is not None


class InMemoryMemoryStore:
    """Convenience bundle; each port remains independently replaceable."""

    def __init__(self) -> None:
        self.events = InMemoryEventLog()
        self.facts = InMemoryFactLedger()
        self.layers = InMemoryLayeredMemory()
        self.snapshots = InMemorySnapshotStore()


__all__ = [
    "InMemoryEventLog",
    "InMemoryFactLedger",
    "InMemoryLayeredMemory",
    "InMemoryMemoryStore",
    "InMemorySnapshotStore",
]
