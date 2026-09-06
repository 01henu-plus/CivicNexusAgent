"""SQL adapters for the small Harness memory ports."""

from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from re import findall
from typing import Any
from uuid import UUID
from civicnexus.memory.models import EventRecord, Fact, FactLedgerSnapshot, MemoryItem, MemoryLayer
from civicnexus.persistence.repositories import TaskRepository


def _event(row: Any, task_id: str | UUID) -> EventRecord:
    return EventRecord(
        event_id=row.event_id,
        task_id=str(task_id),
        sequence=row.seq,
        event_type=row.event_type,
        actor=row.actor_type,
        payload=dict(row.payload_redacted or {}),
        state_before=row.state_before,
        state_after=row.state_after,
        created_at=row.created_at,
    )


class SqlEventLog:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def append(
        self,
        task_id: str | UUID,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        actor: str = "runtime",
        state_before: str | None = None,
        state_after: str | None = None,
    ) -> EventRecord:
        body = dict(payload or {})
        if state_before is not None:
            body.setdefault("state_before", state_before)
        if state_after is not None:
            body.setdefault("state_after", state_after)
        row = self.repository.append_event(
            task_id,
            actor,
            event_type,
            body,
            state_before=state_before,
            state_after=state_after,
        )
        return _event(row, task_id)

    def list(self, task_id: str | UUID, *, after_sequence: int = 0) -> list[EventRecord]:
        return [
            _event(row, task_id)
            for row in self.repository.list_events(task_id, after_seq=after_sequence)
        ]

    read = list


class SqlFactLedger:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def set(
        self,
        scope_id: str | UUID,
        key: str,
        value: Any,
        *,
        source: str = "runtime",
        confidence: float = 1.0,
        status: str | None = None,
        source_event_ids: list[str] | None = None,
        importance: float = 0.5,
        sensitivity: str = "normal",
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Fact:
        row = self.repository.add_fact(
            namespace="task",
            namespace_id=str(scope_id),
            key=key,
            value=value,
            status=status or "candidate",
            confidence=confidence,
            source_event_ids=source_event_ids or [],
            importance=importance,
            sensitivity=sensitivity,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
            source=_source_name(source),
        )
        return _fact(row)

    def get(self, scope_id: str | UUID, key: str) -> Fact | None:
        rows = self.repository.list_facts("task", str(scope_id), key=key)
        return _fact(rows[0]) if rows else None

    def all(self, scope_id: str | UUID) -> dict[str, Fact]:
        return {row.key: _fact(row) for row in self.repository.list_facts("task", str(scope_id))}

    values = all

    def delete(self, scope_id: str | UUID, key: str) -> bool:
        return self.repository.delete_fact("task", str(scope_id), key)

    def update(
        self,
        scope_id: str | UUID,
        values: Mapping[str, Any],
        *,
        source: str = "runtime",
        confidence: float = 1.0,
        status: str | None = None,
    ) -> dict[str, Fact]:
        return {
            key: self.set(scope_id, key, value, source=source, confidence=confidence, status=status)
            for key, value in values.items()
        }

    def snapshot(self, scope_id: str | UUID) -> FactLedgerSnapshot:
        facts = self.all(scope_id)
        return FactLedgerSnapshot(
            scope_id=str(scope_id),
            version=max((item.revision for item in facts.values()), default=0),
            facts=facts,
        )

    def restore(self, snapshot: FactLedgerSnapshot) -> None:
        for fact in snapshot.facts.values():
            self.set(
                snapshot.scope_id,
                fact.key,
                fact.value,
                source=fact.source,
                confidence=fact.confidence,
                status=fact.status,
                source_event_ids=fact.source_event_ids,
                importance=fact.importance,
                sensitivity=fact.sensitivity,
                expires_at=fact.expires_at,
                metadata=fact.metadata,
            )

    def confirm_profile(
        self,
        user_id: str,
        key: str,
        value: Any,
        *,
        fact_id: str | None = None,
        task_id: str | UUID | None = None,
        session_id: str | None = None,
    ) -> Any:
        return self.repository.confirm_profile_fact(
            user_id,
            key,
            value,
            fact_id=fact_id,
            task_id=task_id,
            session_id=session_id,
        )

    confirm_user_profile = confirm_profile


class SqlLayeredMemory:
    """Lexical task/session memory backed by the facts tables."""

    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def put(
        self,
        scope_id: str | UUID,
        layer: MemoryLayer,
        key: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> MemoryItem:
        scope, namespace_id = _split_scope(str(scope_id))
        meta = dict(metadata or {})
        if (
            scope == "user"
            and layer == MemoryLayer.SEMANTIC
            and meta.get("confirmation") == "explicit"
        ):
            profile = next(
                (
                    item
                    for item in self.repository.list_profile_facts(namespace_id)
                    if item.key == key
                ),
                None,
            )
            if profile is None or str(profile.value_json) != content:
                profile = self.repository.confirm_profile_fact(
                    namespace_id,
                    key,
                    content,
                    task_id=meta.get("task_id"),
                    session_id=meta.get("session_id"),
                    event_payload={"key": key, "value": content, "source": "explicit_confirmation"},
                )
            return _profile_item(profile, str(scope_id), layer, meta)
        expires_at = expires_at or _memory_expiry(scope, meta)
        row = self.repository.add_fact(
            namespace=f"memory_{layer.value}",
            namespace_id=namespace_id,
            key=key,
            value=content,
            status="confirmed",
            confidence=1.0,
            importance=importance,
            source="memory",
            metadata=meta,
            expires_at=expires_at,
        )
        return _memory_item(row, str(scope_id), layer, meta, content)

    remember = put

    def get(
        self, scope_id: str | UUID, key: str, *, layer: MemoryLayer | None = None
    ) -> MemoryItem | None:
        return next(
            (item for item in self.search(scope_id, key, layer=layer, limit=20) if item.key == key),
            None,
        )

    def search(
        self,
        scope_id: str | UUID,
        query: str = "",
        *,
        layer: MemoryLayer | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scope, namespace_id = _split_scope(str(scope_id))
        layers = [layer] if layer is not None else list(MemoryLayer)
        terms = _query_terms(query)
        result: list[MemoryItem] = []
        for selected in layers:
            for row in self.repository.list_facts(f"memory_{selected.value}", namespace_id):
                item = _memory_item(
                    row,
                    str(scope_id),
                    selected,
                    getattr(row, "metadata_json", {}),
                    str(row.value_json),
                )
                item.score = sum(term in f"{row.key} {row.value_json}".lower() for term in terms)
                if not terms or item.score:
                    result.append(item)
            if scope == "user" and selected == MemoryLayer.SEMANTIC:
                for row in self.repository.list_profile_facts(namespace_id):
                    item = _profile_item(row, str(scope_id), selected)
                    item.score = sum(
                        term in f"{row.key} {row.value_json}".lower() for term in terms
                    )
                    if not terms or item.score:
                        result.append(item)
        return sorted(
            result,
            key=lambda item: (
                -item.score,
                -item.importance,
                -item.updated_at.timestamp(),
                str(item.memory_id),
            ),
        )[:limit]

    recall = search

    def delete(self, scope_id: str | UUID, key: str, *, layer: MemoryLayer | None = None) -> bool:
        item = self.get(scope_id, key, layer=layer)
        if item is None:
            return False
        scope, namespace_id = _split_scope(str(scope_id))
        if (
            scope == "user"
            and (layer is None or layer == MemoryLayer.SEMANTIC)
            and any(
                row.key == key
                for row in self.repository.list_profile_facts(namespace_id, include_inactive=True)
            )
        ):
            self.repository.forget_profile_fact(namespace_id, key)
            return True
        return self.repository.delete_fact(
            f"memory_{(layer or item.layer).value}", namespace_id, key
        )


def _split_scope(scope: str) -> tuple[str, str]:
    return tuple(scope.split(":", 1)) if ":" in scope else ("task", scope)  # type: ignore[return-value]


def _query_terms(query: str) -> set[str]:
    if not query or not query.strip():
        return set()
    text = query.lower().strip()
    terms = set(findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text))
    return terms | ({text} if len(text) > 1 else set())


def _memory_expiry(scope: str, metadata: Mapping[str, Any]) -> datetime | None:
    value = metadata.get("expires_at")
    if value:
        return (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    if scope == "task":
        return datetime.now(timezone.utc) + timedelta(days=1)
    if scope == "session":
        return datetime.now(timezone.utc) + timedelta(days=30)
    return None


def _source_name(source: str) -> str:
    for name in ("correction", "user", "tool", "rule"):
        if name in source.lower():
            return name
    return "inference"


def _fact(row: Any) -> Fact:
    return Fact(
        fact_id=str(row.fact_id),
        scope_id=f"{row.namespace}:{row.namespace_id}",
        key=row.key,
        value=row.value_json,
        source=getattr(row, "source", "runtime"),
        metadata=dict(getattr(row, "metadata_json", {}) or {}),
        confidence=row.confidence,
        revision=row.version,
        status=getattr(row, "status", "candidate") or "candidate",
        source_event_ids=list(getattr(row, "source_event_ids", []) or []),
        confirmed_event_id=getattr(row, "confirmed_event_id", None),
        supersedes_fact_id=getattr(row, "supersedes_fact_id", None),
        importance=getattr(row, "importance", 0.5),
        sensitivity=getattr(row, "sensitivity", "normal"),
        expires_at=getattr(row, "expires_at", None),
        updated_at=row.updated_at or datetime.now(timezone.utc),
    )


def _memory_item(
    row: Any, scope_id: str, layer: MemoryLayer, metadata: Any, content: str
) -> MemoryItem:
    return MemoryItem(
        memory_id=row.fact_id,
        scope_id=scope_id,
        layer=layer,
        key=row.key,
        content=content,
        metadata=dict(metadata or {}),
        importance=row.importance,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _profile_item(
    row: Any, scope_id: str, layer: MemoryLayer, metadata: Mapping[str, Any] | None = None
) -> MemoryItem:
    return MemoryItem(
        memory_id=row.profile_fact_id,
        scope_id=scope_id,
        layer=layer,
        key=row.key,
        content=str(row.value_json),
        metadata=dict(metadata or {"confirmation": "explicit"}),
        importance=row.importance,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


__all__ = ["SqlEventLog", "SqlFactLedger", "SqlLayeredMemory"]
