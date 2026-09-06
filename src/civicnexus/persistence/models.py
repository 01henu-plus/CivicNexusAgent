"""Portable SQLAlchemy persistence models for tasks, events and memory."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, synonym
from sqlalchemy.types import JSON, TypeDecorator
from civicnexus.persistence.database import Base


class PortableJSON(TypeDecorator[Any]):
    """Use JSONB on PostgreSQL and JSON on SQLite/test databases."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        return dialect.type_descriptor(JSONB() if dialect.name == "postgresql" else JSON())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskModel(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    user_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="RECEIVED", index=True, nullable=False)
    context_json: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    checkpoint_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
    # Domain objects call this identifier ``task_id`` while the compact table
    # uses ``id`` as its primary-key attribute.
    task_id = synonym("id")


class RunEventModel(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("task_id", "seq", name="uq_run_events_task_seq"),
        Index("ix_run_events_task_created", "task_id", "created_at"),
    )
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Session/profile events may not belong to a task. Task run events always
    # provide this value and receive a per-task sequence number.
    task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tasks.id"), index=True, nullable=True
    )
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(96), nullable=False)
    state_before: Mapped[str | None] = mapped_column(String(64), nullable=True)
    state_after: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_redacted: Mapped[Any] = mapped_column(PortableJSON, default=dict, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    @property
    def payload_json(self) -> Any:
        """Backward-compatible view of the redacted payload."""
        return self.payload_redacted


# Both names are used in the design; one append-only table is sufficient.
ConversationEventModel = RunEventModel
EventModel = RunEventModel


class FactModel(Base):
    __tablename__ = "facts"
    __table_args__ = (
        Index("ix_facts_namespace_key", "namespace", "namespace_id", "key"),
        Index("ix_facts_active", "namespace", "namespace_id", "status", "expires_at"),
    )
    fact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(16), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_json: Mapped[Any] = mapped_column(PortableJSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="inference", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source_event_ids: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    confirmed_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    supersedes_fact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(16), default="normal", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UserProfileFactModel(Base):
    __tablename__ = "user_profile_facts"
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_profile_key"),
        Index("ix_user_profile_active", "user_id", "status", "expires_at"),
    )
    profile_fact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_json: Mapped[Any] = mapped_column(PortableJSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="confirmed", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source_fact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    confirmation_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(16), default="normal", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UserAccountModel(Base):
    """Persisted credentials for self-registered ordinary users.

    Administrator and the bootstrap demo account remain configuration-backed;
    this table only stores accounts created through the public registration
    endpoint.  Passwords are stored as PBKDF2 encoded hashes, never plaintext.
    """

    __tablename__ = "user_accounts"
    __table_args__ = (
        UniqueConstraint("username", name="uq_user_accounts_username"),
        Index("ix_user_accounts_status", "status"),
    )

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ContextSnapshotModel(Base):
    __tablename__ = "context_snapshots"
    __table_args__ = (Index("ix_snapshots_task_seq", "task_id", "source_seq"),)
    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id"), index=True, nullable=False
    )
    source_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    recent_event_ids: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    summary_ids: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    active_fact_ids: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    rendered_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    unit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compression_stats: Mapped[dict[str, Any]] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    # Optional rich fields let ContextManager restore a snapshot without
    # consulting Redis. They are derived from the immutable event/fact source.
    status: Mapped[str] = mapped_column(String(64), default="RECEIVED", nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    facts: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        PortableJSON, default=list, nullable=False
    )
    recent_messages: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    memory_refs: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class LocalCaseModel(Base):
    __tablename__ = "local_cases"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_local_case_idempotency"),)
    case_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id"), index=True, nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class EvaluationReportModel(Base):
    __tablename__ = "evaluation_reports"
    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict, nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


# Short aliases keep the model layer pleasant to use from small adapters and
# preserve compatibility with either naming convention.
Task = TaskModel
RunEvent = RunEventModel
ConversationEvent = RunEventModel
Fact = FactModel
UserProfileFact = UserProfileFactModel
UserAccount = UserAccountModel
ContextSnapshot = ContextSnapshotModel
LocalCase = LocalCaseModel
EvaluationReport = EvaluationReportModel
__all__ = [
    "TaskModel",
    "Task",
    "RunEventModel",
    "RunEvent",
    "EventModel",
    "ConversationEventModel",
    "ConversationEvent",
    "FactModel",
    "Fact",
    "UserProfileFactModel",
    "UserProfileFact",
    "UserAccountModel",
    "UserAccount",
    "ContextSnapshotModel",
    "ContextSnapshot",
    "LocalCaseModel",
    "LocalCase",
    "EvaluationReportModel",
    "EvaluationReport",
]
