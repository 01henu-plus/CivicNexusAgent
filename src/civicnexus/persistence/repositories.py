"""Small SQLAlchemy repository used by the Harness as its durable source."""

from __future__ import annotations
import hashlib
import json
import re
from copy import copy
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import UUID, uuid4
from sqlalchemy import desc, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from civicnexus.domain.context import TaskContext
from civicnexus.persistence.database import SessionLocal, session_scope
from civicnexus.persistence.models import (
    ContextSnapshotModel,
    EvaluationReportModel,
    FactModel,
    LocalCaseModel,
    RunEventModel,
    TaskModel,
    UserAccountModel,
    UserProfileFactModel,
)


def _id(value: UUID | str) -> str:
    return str(value)


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _redact(value: Any) -> Any:
    hidden = {"token", "api_key", "apikey", "password", "secret"}
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in hidden else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "<email>", re.sub(r"\b1\d{10}\b", "<phone>", value)
        )
    return value


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _tool_identity(
    name: str, arguments: Any, idempotency_key: str | None = None
) -> tuple[str, str]:
    args = arguments if isinstance(arguments, dict) else {}
    return str(name), str(
        idempotency_key
        or args.get("idempotency_key")
        or json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    )


def _as_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None


def _query_terms(query: str) -> set[str]:
    if not query or not query.strip():
        return set()
    text = query.lower().strip()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)) | (
        {text} if len(text) > 1 else set()
    )


def _normalise_snapshot(row: ContextSnapshotModel | None) -> ContextSnapshotModel | None:
    if row is None:
        return None
    for key in ("facts", "state", "stats", "compression_stats"):
        if getattr(row, key, None) is None:
            setattr(row, key, {})
    for key in (
        "evidence",
        "recent_messages",
        "memory_refs",
        "recent_event_ids",
        "summary_ids",
        "active_fact_ids",
    ):
        if getattr(row, key, None) is None:
            setattr(row, key, [])
    if row.summary is None:
        row.summary = ""
    if row.rendered_context is None:
        row.rendered_context = ""
    if row.status is None:
        row.status = "RECEIVED"
    return row


def _source_rank(source: str | None) -> int:
    text = (source or "").lower()
    if "correction" in text or "correct" in text:
        return 5
    if "user" in text or "explicit" in text:
        return 4
    if "tool" in text or "verified" in text:
        return 3
    if any(word in text for word in ("rule", "runtime", "context", "memory")):
        return 2
    return 1


def _facts_at(rows: list[FactModel], event_seq: dict[str, int], cutoff: int) -> list[FactModel]:
    """Reduce versioned rows to values visible at an event checkpoint."""
    if cutoff <= 0:
        return []
    grouped: dict[str, list[FactModel]] = {}
    for row in rows:
        refs = [str(ref) for ref in (row.source_event_ids or [])]
        if refs and any(event_seq.get(ref, cutoff + 1) > cutoff for ref in refs):
            continue
        grouped.setdefault(row.key, []).append(row)
    result: list[FactModel] = []
    for values in grouped.values():
        values.sort(key=lambda row: (int(row.version), str(row.fact_id)))
        latest = values[-1]
        latest_status = _fact_status_at(latest, event_seq, cutoff)
        if latest_status in {"candidate", "confirmed"}:
            result.append(_fact_view(latest, latest_status))
        elif latest_status == "superseded":
            # Its successor is outside the cutoff, so the prior value remains
            # the effective fact at this historical point.
            result.append(_fact_view(latest, latest_status))
        elif (
            latest_status == "rejected"
            and latest.supersedes_fact_id
            and not _tombstone_seen(latest, event_seq, cutoff)
        ):
            prior = next(
                (
                    row
                    for row in reversed(values[:-1])
                    if row.fact_id == latest.supersedes_fact_id
                    and _fact_status_at(row, event_seq, cutoff)
                    in {"candidate", "confirmed", "superseded"}
                ),
                None,
            )
            if prior is not None:
                result.append(_fact_view(prior, _fact_status_at(prior, event_seq, cutoff)))
    return result


def _fact_status_at(row: FactModel, event_seq: dict[str, int], cutoff: int) -> str:
    status = row.status
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    marker = metadata.get("_tombstone_event_id")
    tombstone_seq = event_seq.get(str(marker)) if marker else None
    if status in {"rejected", "forgotten"} and tombstone_seq is not None and tombstone_seq > cutoff:
        return _confirmed_status_at(row, event_seq, cutoff)
    if status == "superseded":
        return _confirmed_status_at(row, event_seq, cutoff)
    return status


def _confirmed_status_at(row: FactModel, event_seq: dict[str, int], cutoff: int) -> str:
    marker = row.confirmed_event_id
    return (
        "confirmed" if marker and event_seq.get(str(marker), cutoff + 1) <= cutoff else "candidate"
    )


def _fact_view(row: FactModel, status: str) -> FactModel:
    view = copy(row)
    view.status = status
    return view


def _tombstone_seen(row: FactModel, event_seq: dict[str, int], cutoff: int) -> bool:
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    marker = metadata.get("_tombstone_event_id")
    return bool(marker and event_seq.get(str(marker), cutoff + 1) <= cutoff)


def _summary_parts(messages: list[str]) -> list[str]:
    return [
        "历史分段：" + "；".join(messages[index : index + 8])[:1200]
        for index in range(0, len(messages), 8)
    ]


def _clip(value: str, limit: int) -> str:
    return "" if limit <= 0 else value if len(value) <= limit else value[: max(1, limit - 1)] + "…"


def _summary_id(value: str) -> str:
    return "summary-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _replay_events(
    events: list[RunEventModel], base: dict[str, Any], default_status: str
) -> tuple[Any, ...]:
    """Reduce selected immutable events into snapshot fields without current-state leakage."""
    if not events:
        state = dict(base.get("state") or {})
        for key in (
            "routing",
            "review",
            "skill",
            "missing_fields",
            "review_round",
            "pending_tools",
            "created_case_id",
            "reply",
            "pending_profile_fact",
        ):
            if base.get(key) not in (None, {}, []):
                state[key] = base[key]
        return (
            str(base.get("status", default_status)),
            list(base.get("recent_messages") or base.get("messages") or []),
            state,
            str(base.get("summary") or ""),
            list(base.get("evidence") or [])[:3],
            list(base.get("memory_refs") or []),
            max(1, int(base.get("checkpoint_version", 1) or 1)),
        )
    status, messages, state, evidence, memory_refs = "RECEIVED", [], {}, [], []
    roles: list[str] = []
    version, review_round = 1, 0
    successful: set[tuple[str, str]] = set()
    for event in events:
        payload = event.payload_redacted if isinstance(event.payload_redacted, dict) else {}
        kind = str(event.event_type).upper()
        if kind in {"TASK_CREATED", "USER_MESSAGE", "ASSISTANT_MESSAGE"}:
            if message := payload.get("message", payload.get("text")):
                messages.append(str(message))
                roles.append("assistant" if kind == "ASSISTANT_MESSAGE" else "user")
                if kind == "ASSISTANT_MESSAGE":
                    state["reply"] = str(message)
        elif kind == "STATE_TRANSITION":
            target = event.state_after or payload.get("to") or payload.get("state_after")
            status = (
                str(target.get("value") or target.get("name"))
                if isinstance(target, dict)
                else str(target or status)
            )
            review_round += status == "REVIEWING"
        elif kind == "AGENT_PROPOSAL":
            patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
            for key in ("routing", "review", "skill"):
                if key in patch:
                    state[key] = patch[key]
            state["missing_fields"] = payload.get("missing_fields", [])
            state["pending_tools"] = payload.get("tool_requests", [])
            if payload.get("message_zh"):
                state["reply"] = payload["message_zh"]
            if payload.get("evidence"):
                evidence = list(payload["evidence"])[:3]
        elif kind == "TOOL_CALL" and payload.get("success") is True:
            state["pending_tools"] = []
            name, result = payload.get("name"), payload.get("result")
            successful.add(_tool_identity(name or "", payload.get("arguments") or {}))
            if name == "case_search" and isinstance(result, list):
                evidence = result[:3]
            elif name == "create_local_case" and isinstance(result, dict) and result.get("case_id"):
                state["created_case_id"] = result["case_id"]
        elif kind == "MEMORY_READ":
            memory_refs = [
                str(item.get("memory_id"))
                for item in payload.get("hits", [])
                if isinstance(item, dict) and item.get("memory_id")
            ]
        elif kind == "MEMORY_WRITE":
            candidate = payload.get("profile_candidate")
            if (
                isinstance(candidate, dict)
                and candidate.get("key")
                and candidate.get("value") is not None
            ):
                state["pending_profile_fact"] = dict(candidate)
            if payload.get("reply"):
                state["reply"] = str(payload["reply"])
        elif kind in {
            "PROFILE_CONFIRMATION",
            "PROFILE_CONFIRMED",
            "PROFILE_REJECTION",
            "PROFILE_REJECTED",
            "PROFILE_FORGET",
            "PROFILE_FORGOTTEN",
        }:
            state.pop("pending_profile_fact", None)
        elif kind == "HARNESS_PHASE":
            detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
            if payload.get("phase") == "SELECT_WORKFLOW_STEP" and detail.get("skill"):
                manifest = detail.get("skill_manifest")
                state["skill"] = (
                    dict(manifest)
                    if isinstance(manifest, dict)
                    else {"name": detail["skill"], "version": detail.get("skill_version")}
                )
        elif kind == "CHECKPOINT":
            version = max(version, int(payload.get("version", version) or version))
    pending = state.get("pending_tools") or []
    state["pending_tools"] = [
        item
        for item in pending
        if isinstance(item, dict)
        and _tool_identity(
            item.get("name", ""), item.get("arguments") or {}, item.get("idempotency_key")
        )
        not in successful
    ]
    if roles:
        state["message_roles"] = roles
    state["review_round"] = review_round
    state = {key: value for key, value in state.items() if value not in (None, {}, [])}
    summary = _clip("；".join(_summary_parts(messages[:-6])), 600)
    return status, messages, state, summary, evidence, memory_refs, version


class TaskRepository:
    def __init__(
        self,
        session_factory: sessionmaker[Session] | Session | Engine = SessionLocal,
        *,
        session: Session | None = None,
        db: Session | None = None,
        engine: Engine | None = None,
    ) -> None:
        session = session or db
        if engine is not None:
            session_factory = engine
        if isinstance(session_factory, Session):
            self.session_factory, self.session = SessionLocal, session_factory
        elif isinstance(session_factory, Engine):
            self.session_factory, self.session = (
                sessionmaker(
                    bind=session_factory, class_=Session, autoflush=False, expire_on_commit=False
                ),
                session,
            )
        else:
            self.session_factory, self.session = session_factory, session

    @contextmanager
    def _scope(self) -> Iterator[Session]:
        if self.session is not None:
            yield self.session
        else:
            with session_scope(self.session_factory) as db:
                yield db

    @staticmethod
    def _update_task(
        row: TaskModel,
        context: TaskContext,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if user_id is not None:
            row.user_id = user_id
        if session_id is not None:
            row.session_id = session_id
        row.user_message, row.status, row.context_json = (
            _redact(context.user_message),
            context.status.value,
            _redact(_json(context)),
        )
        row.checkpoint_version = max(row.checkpoint_version, context.checkpoint_version)

    def create_task(
        self, context: TaskContext, *, user_id: str | None = None, session_id: str | None = None
    ) -> TaskModel:
        with self._scope() as db:
            row = db.get(TaskModel, _id(context.task_id))
            if row is None:
                row = TaskModel(
                    id=_id(context.task_id),
                    user_id=user_id or context.user_id,
                    session_id=session_id or context.session_id,
                    user_message=_redact(context.user_message),
                    status=context.status.value,
                    context_json=_redact(_json(context)),
                )
                db.add(row)
            else:
                self._update_task(
                    row, context, user_id or context.user_id, session_id or context.session_id
                )
            db.flush()
            return row

    def save_task(
        self, context: TaskContext, *, user_id: str | None = None, session_id: str | None = None
    ) -> TaskModel:
        with self._scope() as db:
            row = db.get(TaskModel, _id(context.task_id))
            if row is None:
                row = TaskModel(id=_id(context.task_id))
                db.add(row)
            self._update_task(
                row, context, user_id or context.user_id, session_id or context.session_id
            )
            db.flush()
            if context.checkpoint or context.checkpoint_version:
                self._save_task_snapshot(db, context)
            return row

    def save(self, value: Any) -> Any:
        return (
            self.save_task(value)
            if isinstance(value, TaskContext) or hasattr(value, "user_message")
            else self.save_snapshot(value)
        )

    def get_task_model(self, task_id: UUID | str) -> TaskModel | None:
        with self._scope() as db:
            return db.get(TaskModel, _id(task_id))

    def get_task(self, task_id: UUID | str) -> TaskContext | None:
        row = self.get_task_model(task_id)
        if row is None:
            return None
        data = dict(row.context_json or {})
        data.update(
            {
                "task_id": data.get("task_id", row.id),
                "user_message": data.get("user_message", row.user_message),
                "status": row.status,
            }
        )
        return TaskContext.model_validate(data)

    def list_tasks(self, limit: int = 50) -> list[TaskModel]:
        with self._scope() as db:
            return list(
                db.scalars(select(TaskModel).order_by(desc(TaskModel.updated_at)).limit(limit))
            )

    @staticmethod
    def _account_username(username: str) -> str:
        """Use one canonical representation for account lookup and creation."""
        return username.strip().lower()

    def get_user_account(self, username: str) -> UserAccountModel | None:
        """Return an active or inactive registered account by username."""
        key = self._account_username(username)
        with self._scope() as db:
            return db.scalar(select(UserAccountModel).where(UserAccountModel.username == key))

    # ``find`` is a convenient name for callers that do not need to know the
    # persistence model's exact method name.
    find_user_account = get_user_account

    def create_user_account(
        self,
        username: str,
        password_hash: str,
        *,
        display_name: str | None = None,
        user_id: str | None = None,
    ) -> UserAccountModel:
        """Create one ordinary-user account and fail on a duplicate username."""
        key = self._account_username(username)
        if not key:
            raise ValueError("username is required")
        if not password_hash:
            raise ValueError("password_hash is required")
        with self._scope() as db:
            if (
                db.scalar(select(UserAccountModel).where(UserAccountModel.username == key))
                is not None
            ):
                raise ValueError("username already exists")
            row = UserAccountModel(
                user_id=user_id or f"user-{uuid4().hex}",
                username=key,
                password_hash=password_hash,
                display_name=(display_name or key).strip() or key,
            )
            db.add(row)
            try:
                db.flush()
            except IntegrityError as exc:
                db.rollback()
                raise ValueError("username already exists") from exc
            return row

    add_user_account = create_user_account

    def overview(self) -> dict[str, Any]:
        with self._scope() as db:
            rows = db.execute(
                select(TaskModel.status, func.count()).group_by(TaskModel.status)
            ).all()
            return {
                "task_count": sum(int(count) for _, count in rows),
                "status_counts": {str(status): int(count) for status, count in rows},
                "event_count": int(db.scalar(select(func.count()).select_from(RunEventModel)) or 0),
                "case_count": int(db.scalar(select(func.count()).select_from(LocalCaseModel)) or 0),
            }

    def append_event(
        self,
        task_id: UUID | str | None,
        actor_type: str | None = None,
        event_type: str | Any | None = None,
        payload: Any = None,
        *,
        actor: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        payload_redacted: Any | None = None,
        created_at: datetime | None = None,
        state_before: str | Any | None = None,
        state_after: str | Any | None = None,
    ) -> RunEventModel:
        # Accept both (task, actor, type, payload) and (task, type, payload, actor=...).
        if event_type is not None and not isinstance(event_type, str):
            payload, event_type, actor_type = event_type, actor_type, actor or "runtime"
        elif event_type is None:
            event_type, actor_type = actor_type, actor or "runtime"
        if not event_type:
            raise ValueError("event_type is required")
        with self._scope() as db:
            return self._append_event(
                db,
                task_id,
                actor_type,
                str(event_type),
                payload,
                actor=actor,
                session_id=session_id,
                user_id=user_id,
                payload_redacted=payload_redacted,
                created_at=created_at,
                state_before=state_before,
                state_after=state_after,
            )

    def _append_event(
        self,
        db: Session,
        task_id: UUID | str | None,
        actor_type: str | None,
        event_type: str,
        payload: Any = None,
        *,
        actor: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        payload_redacted: Any | None = None,
        created_at: datetime | None = None,
        state_before: str | Any | None = None,
        state_after: str | Any | None = None,
    ) -> RunEventModel:
        task_key = _id(task_id) if task_id is not None else None
        task = db.get(TaskModel, task_key) if task_key else None
        if task_key and task is None:
            raise KeyError(f"task not found: {task_key}")
        if task is not None:
            task = db.scalar(select(TaskModel).where(TaskModel.id == task_key).with_for_update())
            seq = (
                int(
                    db.scalar(
                        select(func.max(RunEventModel.seq)).where(RunEventModel.task_id == task_key)
                    )
                    or 0
                )
                + 1
            )
            session_id = session_id if session_id is not None else task.session_id
            user_id = user_id if user_id is not None else task.user_id
        else:
            seq = 0
        redacted = _redact(_json(payload if payload_redacted is None else payload_redacted))
        row = RunEventModel(
            event_id=str(uuid4()),
            task_id=task_key,
            session_id=session_id,
            user_id=user_id,
            seq=seq,
            actor_type=actor or actor_type or "runtime",
            event_type=event_type,
            state_before=getattr(state_before, "value", state_before),
            state_after=getattr(state_after, "value", state_after),
            payload_redacted=redacted,
            payload_hash=_payload_hash(redacted),
            created_at=created_at or datetime.now(timezone.utc),
        )
        db.add(row)
        db.flush()
        return row

    append_run_event = append_event

    def list_events(
        self, task_id: UUID | str, *, after_seq: int = 0, after_sequence: int | None = None
    ) -> list[RunEventModel]:
        after_seq = after_seq if after_sequence is None else after_sequence
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        with self._scope() as db:
            return list(
                db.scalars(
                    select(RunEventModel)
                    .where(RunEventModel.task_id == _id(task_id), RunEventModel.seq > after_seq)
                    .order_by(RunEventModel.seq)
                )
            )

    def list_user_events(
        self, user_id: str, *, session_id: str | None = None
    ) -> list[RunEventModel]:
        with self._scope() as db:
            query = select(RunEventModel).where(RunEventModel.user_id == user_id)
            if session_id is not None:
                query = query.where(RunEventModel.session_id == session_id)
            return list(
                db.scalars(query.order_by(RunEventModel.created_at, RunEventModel.event_id))
            )

    def add_fact(
        self,
        *,
        namespace: str,
        namespace_id: str,
        key: str,
        value: Any,
        status: str = "candidate",
        confidence: float = 0.0,
        source_event_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        sensitivity: str = "normal",
        expires_at: datetime | None = None,
        source: str = "inference",
    ) -> FactModel:
        if not namespace or not namespace_id or not key:
            raise ValueError("namespace, namespace_id and key are required")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with self._scope() as db:
            return self._add_fact(
                db,
                namespace=namespace,
                namespace_id=_id(namespace_id),
                key=key,
                value=value,
                status=status,
                confidence=confidence,
                source_event_ids=source_event_ids or [],
                metadata=metadata or {},
                importance=importance,
                sensitivity=sensitivity,
                expires_at=expires_at,
                source=source,
            )

    def _add_fact(self, db: Session, **data: Any) -> FactModel:
        source = data.pop("source", "inference")
        value, metadata = data.pop("value"), data.pop("metadata", {})
        rows = list(
            db.scalars(
                select(FactModel)
                .where(
                    FactModel.namespace == data["namespace"],
                    FactModel.namespace_id == data["namespace_id"],
                    FactModel.key == data["key"],
                )
                .order_by(desc(FactModel.version), desc(FactModel.updated_at))
            )
        )
        latest = rows[0] if rows else None
        old = next((row for row in rows if row.status in {"candidate", "confirmed"}), None)
        if old is not None and old.value_json == value:
            return old
        common = dict(data, value_json=_json(value), metadata_json=_json(metadata), source=source)
        # Version numbers continue across rejected/forgotten rows so the
        # supersedes chain remains auditable after a deletion.
        version = (latest.version + 1) if latest else 1
        if old is not None and _source_rank(source) < _source_rank(old.source):
            common["status"], common["supersedes_fact_id"] = "rejected", old.fact_id
            row = FactModel(fact_id=str(uuid4()), version=version, **common)
            db.add(row)
            db.flush()
            return old
        if old is not None:
            old.status = "superseded"
            old.updated_at = datetime.now(timezone.utc)
        row = FactModel(
            fact_id=str(uuid4()),
            version=version,
            supersedes_fact_id=(old or latest).fact_id if (old or latest) else None,
            **common,
        )
        db.add(row)
        db.flush()
        return row

    def list_facts(
        self,
        namespace: str,
        namespace_id: str | UUID,
        *,
        key: str | None = None,
        include_inactive: bool = False,
    ) -> list[FactModel]:
        with self._scope() as db:
            query = select(FactModel).where(
                FactModel.namespace == namespace, FactModel.namespace_id == _id(namespace_id)
            )
            if key is not None:
                query = query.where(FactModel.key == key)
            if not include_inactive:
                now = datetime.now(timezone.utc)
                query = query.where(
                    FactModel.status.in_(["candidate", "confirmed"]),
                    (FactModel.expires_at.is_(None)) | (FactModel.expires_at > now),
                )
            return list(db.scalars(query.order_by(desc(FactModel.version))))

    def facts_as_of(self, task_id: UUID | str, source_seq: int) -> list[FactModel]:
        if source_seq < 0:
            raise ValueError("source_seq must be non-negative")
        rows = self.list_facts("task", task_id, include_inactive=True)
        events = {str(row.event_id): row.seq for row in self.list_events(task_id)}
        return _facts_at(rows, events, source_seq)

    def delete_fact(
        self, namespace: str, namespace_id: str | UUID, key: str, *, status: str = "rejected"
    ) -> bool:
        with self._scope() as db:
            row = db.scalar(
                select(FactModel)
                .where(
                    FactModel.namespace == namespace,
                    FactModel.namespace_id == _id(namespace_id),
                    FactModel.key == key,
                    FactModel.status.in_(["candidate", "confirmed"]),
                )
                .order_by(desc(FactModel.version))
            )
            if row is None:
                return False
            # Only task facts have a task foreign key. Session/user (and
            # memory) facts still get an audit event, but their namespace id
            # must not be passed as a task id.
            task_id = _id(namespace_id) if namespace == "task" else None
            event_kwargs: dict[str, Any] = {}
            if namespace == "user":
                event_kwargs["user_id"] = _id(namespace_id)
            elif namespace == "session":
                event_kwargs["session_id"] = _id(namespace_id)
            event = self._append_event(
                db,
                task_id,
                "system",
                "FACT_DELETED",
                {
                    "namespace": namespace,
                    "namespace_id": _id(namespace_id),
                    "key": key,
                    "status": status,
                },
                **event_kwargs,
            )
            metadata = dict(row.metadata_json or {})
            metadata["_tombstone_event_id"] = event.event_id
            row.metadata_json = metadata
            row.status, row.updated_at = status, datetime.now(timezone.utc)
            db.flush()
            return True

    remove_fact = delete_fact

    def _tombstone_fact(
        self, db: Session, fact: FactModel, status: str, *, task_id: UUID | str | None = None
    ) -> None:
        """Attach an immutable deletion marker to a versioned fact."""
        namespace, namespace_id = fact.namespace, fact.namespace_id
        event_task = _id(task_id) if namespace == "task" else None
        kwargs: dict[str, Any] = {}
        if namespace == "user":
            kwargs["user_id"] = namespace_id
        elif namespace == "session":
            kwargs["session_id"] = namespace_id
        event = self._append_event(
            db,
            event_task,
            "system",
            "FACT_DELETED",
            {
                "namespace": namespace,
                "namespace_id": namespace_id,
                "key": fact.key,
                "status": status,
            },
            **kwargs,
        )
        metadata = dict(fact.metadata_json or {})
        metadata["_tombstone_event_id"] = event.event_id
        fact.metadata_json, fact.updated_at = metadata, datetime.now(timezone.utc)

    def confirm_profile_fact(
        self,
        user_id: str,
        key: str,
        value: Any = None,
        *,
        fact_id: str | None = None,
        task_id: UUID | str | None = None,
        session_id: str | None = None,
        event_id: str | None = None,
        event_payload: Any | None = None,
    ) -> UserProfileFactModel:
        with self._scope() as db:
            task_row = db.get(TaskModel, _id(task_id)) if task_id is not None else None
            if task_id is not None and (
                task_row is None or task_row.user_id not in {None, user_id}
            ):
                raise PermissionError("task belongs to another user")
            if (
                task_row is not None
                and session_id is not None
                and task_row.session_id not in {None, session_id}
            ):
                raise PermissionError("task belongs to another session")
            fact = db.get(FactModel, fact_id) if fact_id else None
            if fact_id and fact is None:
                raise ValueError("fact was not found")
            if fact is not None:
                if fact.key != key:
                    raise ValueError("fact key does not match profile key")
                if fact.status not in {"candidate", "confirmed"}:
                    raise ValueError("fact is no longer eligible for confirmation")
                if fact.namespace == "user" and fact.namespace_id != user_id:
                    raise PermissionError("fact belongs to another user")
                if fact.namespace == "task":
                    if task_id is None or fact.namespace_id != _id(task_id):
                        raise PermissionError("task fact does not belong to this task")
                    if task_row is None:
                        raise PermissionError("task belongs to another user")
                if fact.namespace not in {"user", "task"}:
                    raise PermissionError("only user or task facts may become profile facts")
                value = fact.value_json
            if value is None:
                raise ValueError("profile value is required")
            row = db.scalar(
                select(UserProfileFactModel).where(
                    UserProfileFactModel.user_id == user_id, UserProfileFactModel.key == key
                )
            )
            if event_id is not None:
                # Reusing an event from another task/user would break the
                # audit trail. Validate ownership and scope before allowing
                # the idempotent fast path below.
                event = db.get(RunEventModel, event_id)
                if event is None or str(event.event_type).lower() not in {
                    "profile_confirmation",
                    "confirmation",
                }:
                    raise ValueError("confirmation event was not found")
                if event.user_id != user_id:
                    owner = db.get(TaskModel, event.task_id) if event.task_id else None
                    if event.user_id is not None or owner is None or owner.user_id != user_id:
                        raise PermissionError("confirmation event belongs to another user")
                if task_id is not None and event.task_id != _id(task_id):
                    raise PermissionError("confirmation event belongs to another task")
                if session_id is not None and event.session_id != session_id:
                    raise PermissionError("confirmation event belongs to another session")
            if (
                row is not None
                and row.status == "confirmed"
                and row.value_json == _json(value)
                and (fact_id is None or row.source_fact_id == fact_id)
            ):
                return row
            if event_id is None:
                event_id = self._append_event(
                    db,
                    task_id,
                    "user",
                    "profile_confirmation",
                    event_payload or {"key": key, "value": value, "confirmed": True},
                    session_id=session_id,
                    user_id=user_id,
                ).event_id
            if row is None:
                row = UserProfileFactModel(
                    profile_fact_id=str(uuid4()),
                    user_id=user_id,
                    key=key,
                    value_json=_json(value),
                    source_fact_id=fact_id,
                    confirmation_event_id=event_id,
                    importance=getattr(fact, "importance", 0.5),
                    confidence=getattr(fact, "confidence", 1.0),
                )
                db.add(row)
            else:
                previous = row.source_fact_id
                row.value_json, row.status, row.version = _json(value), "confirmed", row.version + 1
                row.source_fact_id, row.confirmation_event_id, row.updated_at = (
                    fact_id,
                    event_id,
                    datetime.now(timezone.utc),
                )
                if (
                    previous
                    and previous != fact_id
                    and (old := db.get(FactModel, previous)) is not None
                ):
                    old.status = "superseded"
            if fact is not None:
                fact.status, fact.confirmed_event_id = "confirmed", event_id
            db.flush()
            return row

    confirm_user_profile_fact = confirm_profile_fact

    def reject_profile_fact(
        self,
        user_id: str,
        key: str,
        *,
        fact_id: str | None = None,
        task_id: UUID | str | None = None,
        session_id: str | None = None,
    ) -> None:
        with self._scope() as db:
            task = db.get(TaskModel, _id(task_id)) if task_id is not None else None
            if task_id is not None and (task is None or task.user_id not in {None, user_id}):
                raise PermissionError("task belongs to another user")
            if (
                task is not None
                and session_id is not None
                and task.session_id not in {None, session_id}
            ):
                raise PermissionError("task belongs to another session")
            fact = db.get(FactModel, fact_id) if fact_id else None
            if fact_id and fact is None:
                raise ValueError("fact was not found")
            if fact is not None:
                if fact.key != key:
                    raise ValueError("fact key does not match profile key")
                if fact.namespace == "user" and fact.namespace_id != user_id:
                    raise PermissionError("fact belongs to another user")
                if fact.namespace == "task" and (
                    task_id is None or fact.namespace_id != _id(task_id)
                ):
                    raise PermissionError("task fact does not belong to this task")
                if fact.namespace not in {"user", "task"}:
                    raise PermissionError("only user or task facts may be rejected")
                fact.status = "rejected"
                self._tombstone_fact(db, fact, "rejected", task_id=task_id)
            row = db.scalar(
                select(UserProfileFactModel).where(
                    UserProfileFactModel.user_id == user_id, UserProfileFactModel.key == key
                )
            )
            if row is not None and (fact_id is None or row.source_fact_id in {None, fact_id}):
                row.status = "rejected"
            self._append_event(
                db,
                task_id,
                "user",
                "profile_rejection",
                {"key": key},
                session_id=session_id,
                user_id=user_id,
            )

    def forget_profile_fact(
        self,
        user_id: str,
        key: str,
        *,
        fact_id: str | None = None,
        task_id: UUID | str | None = None,
        session_id: str | None = None,
    ) -> None:
        with self._scope() as db:
            task = db.get(TaskModel, _id(task_id)) if task_id is not None else None
            if task_id is not None and (task is None or task.user_id not in {None, user_id}):
                raise PermissionError("task belongs to another user")
            if (
                task is not None
                and session_id is not None
                and task.session_id not in {None, session_id}
            ):
                raise PermissionError("task belongs to another session")
            fact = db.get(FactModel, fact_id) if fact_id else None
            if fact_id and fact is None:
                raise ValueError("fact was not found")
            if fact is None and task_id is not None:
                fact = db.scalar(
                    select(FactModel)
                    .where(
                        FactModel.namespace == "task",
                        FactModel.namespace_id == _id(task_id),
                        FactModel.key == key,
                        FactModel.status.in_(["candidate", "confirmed"]),
                    )
                    .order_by(desc(FactModel.version))
                )
            if fact is not None:
                if fact.key != key:
                    raise ValueError("fact key does not match profile key")
                if fact.namespace == "user" and fact.namespace_id != user_id:
                    raise PermissionError("fact belongs to another user")
                if fact.namespace == "task" and (
                    task_id is None or fact.namespace_id != _id(task_id)
                ):
                    raise PermissionError("task fact does not belong to this task")
                if fact.namespace not in {"user", "task"}:
                    raise PermissionError("only user or task facts may be forgotten")
            row = db.scalar(
                select(UserProfileFactModel).where(
                    UserProfileFactModel.user_id == user_id, UserProfileFactModel.key == key
                )
            )
            if row is not None:
                row.status = "forgotten"
                if fact is None and row.source_fact_id:
                    fact = db.get(FactModel, row.source_fact_id)
            if fact is not None and fact.status in {"candidate", "confirmed"}:
                fact.status = "forgotten"
                self._tombstone_fact(db, fact, "forgotten", task_id=task_id)
            self._append_event(
                db,
                task_id,
                "user",
                "profile_forget",
                {"key": key, "fact_id": fact.fact_id if fact else fact_id},
                session_id=session_id,
                user_id=user_id,
            )

    def list_profile_facts(
        self, user_id: str, *, include_inactive: bool = False
    ) -> list[UserProfileFactModel]:
        with self._scope() as db:
            query = select(UserProfileFactModel).where(UserProfileFactModel.user_id == user_id)
            if not include_inactive:
                now = datetime.now(timezone.utc)
                query = query.where(
                    UserProfileFactModel.status == "confirmed",
                    (UserProfileFactModel.expires_at.is_(None))
                    | (UserProfileFactModel.expires_at > now),
                )
            return list(db.scalars(query.order_by(desc(UserProfileFactModel.updated_at))))

    def recall_memory(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        task_id: UUID | str | None = None,
        query: str = "",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        terms = _query_terms(query)
        candidates: list[tuple[float, dict[str, Any]]] = []
        namespaces = [("session", session_id)] if session_id else []
        if task_id:
            task = self.get_task_model(task_id)
            if task is None:
                # A dangling task id must never expose facts left under that
                # namespace.  Keep profile/session recall available.
                task_id = None
            elif task.user_id != user_id:
                raise PermissionError("task belongs to another user")
            elif session_id is not None and session_id not in {
                task.session_id,
                f"{task.user_id}:{task.session_id}",
            }:
                raise PermissionError("task belongs to another session")
            if task_id:
                namespaces.append(("task", _id(task_id)))
        for namespace, namespace_id in namespaces:
            for fact in self.list_facts(namespace, namespace_id):
                text = f"{fact.key} {fact.value_json}".lower()
                overlap = sum(term in text for term in terms)
                if terms and not overlap:
                    continue
                score = fact.importance + 0.1 * overlap
                candidates.append(
                    (
                        score,
                        {
                            "memory_id": fact.fact_id,
                            "namespace": namespace,
                            "namespace_id": namespace_id,
                            "key": fact.key,
                            "value": fact.value_json,
                            "source": "fact",
                            "score": score,
                        },
                    )
                )
        for profile in self.list_profile_facts(user_id):
            text = f"{profile.key} {profile.value_json}".lower()
            overlap = sum(term in text for term in terms)
            if terms and not overlap:
                continue
            score = profile.importance + 0.1 * overlap
            candidates.append(
                (
                    score,
                    {
                        "memory_id": profile.profile_fact_id,
                        "namespace": "user",
                        "namespace_id": user_id,
                        "key": profile.key,
                        "value": profile.value_json,
                        "source": "profile",
                        "score": score,
                    },
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]["source"] == "profile"), reverse=True)
        result, seen = [], set()
        for _, item in candidates:
            identity = (item["namespace"], item["key"])
            if identity in seen:
                continue
            seen.add(identity)
            result.append(item)
            if len(result) == top_k:
                break
        return result

    lookup_memory = recall_memory

    def save_snapshot(self, snapshot: Any) -> ContextSnapshotModel:
        raw = _json(snapshot)
        data = dict(raw) if isinstance(raw, dict) else dict(vars(snapshot))
        data.setdefault("snapshot_id", str(uuid4()))
        data["task_id"] = _id(data["task_id"])
        defaults = {
            "recent_event_ids": [],
            "summary_ids": [],
            "active_fact_ids": [],
            "facts": {},
            "evidence": [],
            "recent_messages": [],
            "memory_refs": [],
            "state": {},
            "stats": {},
            "compression_stats": {},
            "status": "RECEIVED",
            "summary": "",
            "rendered_context": "",
            "version": 1,
            "source_seq": 0,
            "unit_count": 0,
        }
        for key, value in defaults.items():
            data.setdefault(key, value)
        if "created_at" in data:
            data["created_at"] = _as_datetime(data["created_at"])
        allowed = set(ContextSnapshotModel.__table__.columns.keys())
        with self._scope() as db:
            incoming_version, incoming_seq = int(data["version"]), int(data["source_seq"] or 0)
            latest = db.scalar(
                select(ContextSnapshotModel)
                .where(ContextSnapshotModel.task_id == data["task_id"])
                .order_by(desc(ContextSnapshotModel.version), desc(ContextSnapshotModel.source_seq))
                .limit(1)
            )
            if latest is not None and (
                incoming_version < latest.version
                or incoming_version == latest.version
                and incoming_seq < latest.source_seq
            ):
                raise ValueError("cannot overwrite a newer context snapshot")
            row = db.get(ContextSnapshotModel, data["snapshot_id"])
            if row is not None and row.task_id != data["task_id"]:
                raise ValueError("snapshot belongs to another task")
            if row is None:
                row = ContextSnapshotModel(
                    **{key: value for key, value in data.items() if key in allowed}
                )
                db.add(row)
            else:
                for key, value in data.items():
                    if key in allowed and key != "snapshot_id":
                        setattr(row, key, value)
            db.flush()
            return row

    save_context_snapshot = save_snapshot

    def _save_task_snapshot(self, db: Session, context: TaskContext) -> ContextSnapshotModel:
        task_key, data = _id(context.task_id), _json(context)
        all_events = list(
            db.scalars(
                select(RunEventModel)
                .where(RunEventModel.task_id == task_key)
                .order_by(RunEventModel.seq)
            )
        )
        seq = int(data.get("last_event_seq", 0) or 0)
        rows = list(
            db.scalars(
                select(FactModel).where(
                    FactModel.namespace == "task", FactModel.namespace_id == task_key
                )
            )
        )
        active = {
            row.key: row
            for row in _facts_at(rows, {str(item.event_id): item.seq for item in all_events}, seq)
        }
        version = max(1, int(data.get("checkpoint_version", 1)))
        newest = db.scalar(
            select(ContextSnapshotModel)
            .where(ContextSnapshotModel.task_id == task_key)
            .order_by(desc(ContextSnapshotModel.version), desc(ContextSnapshotModel.source_seq))
            .limit(1)
        )
        if newest is not None and (
            version < newest.version or version == newest.version and seq < newest.source_seq
        ):
            raise ValueError("cannot persist an older task checkpoint")
        latest = db.scalar(
            select(ContextSnapshotModel)
            .where(
                ContextSnapshotModel.task_id == task_key, ContextSnapshotModel.version == version
            )
            .order_by(desc(ContextSnapshotModel.created_at))
            .limit(1)
        )
        if latest is None:
            latest = ContextSnapshotModel(
                snapshot_id=str(uuid4()), task_id=task_key, version=version
            )
            db.add(latest)
        # Use the same compiler as the live Harness so a durable checkpoint
        # and a replayed checkpoint have one canonical representation.
        from civicnexus.runtime.context_manager import ContextManager

        data["extracted"] = {key: row.value_json for key, row in active.items()} if seq else {}
        data["active_fact_ids"] = [row.fact_id for row in active.values()]
        data["source_seq"], data["version"] = seq, version
        snapshot = ContextManager(max_chars=3000).compact(
            data,
            messages=data.get("messages") or [],
            event_ids=[str(item.event_id) for item in all_events],
            memory_refs=[
                str(item.get("memory_id", item)) if isinstance(item, dict) else str(item)
                for item in (data.get("memory_hits") or [])
            ],
        )
        values = _json(snapshot)
        for key in ContextSnapshotModel.__table__.columns.keys():
            if key != "snapshot_id" and key in values:
                # Pydantic's JSON dump serialises datetimes as ISO strings,
                # while SQLAlchemy's DateTime columns require datetime
                # objects (notably on SQLite).  Convert only this column
                # back at the persistence boundary.
                setattr(
                    latest, key, _as_datetime(values[key]) if key == "created_at" else values[key]
                )
        db.flush()
        return latest

    def get_latest_snapshot(self, task_id: UUID | str) -> ContextSnapshotModel | None:
        with self._scope() as db:
            return _normalise_snapshot(
                db.scalar(
                    select(ContextSnapshotModel)
                    .where(ContextSnapshotModel.task_id == _id(task_id))
                    .order_by(
                        desc(ContextSnapshotModel.source_seq),
                        desc(ContextSnapshotModel.version),
                        desc(ContextSnapshotModel.created_at),
                    )
                    .limit(1)
                )
            )

    def load_context_snapshot(self, task_id: UUID | str) -> Any | None:
        row = self.get_latest_snapshot(task_id)
        if row is None:
            return None
        from civicnexus.memory.models import ContextSnapshot

        return ContextSnapshot.model_validate(
            {
                column.name: getattr(row, column.name)
                for column in ContextSnapshotModel.__table__.columns
                if column.name in ContextSnapshot.model_fields
            }
        )

    def delete_snapshots(self, task_id: UUID | str) -> int:
        with self._scope() as db:
            rows = list(
                db.scalars(
                    select(ContextSnapshotModel).where(ContextSnapshotModel.task_id == _id(task_id))
                )
            )
            for row in rows:
                db.delete(row)
            return len(rows)

    def rebuild_snapshot(
        self, task_id: UUID | str, source_seq: int | None = None
    ) -> ContextSnapshotModel:
        task_key = _id(task_id)
        if source_seq is not None and source_seq < 0:
            raise ValueError("source_seq must be non-negative")
        with self._scope() as db:
            task = db.get(TaskModel, task_key)
            if task is None:
                raise KeyError(f"task not found: {task_key}")
            all_event_rows = list(
                db.scalars(
                    select(RunEventModel)
                    .where(RunEventModel.task_id == task_key)
                    .order_by(RunEventModel.seq)
                )
            )
            query = select(RunEventModel).where(RunEventModel.task_id == task_key)
            if source_seq is not None:
                query = query.where(RunEventModel.seq <= source_seq)
            events = list(db.scalars(query.order_by(RunEventModel.seq)))
            seq = (
                min(source_seq, all_event_rows[-1].seq)
                if source_seq is not None and all_event_rows
                else (source_seq or 0)
                if source_seq is not None
                else (events[-1].seq if events else 0)
            )
            rows = list(
                db.scalars(
                    select(FactModel).where(
                        FactModel.namespace == "task", FactModel.namespace_id == task_key
                    )
                )
            )
            facts = (
                {}
                if seq == 0
                else {
                    row.key: row
                    for row in _facts_at(
                        rows, {str(event.event_id): event.seq for event in all_event_rows}, seq
                    )
                }
            )
            base = dict(task.context_json or {})
            replay_base = base if source_seq is None else {}
            default_status = task.status if source_seq is None else "RECEIVED"
            status, messages, state, summary, evidence, memory_refs, version = _replay_events(
                events, replay_base, default_status
            )
            fact_values = {key: row.value_json for key, row in facts.items()}
            candidate = state.get("pending_profile_fact")
            if isinstance(candidate, dict) and candidate.get("key") in facts:
                # A later correction may version the fact without another
                # MEMORY_WRITE event.  Keep the replayed profile prompt
                # aligned with the effective fact at this checkpoint.
                fact = facts[candidate["key"]]
                candidate["value"] = fact.value_json
                candidate["fact_id"] = fact.fact_id
            roles = state.get("message_roles") or []
            replay_messages = [
                {
                    "role": roles[index]
                    if index < len(roles)
                    else ("user" if index % 2 == 0 else "assistant"),
                    "content": message,
                }
                for index, message in enumerate(messages)
            ]
            data = {
                "task_id": task_key,
                "user_id": task.user_id or "anonymous",
                "session_id": task.session_id or "default",
                "user_message": next(
                    (
                        message
                        for message, role in reversed(list(zip(messages, roles, strict=False)))
                        if role == "user"
                    ),
                    messages[-1] if messages else task.user_message,
                ),
                "status": status,
                "messages": replay_messages[-6:],
                "extracted": fact_values,
                "active_fact_ids": [row.fact_id for row in facts.values()],
                "evidence": evidence,
                "memory_refs": memory_refs,
                "last_event_seq": seq,
                "checkpoint_version": version,
                "checkpoint": f"{task_key}:{version}" if version else None,
                "state": state,
                "version": version,
                "source_seq": seq,
                "event_ids": [str(event.event_id) for event in events],
            }
            from civicnexus.runtime.context_manager import ContextManager

            snapshot = ContextManager(max_chars=3000).compact(
                data,
                messages=replay_messages,
                memory_refs=memory_refs,
                event_ids=[str(event.event_id) for event in events],
            )
            values = _json(snapshot)
            values["task_id"] = task_key
            values["snapshot_id"] = str(uuid4())
            if "created_at" in values:
                values["created_at"] = _as_datetime(values["created_at"])
            snapshot = ContextSnapshotModel(
                **{
                    key: value
                    for key, value in values.items()
                    if key in ContextSnapshotModel.__table__.columns.keys()
                }
            )
            db.add(snapshot)
            db.flush()
            return snapshot

    rebuild_context_snapshot = rebuild_snapshot

    def create_local_case(
        self, task_id: UUID | str, payload: dict[str, Any], *, idempotency_key: str | None = None
    ) -> LocalCaseModel:
        key = idempotency_key or f"{_id(task_id)}:create_local_case"
        with self._scope() as db:
            row = db.scalar(select(LocalCaseModel).where(LocalCaseModel.idempotency_key == key))
            if row is not None:
                return row
            row = LocalCaseModel(
                case_id=str(uuid4()),
                task_id=_id(task_id),
                idempotency_key=key,
                payload_json=_json(payload),
            )
            db.add(row)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                row = db.scalar(select(LocalCaseModel).where(LocalCaseModel.idempotency_key == key))
                if row is None:
                    raise
            return row

    get_or_create_local_case = create_local_case

    def save_evaluation_report(
        self, dataset_name: str, metrics: dict[str, Any], *, run_id: str | None = None
    ) -> EvaluationReportModel:
        with self._scope() as db:
            row = EvaluationReportModel(
                report_id=str(uuid4()),
                dataset_name=dataset_name,
                metrics_json=_json(metrics),
                run_id=run_id,
            )
            db.add(row)
            db.flush()
            return row

    def latest_evaluation_report(
        self, dataset_name: str | None = None
    ) -> EvaluationReportModel | None:
        with self._scope() as db:
            query = select(EvaluationReportModel)
            if dataset_name:
                query = query.where(EvaluationReportModel.dataset_name == dataset_name)
            return db.scalar(query.order_by(desc(EvaluationReportModel.created_at)).limit(1))


SqlAlchemyRepository = TaskRepository
__all__ = ["TaskRepository", "SqlAlchemyRepository"]
