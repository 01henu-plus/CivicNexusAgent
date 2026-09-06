from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any, Callable

from civicnexus.agents.case_analysis_agent import CaseAnalysisAgent
from civicnexus.agents.decision import DecisionEngine
from civicnexus.agents.intake_agent import IntakeAgent
from civicnexus.agents.review_agent import ReviewAgent
from civicnexus.agents.routing_agent import RoutingAgent
from civicnexus.core.config import Settings, get_settings
from civicnexus.domain.context import TaskContext
from civicnexus.evaluation import EvaluationRunner, runtime_metrics
from civicnexus.llm import OpenAICompatibleClient
from civicnexus.persistence.adapters import SqlEventLog, SqlFactLedger, SqlLayeredMemory
from civicnexus.persistence.cache import MemoryCache, RedisCheckpointStore
from civicnexus.persistence.database import create_db_engine, create_session_factory, init_db
from civicnexus.persistence.repositories import TaskRepository
from civicnexus.runtime.context_manager import ContextManager
from civicnexus.runtime.engine import AgentRuntime, RunBudget, RunTrace
from civicnexus.skills import SkillRegistry
from civicnexus.tools.builtins import build_tool_registry


_PBKDF2_ITERATIONS = 300_000


def hash_password(password: str) -> str:
    """Hash a password with a salted PBKDF2-SHA256 record."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt_text}${digest_text}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify a PBKDF2 record; malformed records simply fail verification."""
    try:
        algorithm, rounds, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(rounds)
        if iterations <= 0:
            return False
        padding = "=" * (-len(salt_text) % 4)
        salt = base64.urlsafe_b64decode(salt_text + padding)
        padding = "=" * (-len(digest_text) % 4)
        expected = base64.urlsafe_b64decode(digest_text + padding)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


class AuthTokenStore:
    """Small opaque-token store backed by the activity cache."""

    def __init__(self, cache: Any, ttl: int, *, prefix: str) -> None:
        self.cache = cache
        self.ttl = ttl
        self.prefix = prefix.rstrip(":")
        self._local: dict[str, float] = {}
        self._local_payloads: dict[str, dict[str, str]] = {}

    def issue(self, *, role: str, user_id: str, display_name: str) -> str:
        token = secrets.token_urlsafe(32)
        expires = time.time() + self.ttl
        payload = {"role": role, "user_id": user_id, "display_name": display_name}
        if hasattr(self.cache, "client"):
            self.cache.client.set(
                f"{self.prefix}:{token}", json.dumps(payload, ensure_ascii=False), ex=self.ttl
            )
        else:
            self._local[token] = expires
            self._local_payloads[token] = payload
        return token

    def resolve(self, token: str) -> dict[str, str] | None:
        if hasattr(self.cache, "client"):
            raw = self.cache.client.get(f"{self.prefix}:{token}")
            if not raw:
                return None
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None
        expires = self._local.get(token)
        if not expires or expires <= time.time():
            self._local.pop(token, None)
            self._local_payloads.pop(token, None)
            return None
        return self._local_payloads.get(token)


class AdminTokenStore:
    def __init__(self, cache: Any, ttl: int) -> None:
        self._store = AuthTokenStore(cache, ttl, prefix="civicnexus:admin")

    def issue(self) -> str:
        return self._store.issue(role="admin", user_id="admin", display_name="管理员")

    def resolve(self, token: str) -> dict[str, str] | None:
        principal = self._store.resolve(token)
        return principal if principal and principal.get("role") == "admin" else None

    def valid(self, token: str) -> bool:
        return self.resolve(token) is not None


class TaskService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        db_engine = create_db_engine(self.settings.database_url)
        init_db(db_engine)
        factory = create_session_factory(db_engine)
        self.repository = TaskRepository(factory)
        self.checkpoint = self._checkpoint_store()
        self.tokens = AdminTokenStore(self.checkpoint, self.settings.admin_token_ttl_seconds)
        self.user_tokens = AuthTokenStore(
            self.checkpoint, self.settings.user_token_ttl_seconds, prefix="civicnexus:user"
        )
        self.skills = SkillRegistry(Path(self.settings.skill_directory))
        self.evaluator = EvaluationRunner()
        self.memory = SqlLayeredMemory(self.repository)
        self.events = SqlEventLog(self.repository)
        self.facts = SqlFactLedger(self.repository)
        self.context_manager = ContextManager(
            max_chars=self.settings.context_max_units,
            max_messages=self.settings.context_recent_messages,
            snapshot_store=self.repository,
        )
        llm = OpenAICompatibleClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_tokens=self.settings.llm_max_tokens,
        )
        decisions = DecisionEngine(llm)

        def lookup(context, query, limit):
            return [
                item.model_dump(mode="json")
                for item in self.memory.search(
                    f"session:{context.user_id}:{context.session_id}", query, limit=limit
                )
            ]

        def create_case(context, args):
            row = self.repository.create_local_case(
                context.task_id,
                args.model_dump(),
                idempotency_key=args.idempotency_key,
            )
            return {"case_id": row.case_id, "dry_run": True}

        self.tools = build_tool_registry(memory_lookup=lookup, create_case=create_case)
        self.runtime = AgentRuntime(
            [
                IntakeAgent(decisions),
                CaseAnalysisAgent(decisions),
                RoutingAgent(decisions),
                ReviewAgent(decisions, max_rounds=self.settings.max_review_rounds),
            ],
            tools=self.tools,
            skills=self.skills,
            context_manager=self.context_manager,
            event_log=self.events,
            fact_ledger=self.facts,
            memory=self.memory,
            budget=RunBudget(max_review_rounds=self.settings.max_review_rounds),
            task_store=self.repository,
            checkpoint_store=self.checkpoint,
            checkpoint_ttl=self.settings.task_cache_ttl_seconds,
        )

    def _checkpoint_store(self):
        if self.settings.use_local_storage:
            return MemoryCache()
        return RedisCheckpointStore(self.settings.redis_url)

    def create_task(
        self,
        user_id: str,
        session_id: str,
        message: str,
        *,
        on_trace: Callable[[RunTrace], None] | None = None,
    ) -> TaskContext:
        return self.runtime.start(
            user_id=user_id, session_id=session_id, message=message, on_trace=on_trace
        )

    def issue_user_token(
        self,
        *,
        username: str | None = None,
        user_id: str | None = None,
        display_name: str | None = None,
    ) -> dict[str, str]:
        token = self.user_tokens.issue(
            role="user",
            user_id=user_id or self.settings.user_id,
            display_name=display_name or self.settings.user_display_name,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": "user",
            "user_id": user_id or self.settings.user_id,
            "display_name": display_name or self.settings.user_display_name,
            "username": username or self.settings.user_username,
        }

    def issue_admin_token(self) -> dict[str, str]:
        return {
            "access_token": self.tokens.issue(),
            "token_type": "bearer",
            "role": "admin",
            "user_id": "admin",
            "username": self.settings.admin_username,
            "display_name": "管理员",
        }

    def authenticate(self, username: str, password: str) -> dict[str, str] | None:
        """Issue one role-scoped token for the shared login form."""
        username = username.strip()
        username_key = username.lower()
        if (
            username_key == self.settings.admin_username.strip().lower()
            and password == self.settings.admin_password.get_secret_value()
        ):
            return self.issue_admin_token()
        if (
            username_key == self.settings.user_username.strip().lower()
            and password == self.settings.user_password.get_secret_value()
        ):
            return self.issue_user_token(username=self.settings.user_username)
        account = self.repository.get_user_account(username)
        if (
            account
            and account.status == "active"
            and verify_password(password, account.password_hash)
        ):
            return self.issue_user_token(
                username=account.username,
                user_id=account.user_id,
                display_name=account.display_name,
            )
        return None

    def register_user(
        self,
        username: str,
        password: str,
        display_name: str | None = None,
    ) -> dict[str, str]:
        """Create an ordinary account and issue its first session token."""
        username = username.strip().lower()
        if username in {
            self.settings.admin_username.strip().lower(),
            self.settings.user_username.strip().lower(),
        }:
            raise ValueError("username already exists")
        row = self.repository.create_user_account(
            username,
            hash_password(password),
            display_name=display_name,
        )
        return self.issue_user_token(
            username=row.username,
            user_id=row.user_id,
            display_name=row.display_name,
        )

    def resolve_token(self, token: str) -> dict[str, str] | None:
        return self.user_tokens.resolve(token) or self.tokens.resolve(token)

    def add_message(
        self,
        task_id: str,
        user_id: str,
        session_id: str,
        message: str,
        *,
        on_trace: Callable[[RunTrace], None] | None = None,
    ) -> TaskContext:
        context = self.load(task_id)
        if context.user_id != user_id or context.session_id != session_id:
            raise PermissionError("task does not belong to this session")
        return self.runtime.add_message(context, message, on_trace=on_trace)

    def load(self, task_id: str) -> TaskContext:
        cached = self.checkpoint.get_context(task_id)
        context = cached if isinstance(cached, TaskContext) else self.repository.get_task(task_id)
        if context is None:
            raise KeyError(task_id)
        # Redis is an activity cache. A cold/missing cache is rebuilt from the
        # SQL task/event/fact source and then repopulated for the next request.
        if not isinstance(cached, TaskContext) and hasattr(self.repository, "rebuild_snapshot"):
            self.repository.rebuild_snapshot(task_id)
            loader = getattr(self.repository, "load_context_snapshot", None)
            snapshot = loader(task_id) if loader else None
            if snapshot is not None:
                context = self.context_manager.restore(snapshot, context)
        for key, fact in self.facts.all(context.task_id).items():
            context.extracted[key] = fact.value
        if not isinstance(cached, TaskContext):
            self.context_manager.save_checkpoint(context, messages=context.messages)
            self.checkpoint.save_context(
                task_id, context, ttl_seconds=self.settings.task_cache_ttl_seconds
            )
        return context

    def user_view(self, context: TaskContext) -> dict[str, Any]:
        # The public contract is a chat transcript only.  Lifecycle state,
        # facts, memory and trace remain administrator-only.
        return {
            "task_id": str(context.task_id),
            "messages": [item.model_dump(mode="json") for item in context.messages],
            "reply": context.reply,
            "assistant_message": context.reply,
        }

    def admin_detail(self, task_id: str) -> dict[str, Any]:
        context = self.load(task_id)
        raw_events = [_row_dict(item) for item in self.events.list(task_id)]
        events = [_admin_trace_event(item) for item in raw_events]
        facts = [
            _row_dict(item)
            for item in self.repository.list_facts("task", task_id, include_inactive=True)
        ]
        profiles = [
            _row_dict(item)
            for item in self.repository.list_profile_facts(context.user_id, include_inactive=True)
        ]
        snapshot = self.repository.get_latest_snapshot(task_id)
        return {
            "task_id": str(context.task_id),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "user_message": context.user_message,
            "status": context.status,
            "messages": [item.model_dump(mode="json") for item in context.messages],
            "context": context.model_dump(mode="json"),
            "trace": events,
            "events": raw_events,
            "memory": context.memory_hits,
            "memories": context.memory_hits,
            "facts": facts,
            "profile_facts": profiles,
            "snapshot": _row_dict(snapshot) if snapshot else None,
            "skill": context.skill,
            "skills": [context.skill] if context.skill else [],
            "created_case_id": context.created_case_id,
        }

    def overview(self) -> dict[str, Any]:
        data = self.repository.overview()
        counts = data.get("status_counts", {})
        total = data.get("task_count", 0)
        completed = counts.get("COMPLETED", 0)
        active = total - completed - counts.get("FAILED", 0) - counts.get("CANCELLED", 0)
        return {
            "total_tasks": total,
            "active_tasks": active,
            "completed_tasks": completed,
            "failed_tasks": counts.get("FAILED", 0),
            "completion_rate": completed / total if total else 0.0,
            **data,
        }

    def task_list(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": row.id,
                "status": row.status,
                "user_message": row.user_message,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in self.repository.list_tasks()
        ]

    def metrics(self) -> dict[str, Any]:
        return runtime_metrics(self.repository)

    def evaluation_report(self) -> dict[str, Any]:
        """Run and persist the fixed local gold-set report."""
        report = self.evaluator.run()
        payload = report.model_dump(mode="json")
        self.repository.save_evaluation_report(report.dataset, payload, run_id=report.run_id)
        return payload


def _row_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")
    table = getattr(row, "__table__", None)
    if table is not None:
        return {column.name: getattr(row, column.name) for column in table.columns}
    return dict(getattr(row, "__dict__", {}))


def _admin_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    """Expose a stable UI projection while preserving raw event fields."""
    payload = event.get("payload") or event.get("payload_redacted") or {}
    payload = payload if isinstance(payload, dict) else {}
    event_type = str(event.get("event_type") or "")
    return {
        **event,
        "sequence": event.get("sequence", event.get("seq")),
        "timestamp": event.get("created_at"),
        "agent": event.get("actor") or event.get("actor_type") or "Runtime",
        "from_status": event.get("state_before")
        or payload.get("state_before")
        or payload.get("from"),
        "to_status": event.get("state_after") or payload.get("state_after") or payload.get("to"),
        "action": event_type or payload.get("action") or payload.get("name") or "状态更新",
        "tool_name": payload.get("name") if event_type == "TOOL_CALL" else None,
        "summary": payload.get("reason") or payload.get("summary") or event_type,
    }


_service: TaskService | None = None


def get_service() -> TaskService:
    global _service
    if _service is None:
        _service = TaskService()
    return _service
