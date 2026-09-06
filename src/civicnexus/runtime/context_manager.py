"""Deterministic, layered context compilation used by every Agent."""

from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any
from uuid import uuid4
from pydantic import BaseModel
from civicnexus.memory.models import ContextSnapshot, ContextStats
from civicnexus.memory.protocols import SnapshotStore, TaskId


class ContextManager:
    """Keep structured state intact while fitting conversational prose to a budget."""

    def __init__(
        self,
        *,
        max_chars: int = 3000,
        max_messages: int = 6,
        max_evidence: int = 3,
        max_summary_chars: int = 600,
        chars_per_token: int = 4,
        snapshot_store: SnapshotStore | None = None,
        max_units: int | None = None,
        recent_messages: int | None = None,
    ) -> None:
        self.max_chars = max_units if max_units is not None else max_chars
        self.max_messages = recent_messages if recent_messages is not None else max_messages
        self.max_evidence, self.max_summary_chars = (max_evidence, max_summary_chars)
        self.chars_per_token, self.snapshot_store = (chars_per_token, snapshot_store)
        if min(self.max_chars, self.max_messages, self.max_evidence, self.chars_per_token) < 1:
            raise ValueError("context limits must be positive")
        if self.max_summary_chars < 0:
            raise ValueError("summary limit must be non-negative")

    def compact(
        self,
        context: BaseModel | Mapping[str, Any] | Any,
        *,
        messages: Sequence[Any] | None = None,
        summary: str | None = None,
        memory_refs: Sequence[str] | None = None,
        event_ids: Sequence[str] | None = None,
    ) -> ContextSnapshot:
        data = _mapping(context)
        task_id = str(data.get("task_id") or uuid4())
        status = _value(data.get("status"), "RECEIVED")
        version = _int(data.get("version", data.get("checkpoint_version")), 1, 1)
        source_seq = _int(data.get("source_seq", data.get("last_event_seq")), 0, 0)
        facts = _facts(data)
        evidence = _dicts(data.get("evidence"))[: self.max_evidence]
        all_messages = _messages(data, messages)
        recent = all_messages[-self.max_messages :]
        old = all_messages[: -len(recent)] if recent else all_messages
        segments = _segments(old)
        supplied_summary = summary if summary is not None else data.get("summary")
        summary_text = _value(supplied_summary, "")
        if not summary_text and segments:
            summary_text = "；".join(segments)
        summary_text = _clip(summary_text, self.max_summary_chars)
        refs = _strings(memory_refs if memory_refs is not None else data.get("memory_refs"))[:20]
        state = _state(data)
        correction = _latest_correction(all_messages)
        if correction:
            state["last_correction"] = correction
        missing = _strings(data.get("missing_fields"))
        if missing:
            state["unresolved_fields"] = missing
        pending = _dicts(data.get("pending_tools"))
        if pending:
            state["pending_tools"] = pending
        all_event_ids = _strings(event_ids if event_ids is not None else data.get("event_ids"))
        # An explicit event list is authoritative; do not carry an older
        # recent window into a newer checkpoint.
        recent_ids = _strings(data.get("recent_event_ids"))
        if all_event_ids and recent:
            recent_ids = all_event_ids[-len(recent) :]
        summary_ids = (
            [_digest(item) for item in segments] if segments else _strings(data.get("summary_ids"))
        )
        if summary_text and not summary_ids:
            summary_ids = [_digest(summary_text)]
        active_ids = _strings(data.get("active_fact_ids")) or _fact_ids(data)
        roles = [item.role for item in _chat_messages(data, messages)]
        if roles:
            state["message_roles"] = roles[-self.max_messages :]
        raw = _json(
            {
                "status": status,
                "facts": facts,
                "evidence": evidence,
                "messages": all_messages,
                "summary_segments": segments,
                "state": state,
            }
        )
        payload = {
            "status": status,
            "latest_message": recent[-1] if recent else all_messages[-1] if all_messages else "",
            "summary": summary_text,
            "summary_segments": segments,
            "facts": facts,
            "evidence": [_evidence(item) for item in evidence],
            "recent_messages": recent,
            "memory_refs": refs,
            "missing_fields": missing,
            "state": state,
        }
        rendered = _fit(payload, self.max_chars)
        raw_units, compiled_units = (len(raw), len(rendered))
        reduction = round(max(0.0, 1 - compiled_units / raw_units), 6) if raw_units else 0.0
        if not all_event_ids or segments or summary_text:
            coverage = 1.0
        else:
            coverage = min(1.0, len(set(recent_ids)) / len(all_event_ids))
        stats = ContextStats(
            original_chars=raw_units,
            compacted_chars=compiled_units,
            estimated_tokens=ceil(compiled_units / self.chars_per_token),
            reduction_ratio=reduction,
            raw_units=raw_units,
            compiled_units=compiled_units,
            compression_ratio=reduction,
            retained_fact_count=len(facts),
            source_coverage=round(coverage, 6),
            summary_version="local-v1" if not supplied_summary else "provided-v1",
            preserved_fields=[
                "status",
                "facts",
                "latest_message",
                "recent_messages",
                "summary",
                "missing_fields",
                "state",
            ],
        )
        return ContextSnapshot(
            task_id=task_id,
            version=version,
            source_seq=source_seq,
            recent_event_ids=recent_ids,
            summary_ids=summary_ids,
            active_fact_ids=active_ids,
            status=status,
            summary=summary_text,
            facts=facts,
            evidence=evidence,
            recent_messages=recent,
            memory_refs=refs,
            state=state,
            rendered_context=rendered,
            unit_count=compiled_units,
            compression_stats={
                "raw_units": raw_units,
                "compiled_units": compiled_units,
                "compression_ratio": reduction,
                "source_coverage": coverage,
                "retained_fact_count": len(facts),
                "summary_version": stats.summary_version,
                "raw_chars": raw_units,
                "compacted_chars": compiled_units,
                "estimated_tokens": stats.estimated_tokens,
            },
            stats=stats,
        )

    snapshot = compact
    build_snapshot = compact

    def compile(
        self,
        context: BaseModel | Mapping[str, Any] | Any,
        *,
        memories: Sequence[Mapping[str, Any]] = (),
        active_facts: Mapping[str, Any] | None = None,
        messages: Sequence[Any] | None = None,
        summary: str | None = None,
    ) -> Any:
        """Compile the immutable model handed to an Agent."""
        from civicnexus.domain.context import ChatMessage, CompiledContext

        data = _mapping(context)
        if active_facts is not None:
            data["extracted"] = dict(active_facts)
        snap = self.compact(data, messages=messages, summary=summary)
        chat = _chat_messages(data, messages)
        recent = tuple(chat[-self.max_messages :])
        if not recent:
            recent = tuple(
                (ChatMessage(role="user", content=item) for item in snap.recent_messages)
            )
        latest = recent[-1].content if recent else ""
        return CompiledContext(
            task_id=snap.task_id,
            user_id=str(data.get("user_id", "anonymous")),
            session_id=str(data.get("session_id", "default")),
            status=snap.status,
            latest_message=latest,
            recent_messages=recent,
            active_facts=dict(snap.facts),
            memories=tuple((dict(item) for item in memories)),
            summary=snap.summary,
            missing_fields=tuple(_strings(data.get("missing_fields"))),
            evidence=tuple((dict(item) for item in snap.evidence)),
            routing=dict(data.get("routing") or snap.state.get("routing") or {}),
            review=dict(data.get("review") or snap.state.get("review") or {}),
            skill=dict(data.get("skill") or snap.state.get("skill") or {}),
            context_stats=snap.stats.model_dump(mode="json"),
        )

    def save_checkpoint(self, context: Any, **kwargs: Any) -> ContextSnapshot:
        snap = self.compact(context, **kwargs)
        if self.snapshot_store is not None:
            _save(self.snapshot_store, snap)
        return snap

    def load_checkpoint(self, task_id: TaskId) -> ContextSnapshot | None:
        return _load(self.snapshot_store, task_id) if self.snapshot_store is not None else None

    def restore(
        self, snapshot: ContextSnapshot, context: BaseModel | None = None
    ) -> dict[str, Any] | BaseModel:
        from civicnexus.domain.context import ChatMessage

        stats = (
            snapshot.stats
            if isinstance(snapshot.stats, ContextStats)
            else ContextStats.model_validate(snapshot.stats or {})
        )
        state = snapshot.state if isinstance(snapshot.state, Mapping) else {}
        messages = snapshot.recent_messages or []
        roles = state.get("message_roles", [])
        restored_messages = [
            ChatMessage(
                role=roles[index]
                if index < len(roles) and roles[index] in {"user", "assistant"}
                else ("user" if index % 2 == 0 else "assistant"),
                content=text,
            )
            for index, text in enumerate(messages)
        ]
        latest_user = next(
            (item.content for item in reversed(restored_messages) if item.role == "user"), ""
        )
        values: dict[str, Any] = {
            "task_id": snapshot.task_id,
            "status": snapshot.status,
            "summary": snapshot.summary or "",
            "facts": snapshot.facts or {},
            "extracted": snapshot.facts or {},
            "evidence": snapshot.evidence or [],
            "messages": restored_messages,
            "user_message": latest_user or (messages[-1] if messages else ""),
            "memory_refs": snapshot.memory_refs or [],
            "last_event_seq": snapshot.source_seq,
            "rendered_context": snapshot.rendered_context or "",
            "snapshot_id": str(snapshot.snapshot_id),
            "context_stats": stats.model_dump(mode="json"),
            "recent_event_ids": snapshot.recent_event_ids or [],
            "summary_ids": snapshot.summary_ids or [],
            "active_fact_ids": snapshot.active_fact_ids or [],
            **state,
        }
        if context is None:
            return values
        if getattr(context, "user_message", ""):
            values["user_message"] = context.user_message
        fields = getattr(context, "model_fields", {})
        payload = context.model_dump(mode="python")
        payload.update({key: value for key, value in values.items() if key in fields})
        return context.__class__.model_validate(payload)

    def render(self, snapshot: ContextSnapshot) -> str:
        return snapshot.rendered_context


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError("context must be a model, mapping, or object")


def _facts(data: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("extracted", "slots", "facts"):
        value = data.get(key)
        if isinstance(value, Mapping):
            for name, item in value.items():
                result[str(name)] = getattr(item, "value", item)
    return result


def _fact_ids(data: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    source = data.get("active_fact_ids") or data.get("fact_records") or data.get("facts")
    if (
        isinstance(source, Sequence)
        and not isinstance(source, str)
        and all(isinstance(item, (str, int)) for item in source)
    ):
        return [str(item) for item in source if str(item)]
    if isinstance(source, Mapping):
        source = list(source.values())
    if isinstance(source, Sequence) and (not isinstance(source, str)):
        for item in source:
            identifier = getattr(item, "fact_id", None) or (
                item.get("fact_id") if isinstance(item, Mapping) else None
            )
            if identifier:
                result.append(str(identifier))
    return result


def _messages(data: Mapping[str, Any], supplied: Sequence[Any] | None) -> list[str]:
    source: Any = (
        supplied if supplied is not None else data.get("messages", data.get("conversation"))
    )
    if source is None:
        source = [data.get("user_message", "")]
    if isinstance(source, str):
        source = [source]
    if not isinstance(source, Sequence):
        return []
    result = []
    for item in source:
        text = (
            item
            if isinstance(item, str)
            else item.get("content", "")
            if isinstance(item, Mapping)
            else getattr(item, "content", "")
        )
        if isinstance(text, str) and text.strip():
            result.append(text.strip())
    return result


def _chat_messages(data: Mapping[str, Any], supplied: Sequence[Any] | None) -> list[Any]:
    from civicnexus.domain.context import ChatMessage

    source: Any = supplied if supplied is not None else data.get("messages")
    if source is None:
        source = [data.get("user_message", "")]
    if isinstance(source, str):
        source = [source]
    if not isinstance(source, Sequence):
        return []
    result = []
    for item in source:
        if isinstance(item, ChatMessage):
            result.append(item)
        elif (
            isinstance(item, Mapping)
            and isinstance(item.get("content"), str)
            and (item.get("role", "user") in {"user", "assistant"})
        ):
            result.append(ChatMessage(role=item.get("role", "user"), content=item["content"]))
        elif isinstance(item, str) and item.strip():
            result.append(ChatMessage(role="user", content=item.strip()))
    return result


def _state(data: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "routing",
        "review",
        "checkpoint",
        "missing_fields",
        "review_round",
        "skill",
        "pending_tools",
        "pending_profile_fact",
        "created_case_id",
        "reply",
        "checkpoint_version",
        "last_event_seq",
    )
    result = {key: data[key] for key in keys if data.get(key) not in (None, {}, [])}
    if isinstance(data.get("state"), Mapping):
        result.update(data["state"])
    return result


def _segments(messages: Sequence[str]) -> list[str]:
    return [
        "历史分段：" + "；".join(messages[index : index + 8])[:1200]
        for index in range(0, len(messages), 8)
    ]


def _latest_correction(messages: Sequence[str]) -> str:
    markers = ("纠正", "更正", "改为", "不是", "说错", "应为", "忘记")
    return next(
        (item for item in reversed(messages) if any((mark in item for mark in markers))), ""
    )


def _dicts(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(item) for item in value]
        if isinstance(value, Sequence)
        and (not isinstance(value, str))
        and all((isinstance(item, Mapping) for item in value))
        else []
    )


def _strings(value: Any) -> list[str]:
    return (
        [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, Sequence) and (not isinstance(value, str))
        else []
    )


def _value(value: Any, default: str) -> str:
    value = getattr(value, "value", value)
    return default if value is None or not str(value).strip() else str(value).strip()


def _int(value: Any, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _clip(value: str, limit: int) -> str:
    return "" if limit <= 0 else value if len(value) <= limit else value[: max(1, limit - 1)] + "…"


def _evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("case_id", "category_zh", "status_zh", "keywords_zh", "score", "source")
    return {key: item[key] for key in keys if item.get(key) not in (None, "", [])}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _fit(payload: dict[str, Any], limit: int) -> str:
    """Shrink low-value prose, retaining valid JSON and structured facts."""
    text = _json(payload)
    while len(text) > limit:
        if payload.get("summary_segments"):
            payload["summary_segments"] = payload["summary_segments"][:-1]
        elif payload.get("summary"):
            payload["summary"] = _clip(payload["summary"], max(0, len(payload["summary"]) // 2))
        elif len(payload.get("recent_messages", [])) > 1:
            payload["recent_messages"] = payload["recent_messages"][-1:]
        elif payload.get("evidence"):
            payload["evidence"] = payload["evidence"][:-1]
        elif payload.get("memory_refs"):
            payload["memory_refs"] = payload["memory_refs"][:-1]
        else:
            state = payload.get("state", {})
            removable = [
                key
                for key in state
                if key
                not in {
                    "last_correction",
                    "unresolved_fields",
                    "pending_tools",
                    "routing",
                    "review",
                    "skill",
                }
            ]
            if removable:
                state.pop(removable[-1])
            else:
                facts = payload.get("facts", {})
                changed = {key: _clip(str(value), 80) for key, value in facts.items()}
                if changed == facts:
                    break
                payload["facts"] = changed
        new_text = _json(payload)
        if new_text == text:
            break
        text = new_text
    if len(text) > limit:
        # A pathological combination of long keys/values can outlive the
        # normal reductions above.  Keep a tiny structured representation so
        # the rendered context still honours the hard budget and remains JSON.
        facts = {
            str(key)[:32]: _clip(str(value), 40) for key, value in payload.get("facts", {}).items()
        }
        compact: dict[str, Any] = {
            "status": _clip(str(payload.get("status", "")), 32),
            "facts": facts,
        }
        latest = payload.get("latest_message")
        if latest:
            compact["latest_message"] = _clip(str(latest), 64)
        text = _json(compact)
        if len(text) > limit:
            compact.pop("latest_message", None)
            text = _json(compact)
        while len(text) > limit and facts:
            facts.pop(next(reversed(facts)))
            text = _json(compact)
        if len(text) > limit:
            status = str(compact.get("status", ""))
            for size in range(len(status), -1, -1):
                candidate = _json({"status": status[:size]})
                if len(candidate) <= limit:
                    return candidate
            return "0" if limit == 1 else "{}"
    return text


def _digest(value: str) -> str:
    return "summary-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _save(store: Any, snapshot: ContextSnapshot) -> None:
    if hasattr(store, "save_snapshot"):
        store.save_snapshot(snapshot)
    elif hasattr(store, "save_context_snapshot"):
        store.save_context_snapshot(snapshot)
    elif hasattr(store, "save"):
        store.save(snapshot)
    else:
        raise TypeError("snapshot store has no save method")


def _load(store: Any, task_id: TaskId) -> ContextSnapshot | None:
    if hasattr(store, "load"):
        value = store.load(task_id)
    elif hasattr(store, "load_context_snapshot"):
        value = store.load_context_snapshot(task_id)
    elif hasattr(store, "get_latest_snapshot"):
        value = store.get_latest_snapshot(task_id)
    elif hasattr(store, "get_snapshot"):
        value = store.get_snapshot(task_id)
    else:
        raise TypeError("snapshot store has no load method")
    if value is None or isinstance(value, ContextSnapshot):
        return value
    if isinstance(value, Mapping):
        return ContextSnapshot.model_validate(value)
    columns = getattr(getattr(value, "__table__", None), "columns", ())
    return ContextSnapshot.model_validate(
        {column.name: getattr(value, column.name) for column in columns}
    )


__all__ = ["ContextManager"]
