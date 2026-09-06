from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from civicnexus.domain.enums import TaskStatus
from civicnexus.evaluation.runner import Metric, default_predictor
from civicnexus.runtime.state_machine import can_transition

REQUIRED_FACTS = ("category", "location")
KNOWN_EVENTS = {
    "TASK_CREATED",
    "USER_MESSAGE",
    "ASSISTANT_MESSAGE",
    "STATE_TRANSITION",
    "AGENT_PROPOSAL",
    "TOOL_CALL",
    "TOOL_FAILED",
    "HARNESS_PHASE",
    "CHECKPOINT",
    "FACT_DELETED",
    "PROFILE_CONFIRMED",
    "PROFILE_REJECTED",
    "PROFILE_FORGOTTEN",
    "PROFILE_CONFIRMATION",
    "PROFILE_REJECTION",
    "PROFILE_FORGET",
    "MEMORY_WRITE",
    "MEMORY_READ",
}
_METRICS = (
    ("completion_rate", "任务完成率", 0.9, "gte"),
    ("valid_transition_rate", "合法状态转移率", 1.0, "gte"),
    ("event_reconstruction_completeness", "事件重建完整率", 1.0, "gte"),
    ("checkpoint_coverage", "Checkpoint 覆盖率", 1.0, "gte"),
    ("checkpoint_resume_equivalence", "断点恢复一致率", 1.0, "gte"),
    ("snapshot_rebuild_consistency", "快照重建一致率", 1.0, "gte"),
    ("summary_reference_validity", "摘要引用有效率", 1.0, "gte"),
    ("correction_overwrite_rate", "纠正覆盖率", 1.0, "gte"),
    ("unconfirmed_profile_write_rate", "未确认画像写入率", 0.0, "lte"),
    ("profile_confirmation_event_rate", "画像确认事件率", 1.0, "gte"),
    ("expired_memory_false_recall", "过期记忆误召回率", 0.0, "lte"),
    ("namespace_isolation", "记忆命名空间隔离率", 1.0, "gte"),
    ("critical_fact_recall", "关键事实召回率", 1.0, "gte"),
    ("critical_fact_retention", "关键事实保留率", 1.0, "gte"),
    ("memory_recall_at_3", "记忆 Recall@3", 0.9, "gte"),
    ("compression_ratio", "上下文压缩率", 0.5, "gte"),
    ("tool_success_rate", "工具调用成功率", 1.0, "gte"),
    ("duplicate_side_effect_rate", "重复副作用率", 0.0, "lte"),
    ("skill_selection_accuracy", "Skill 选择准确率", 1.0, "gte"),
    ("routing_accuracy", "部门路由准确率", 1.0, "gte"),
    ("memory_hit_rate", "记忆命中率", 0.5, "gte"),
    ("trace_event_count", "Trace 事件数", 0.0, "gte"),
)


def runtime_metrics(repository: Any, *, limit: int = 500) -> dict[str, Any]:
    rows = _read(repository, "list_tasks", limit=limit)
    total, completed = len(rows), 0
    n: defaultdict[str, int] = defaultdict(int)
    series: defaultdict[str, list[Any]] = defaultdict(list)
    idempotent: defaultdict[str, set[str]] = defaultdict(set)
    profiles: dict[str, Any] = {}
    expired: set[str] = set()
    observed_scopes: list[tuple[str | None, set[str], str | None]] = []
    for row in rows:
        task_id = str(getattr(row, "id", getattr(row, "task_id", "")))
        ctx = _map(getattr(row, "context_json", {}))
        user_id = str(getattr(row, "user_id", None) or ctx.get("user_id") or "")
        session_id = str(getattr(row, "session_id", None) or ctx.get("session_id") or "")
        completed += _state(getattr(row, "status", "")) == TaskStatus.COMPLETED.value
        n["checkpoints"] += bool(
            _int(getattr(row, "checkpoint_version", None) or ctx.get("checkpoint_version"))
        )
        stats = _map(ctx.get("context_stats") or ctx.get("stats"))
        if (
            ratio := _number(stats.get("reduction_ratio", stats.get("compression_ratio")))
        ) is not None:
            series["compression"].append(ratio)
        facts = _map(ctx.get("extracted") or ctx.get("facts"))
        series["fact_recall"].append(_fraction(bool(facts.get(key)) for key in REQUIRED_FACTS))
        message = str(ctx.get("user_message") or "")
        if not message and _sequence(ctx.get("messages")) and ctx["messages"]:
            last = ctx["messages"][-1]
            message = str(last.get("content", "") if isinstance(last, Mapping) else last)
        if message:
            expected = default_predictor(message)
            actual_skill = ctx.get("skill", {})
            actual_skill = (
                actual_skill.get("name") if isinstance(actual_skill, Mapping) else actual_skill
            )
            series["skill"].append(actual_skill == expected.skill)
            series["route"].append(
                _map(ctx.get("routing")).get("department") == expected.department
            )
        hits = ctx.get("memory_hits")
        if _sequence(hits):
            n["memory_tasks"] += bool(hits)
            expected_ids = ctx.get("expected_memory_ids") or ctx.get("memory_gold_ids")
            if _sequence(expected_ids) and expected_ids:
                wanted = {str(item) for item in expected_ids}
                actual = {_memory_id(item) for item in hits[:3] if isinstance(item, Mapping)}
                series["memory_recall"].append(len(actual & wanted) / len(wanted))
            allowed = (
                {f"task:{task_id}"}
                | (
                    {f"session:{session_id}", f"session:{user_id}:{session_id}"}
                    if session_id
                    else set()
                )
                | ({f"user:{user_id}"} if user_id else set())
            )
            for item in hits:
                if isinstance(item, Mapping):
                    scope = item.get("scope_id") or item.get("namespace")
                    if scope and ":" not in str(scope) and item.get("namespace_id"):
                        scope = f"{scope}:{item['namespace_id']}"
                    observed_scopes.append(
                        (_memory_id(item), allowed, str(scope) if scope else None)
                    )
        events = _read(repository, "list_events", task_id)
        n["events"] += len(events)
        for event in events:
            kind, payload = str(getattr(event, "event_type", "")).upper(), _payload(event)
            before, after = _states(event, payload)
            n["reconstructable"] += _reconstructable(kind, payload, before, after)
            if before and after:
                series["transitions"].append((before, after))
            if kind == "TOOL_CALL":
                n["tools"] += 1
                n["tool_ok"] += payload.get("success") is True
                args, result = _map(payload.get("arguments")), _map(payload.get("result"))
                key, case_id = (
                    args.get("idempotency_key") or payload.get("idempotency_key"),
                    result.get("case_id") or payload.get("case_id"),
                )
                if key and case_id:
                    idempotent[str(key)].add(str(case_id))
            if kind in {
                "CHECKPOINT_RESUME",
                "CHECKPOINT_RESTORED",
                "SNAPSHOT_RESTORE",
            } and isinstance(payload.get("equivalent"), bool):
                series["resume"].append(payload["equivalent"])
        task_facts = _read(repository, "list_facts", "task", task_id, include_inactive=True)
        for fact in task_facts:
            if "correction" in str(getattr(fact, "source", "")).lower():
                n["correction"] += 1
                n["correction_ok"] += bool(
                    getattr(fact, "supersedes_fact_id", None)
                    or _int(getattr(fact, "version", 0)) > 1
                )
        for namespace, identifier in (
            ("task", task_id),
            ("session", f"{user_id}:{session_id}" if session_id else ""),
            ("user", user_id),
        ):
            if identifier:
                _expired(repository, namespace, identifier, expired)
        if user_id:
            for profile in _read(repository, "list_profile_facts", user_id, include_inactive=True):
                profile_id = getattr(profile, "profile_fact_id", getattr(profile, "id", None))
                if profile_id:
                    profiles[str(profile_id)] = profile
        snapshot = _get(repository, "get_latest_snapshot", task_id)
        if snapshot is not None:
            n["snapshots"] += 1
            series["retention"].append(_retention(snapshot, facts))
            n["snapshots_ok"] += _consistent(snapshot, row, ctx)
            refs, valid = _references(snapshot, task_facts, events)
            n["refs"], n["refs_ok"] = n["refs"] + refs, n["refs_ok"] + valid
    recalled = _expired_recalls(repository, rows, expired)
    isolation_seen = isolation_ok = 0
    for memory_id, allowed, scope in observed_scopes:
        if not memory_id or not scope:
            continue
        isolation_seen += 1
        isolation_ok += scope in allowed
    duplicate = _rate(sum(len(ids) > 1 for ids in idempotent.values()), len(idempotent))
    values = {
        "completion_rate": _rate(completed, total),
        "valid_transition_rate": _transition_rate(series["transitions"]),
        "event_reconstruction_completeness": _rate(n["reconstructable"], n["events"]),
        "checkpoint_coverage": _rate(n["checkpoints"], total),
        "checkpoint_resume_equivalence": _mean(series["resume"]),
        "snapshot_rebuild_consistency": _rate(n["snapshots_ok"], n["snapshots"]),
        "summary_reference_validity": _rate(n["refs_ok"], n["refs"]),
        "correction_overwrite_rate": _rate(n["correction_ok"], n["correction"]),
        "unconfirmed_profile_write_rate": _rate(
            sum(not getattr(p, "confirmation_event_id", None) for p in profiles.values()),
            len(profiles),
        ),
        "profile_confirmation_event_rate": _rate(
            sum(
                bool(getattr(p, "confirmation_event_id", None))
                for p in profiles.values()
                if _state(getattr(p, "status", "")) == "confirmed"
            ),
            sum(_state(getattr(p, "status", "")) == "confirmed" for p in profiles.values()),
        ),
        "expired_memory_false_recall": _rate(len(recalled & expired), len(expired)),
        "namespace_isolation": _rate(isolation_ok, isolation_seen),
        "critical_fact_recall": _mean(series["fact_recall"]),
        "critical_fact_retention": _mean(series["retention"]),
        "memory_recall_at_3": _mean(series["memory_recall"]),
        "compression_ratio": _mean(series["compression"]),
        "tool_success_rate": _rate(n["tool_ok"], n["tools"]),
        "duplicate_side_effect_rate": duplicate,
        "skill_selection_accuracy": _mean(series["skill"]),
        "routing_accuracy": _mean(series["route"]),
        "memory_hit_rate": _rate(n["memory_tasks"], total),
        "trace_event_count": float(n["events"]),
    }
    metrics = [
        _metric(name, values[name], target, comparison, label)
        for name, label, target, comparison in _METRICS
    ]
    return {
        "sample_count": total,
        "metrics": [item.model_dump(mode="json") for item in metrics],
        "evidence": {
            "events": n["events"],
            "transitions": len(series["transitions"]),
            "tool_calls": n["tools"],
            "snapshots": n["snapshots"],
            "profile_facts": len(profiles),
            "expired_memory_items": len(expired),
            "memory_recall_cases": len(series["memory_recall"]),
        },
    }


def _metric(name: str, value: float, target: float, comparison: str, label: str) -> Metric:
    return Metric(
        name=name,
        label=label,
        value=round(value, 6),
        target=target,
        passed=value >= target if comparison == "gte" else value <= target,
        unit="events" if name == "trace_event_count" else None,
        comparison=comparison,
    )


def _read(repo: Any, name: str, *args: Any, **kwargs: Any) -> list[Any]:
    fn = getattr(repo, name, None)
    if not fn:
        return []
    try:
        return list(fn(*args, **kwargs) or [])
    except TypeError:
        # A few lightweight test/demonstration stores omit optional keywords.
        return list(fn(*args) or [])


def _get(repo: Any, name: str, *args: Any) -> Any | None:
    fn = getattr(repo, name, None)
    return fn(*args) if fn else None


def _map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
        return dict(value) if isinstance(value, Mapping) else {}
    table = getattr(value, "__table__", None)
    if table is not None:
        return {column.name: getattr(value, column.name) for column in table.columns}
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, str):
        try:
            value = json.loads(value)
            return dict(value) if isinstance(value, Mapping) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return {}


def _payload(event: Any) -> dict[str, Any]:
    return _map(
        getattr(event, "payload_redacted", None)
        or getattr(event, "payload", None)
        or getattr(event, "payload_json", None)
    )


def _state(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("value") or value.get("name")
    value = getattr(value, "value", value)
    return str(value) if value not in (None, "") else None


def _states(event: Any, payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return (
        _state(
            getattr(event, "state_before", None)
            or payload.get("state_before")
            or payload.get("from")
        ),
        _state(
            getattr(event, "state_after", None) or payload.get("state_after") or payload.get("to")
        ),
    )


def _reconstructable(
    kind: str, payload: Mapping[str, Any], before: str | None, after: str | None
) -> bool:
    if kind not in KNOWN_EVENTS:
        return False
    return (
        bool(before and after)
        if kind == "STATE_TRANSITION"
        else (
            isinstance(payload.get("success"), bool) and bool(payload.get("name"))
            if kind == "TOOL_CALL"
            else True
        )
    )


def _references(snapshot: Any, facts: Sequence[Any], events: Sequence[Any]) -> tuple[int, int]:
    snap, total, valid = _map(snapshot), 0, 0
    event_ids = {str(getattr(e, "event_id", "")) for e in events}
    known = {
        "recent_event_ids": event_ids,
        "active_fact_ids": {str(getattr(f, "fact_id", "")) for f in facts},
        "summary_ids": event_ids,
    }
    for key, ids in known.items():
        refs = snap.get(key)
        if _sequence(refs):
            total += len(refs)
            valid += sum(
                str(ref) in ids or (key == "summary_ids" and str(ref).startswith("summary-"))
                for ref in refs
            )
    if not total:
        return (1, 1)
    return total, valid


def _consistent(snapshot: Any, row: Any, context: Mapping[str, Any]) -> bool:
    snap = _map(snapshot)
    if not snap:
        return False
    current = _state(getattr(row, "status", None) or context.get("status"))
    if snap.get("status") and current and _state(snap["status"]) != current:
        return False
    saved, facts = _map(snap.get("facts")), _map(context.get("extracted") or context.get("facts"))
    return all(saved.get(key) == value for key, value in facts.items() if key in saved)


def _retention(snapshot: Any, facts: Mapping[str, Any]) -> float:
    saved = _map(_map(snapshot).get("facts"))
    return _fraction(saved.get(key) == facts.get(key) for key in REQUIRED_FACTS if key in facts)


def _expired(repo: Any, namespace: str, identifier: str, target: set[str]) -> None:
    now = datetime.now(UTC)
    for fact in _read(repo, "list_facts", namespace, identifier, include_inactive=True):
        expiry = getattr(fact, "expires_at", None)
        if expiry is not None:
            if not isinstance(expiry, datetime):
                expiry = datetime.fromisoformat(str(expiry))
            if (expiry if expiry.tzinfo else expiry.replace(tzinfo=UTC)) <= now:
                item_id = getattr(fact, "fact_id", getattr(fact, "id", None))
                if item_id:
                    target.add(str(item_id))


def _expired_recalls(repo: Any, rows: Sequence[Any], expired: set[str]) -> set[str]:
    if not expired or not hasattr(repo, "recall_memory"):
        return set()
    found: set[str] = set()
    for row in rows:
        try:
            values = repo.recall_memory(
                user_id=str(getattr(row, "user_id", "") or ""),
                session_id=getattr(row, "session_id", None),
                query="",
                top_k=max(3, len(expired)),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        found.update(item_id for value in values or [] if (item_id := _memory_id(value)))
    return found


def _memory_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("memory_id") or value.get("fact_id") or value.get("id")
    else:
        value = getattr(value, "memory_id", getattr(value, "fact_id", getattr(value, "id", None)))
    return str(value) if value else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[int | float | bool]) -> float:
    return _rate(sum(values), len(values))


def _fraction(values: Any) -> float:
    values = list(values)
    return _rate(sum(values), len(values))


def _transition_rate(transitions: Sequence[tuple[str, str]]) -> float:
    valid = 0
    for before, after in transitions:
        try:
            valid += can_transition(TaskStatus(before), TaskStatus(after))
        except (KeyError, ValueError):
            pass
    return _rate(valid, len(transitions))


__all__ = ["runtime_metrics"]
