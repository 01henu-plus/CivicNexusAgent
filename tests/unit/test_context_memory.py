from __future__ import annotations

import json

import pytest

from civicnexus.domain.context import TaskContext
from civicnexus.memory import (
    ContextSnapshot,
    InMemoryEventLog,
    InMemoryFactLedger,
    InMemoryLayeredMemory,
    InMemorySnapshotStore,
    MemoryLayer,
)
from civicnexus.runtime.context_manager import ContextManager
from civicnexus.runtime.engine import AgentRuntime
from civicnexus.skills import SkillRegistry
from civicnexus.tools import ToolRegistry


def test_event_log_is_append_only_and_scoped() -> None:
    log = InMemoryEventLog()
    first = log.append("task-a", "received", {"message": "hello"}, actor="api")
    second = log.append("task-a", "transition", state_before="RECEIVED", state_after="INTAKE")
    log.append("task-b", "received")

    assert (first.sequence, second.sequence) == (1, 2)
    assert [event.event_type for event in log.list("task-a")] == ["received", "transition"]
    assert len(log.list("task-a", after_sequence=1)) == 1

    # Returned records are copies, so a consumer cannot mutate the log.
    first.payload["message"] = "changed"
    assert log.list("task-a")[0].payload["message"] == "hello"


def test_fact_ledger_tracks_revision_and_restores_snapshot() -> None:
    ledger = InMemoryFactLedger()
    first = ledger.set("task-a", "location", "1 Main St", source="user", confidence=0.8)
    second = ledger.set("task-a", "location", "2 Main St", source="user", confidence=1.0)

    assert (first.revision, second.revision) == (1, 2)
    assert ledger.get("task-a", "location").value == "2 Main St"
    assert ledger.get("task-b", "location") is None

    snapshot = ledger.snapshot("task-a")
    ledger.set("task-a", "category", "垃圾处理")
    ledger.restore(snapshot)
    assert set(ledger.all("task-a")) == {"location"}
    assert ledger.snapshot("task-a").version == snapshot.version


def test_layered_memory_search_respects_layer_and_scope() -> None:
    memory = InMemoryLayeredMemory()
    memory.put("session-a", MemoryLayer.EPISODIC, "last_case", "污水井堵塞", importance=0.9)
    memory.put("session-a", MemoryLayer.SEMANTIC, "route", "道路部门", importance=0.4)
    memory.put("session-b", MemoryLayer.EPISODIC, "other", "污水井堵塞", importance=1.0)

    hits = memory.search("session-a", "污水", layer=MemoryLayer.EPISODIC)
    assert [hit.key for hit in hits] == ["last_case"]
    assert memory.search("session-a", "污水", layer=MemoryLayer.SEMANTIC) == []
    assert memory.get("session-b", "last_case") is None


def test_context_compaction_preserves_structured_facts_and_budget() -> None:
    context = TaskContext(
        user_message="补充地址：人民路 1 号",
        extracted={"category": "污水与下水道", "location": "人民路 1 号"},
        evidence=[{"case_id": "case-1", "category_zh": "排水"}],
    )
    context.append_message("user", "污水井堵了")
    context.append_message("assistant", "请补充地址")
    context.append_message("user", "补充地址：人民路 1 号")

    snapshot = ContextManager(max_chars=360, max_messages=2).compact(context)

    assert len(snapshot.rendered_context) <= 360
    assert snapshot.facts["category"] == "污水与下水道"
    assert snapshot.recent_messages[-1] == "补充地址：人民路 1 号"
    assert snapshot.stats.compacted_chars == len(snapshot.rendered_context)
    assert snapshot.compression_stats["compacted_chars"] == len(snapshot.rendered_context)
    assert json.loads(snapshot.rendered_context)["facts"]["location"] == "人民路 1 号"


def test_checkpoint_round_trip_and_compiled_context() -> None:
    store = InMemorySnapshotStore()
    manager = ContextManager(max_chars=500, snapshot_store=store)
    context = TaskContext(user_id="u1", session_id="s1", user_message="下水道堵塞")
    context.append_message("user", "下水道堵塞")
    context.extracted = {"category": "污水与下水道"}

    snapshot = manager.save_checkpoint(context)
    loaded = manager.load_checkpoint(context.task_id)
    assert loaded is not None
    assert loaded.snapshot_id == snapshot.snapshot_id

    restored = manager.restore(loaded, context)
    assert isinstance(restored, TaskContext)
    assert restored.user_message == "下水道堵塞"
    assert restored.extracted["category"] == "污水与下水道"

    compiled = manager.compile(context, memories=[{"key": "route", "value": "排水"}])
    assert compiled.latest_message == "下水道堵塞"
    assert compiled.active_facts["category"] == "污水与下水道"
    assert compiled.memories[0]["key"] == "route"


def test_snapshot_store_rejects_stale_versions() -> None:
    store = InMemorySnapshotStore()
    store.save(ContextSnapshot(task_id="t", version=2))
    with pytest.raises(ValueError, match="newer"):
        store.save(ContextSnapshot(task_id="t", version=1))


def test_address_correction_takes_priority_over_profile_prompt() -> None:
    runtime = AgentRuntime([], tools=ToolRegistry(), skills=SkillRegistry("configs/skills"))
    context = TaskContext(
        status="COMPLETED",
        user_message="地址A",
        pending_profile_fact={"key": "preferred_area", "value": "地址A"},
    )

    assert runtime._handle_profile_intent(context, "不是，地址改为地址B") is False


def test_completed_case_reply_explains_next_steps() -> None:
    runtime = AgentRuntime([], tools=ToolRegistry(), skills=SkillRegistry("configs/skills"))
    context = TaskContext(
        status="COMPLETED",
        user_message="小区门口积水",
        extracted={"category": "排水、积水与洪涝", "location": "小区门口"},
        routing={"department": "排水部门", "priority": "普通"},
        created_case_id="LOCAL-DEMO-01",
    )

    reply = runtime._case_completion_reply(context, include_profile_prompt=True)

    assert "LOCAL-DEMO-01" in reply
    assert "处理方向：排水部门" in reply
    assert "你接下来可以" in reply
    assert "真实城市管理平台" in reply
    assert "触电" in reply
    assert "常用区域" in reply


def test_replay_profile_candidate_uses_fact_value_at_checkpoint() -> None:
    """A correction may version a fact without emitting another memory write."""
    events = InMemoryEventLog()
    facts = InMemoryFactLedger()
    runtime = AgentRuntime(
        [],
        tools=ToolRegistry(),
        skills=SkillRegistry("configs/skills"),
        event_log=events,
        fact_ledger=facts,
    )
    context = TaskContext(user_id="u1", session_id="s1", user_message="地址A")
    events.append(context.task_id, "TASK_CREATED", {"message": "地址A"}, actor="user")
    prompt = events.append(
        context.task_id,
        "MEMORY_WRITE",
        {"profile_candidate": {"key": "preferred_area", "value": "地址A"}},
        actor="harness",
    )
    facts.set(
        context.task_id,
        "preferred_area",
        "地址A",
        source="user",
        status="candidate",
        source_event_ids=[str(prompt.event_id)],
    )
    correction = events.append(
        context.task_id,
        "USER_MESSAGE",
        {"message": "不是，地址改为地址B"},
        actor="user",
    )
    facts.set(
        context.task_id,
        "preferred_area",
        "地址B",
        source="correction",
        status="candidate",
        source_event_ids=[str(correction.event_id)],
    )

    old = runtime.rebuild_snapshot(context, source_seq=prompt.sequence)
    current = runtime.rebuild_snapshot(context)

    assert old.state["pending_profile_fact"]["value"] == "地址A"
    assert current.state["pending_profile_fact"]["value"] == "地址B"
