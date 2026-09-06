from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from civicnexus.domain.context import TaskContext
from civicnexus.persistence.cache import MemoryCache
from civicnexus.persistence.database import create_db_engine, init_db
from civicnexus.persistence.repositories import TaskRepository


def repository() -> TaskRepository:
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return TaskRepository(factory)


def test_event_sequence_snapshot_and_local_case_are_recoverable() -> None:
    repo = repository()
    context = TaskContext(user_message="井盖附近积水")
    repo.create_task(context, user_id="u-1", session_id="s-1")

    first = repo.append_event(context.task_id, "user", "message", {"text": "井盖附近积水"})
    second = repo.append_event(context.task_id, "agent", "proposal", {"token": "secret"})
    context.checkpoint = "checkpoint-1"
    context.checkpoint_version = 1
    repo.save_task(context)
    repo.add_fact(
        namespace="task",
        namespace_id=str(context.task_id),
        key="location",
        value="井盖附近",
        source="user",
        source_event_ids=[first.event_id],
    )

    assert [first.seq, second.seq] == [1, 2]
    assert second.payload_redacted["token"] == "[REDACTED]"
    assert [event.seq for event in repo.list_events(context.task_id)] == [1, 2]
    assert repo.get_latest_snapshot(context.task_id) is not None

    snapshot = repo.rebuild_snapshot(context.task_id)
    assert snapshot.source_seq == 2
    assert snapshot.active_fact_ids

    case_one = repo.create_local_case(context.task_id, {"department": "排水部门"})
    case_two = repo.create_local_case(context.task_id, {"department": "其他部门"})
    assert case_one.case_id == case_two.case_id


def test_fact_versions_profile_confirmation_and_namespace_isolation() -> None:
    repo = repository()
    context = TaskContext(user_message="地址先说错了")
    repo.create_task(context, user_id="u-1", session_id="s-1")
    old = repo.add_fact(
        namespace="task", namespace_id=str(context.task_id), key="location", value="A", source="llm"
    )
    new = repo.add_fact(
        namespace="task",
        namespace_id=str(context.task_id),
        key="location",
        value="B",
        source="correction",
    )
    assert (
        repo.list_facts("task", str(context.task_id), include_inactive=True)[-1].status
        == "superseded"
    )
    assert new.supersedes_fact_id == old.fact_id

    candidate = repo.add_fact(
        namespace="user", namespace_id="u-1", key="preferred_area", value="城北", source="user"
    )
    profile = repo.confirm_profile_fact(
        "u-1",
        "preferred_area",
        fact_id=candidate.fact_id,
        task_id=context.task_id,
        event_payload={"confirmed": True},
    )
    assert profile.confirmation_event_id
    assert repo.list_events(context.task_id)[-1].event_type == "profile_confirmation"
    assert repo.list_profile_facts("u-1")[0].value_json == "城北"
    assert repo.list_profile_facts("u-2") == []

    repo.forget_profile_fact("u-1", "preferred_area", task_id=context.task_id)
    assert repo.list_profile_facts("u-1") == []
    assert repo.list_profile_facts("u-1", include_inactive=True)[0].status == "forgotten"


def test_domain_snapshot_round_trip_keeps_compiled_fields() -> None:
    from civicnexus.memory.models import ContextSnapshot

    repo = repository()
    context = TaskContext(user_message="snapshot")
    repo.create_task(context)
    snapshot = ContextSnapshot(
        task_id=str(context.task_id),
        source_seq=3,
        status="INTAKE",
        summary="历史摘要",
        facts={"location": "人民路"},
        recent_messages=["snapshot"],
        rendered_context='{"facts":{"location":"人民路"}}',
        unit_count=33,
    )
    repo.save_snapshot(snapshot)
    restored = repo.load_context_snapshot(context.task_id)
    assert restored is not None
    assert restored.summary == snapshot.summary
    assert restored.facts == snapshot.facts
    assert restored.recent_messages == snapshot.recent_messages


def test_expired_facts_and_memory_cache_are_not_recalled() -> None:
    repo = repository()
    repo.add_fact(
        namespace="session",
        namespace_id="s-old",
        key="category",
        value="垃圾",
        source="user",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert repo.list_facts("session", "s-old") == []

    now = [10.0]
    cache = MemoryCache(clock=lambda: now[0])
    context = TaskContext(user_message="test")
    cache.save_context(context, ttl_seconds=5)
    assert cache.get_context(context.task_id) == context
    now[0] = 16.0
    assert cache.get_context(context.task_id) is None
