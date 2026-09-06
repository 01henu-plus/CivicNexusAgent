from __future__ import annotations
import json
import re
import time
from typing import Any, Callable
from pydantic import BaseModel, Field
from civicnexus.agents.base import Agent
from civicnexus.domain.context import CompiledContext, TaskContext
from civicnexus.domain.enums import HarnessPhase, TaskStatus
from civicnexus.domain.results import AgentResult
from civicnexus.memory import (
    EventLog,
    FactLedger,
    InMemoryEventLog,
    InMemoryFactLedger,
    InMemoryLayeredMemory,
    LayeredMemory,
    MemoryLayer,
)
from civicnexus.runtime.context_manager import ContextManager
from civicnexus.runtime.state_machine import assert_transition
from civicnexus.runtime.workflow import Workflow
from civicnexus.skills import SkillRegistry
from civicnexus.tools import ToolCallRecord, ToolContext, ToolRegistry


class RunBudget(BaseModel):
    max_steps: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=8, ge=1)
    max_review_rounds: int = Field(default=2, ge=0)


class RunTrace(BaseModel):
    sequence: int
    phase: HarnessPhase
    status: TaskStatus
    agent: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class PolicyGate:
    def validate(
        self,
        context: TaskContext,
        proposal: AgentResult,
        allowed_tools: set[str],
        required_fields: set[str] | None = None,
    ) -> None:
        if not proposal.next_status:
            raise ValueError(f"Agent {proposal.agent_name} did not propose a next status.")
        assert_transition(context.status, proposal.next_status)
        protected = {
            "task_id",
            "user_id",
            "session_id",
            "status",
            "messages",
            "trace",
            "state_history",
            "checkpoint",
            "checkpoint_version",
            "last_event_seq",
        }
        if blocked := protected.intersection(proposal.patch):
            raise ValueError(
                f"Agent patch cannot write Harness fields: {', '.join(sorted(blocked))}"
            )
        denied = [call.name for call in proposal.tool_requests if call.name not in allowed_tools]
        if denied:
            raise ValueError(f"Tools are not allowed by the active skill: {', '.join(denied)}")
        if proposal.next_status == TaskStatus.CREATE_LOCAL_CASE:
            facts = {**context.extracted, **(proposal.patch.get("extracted") or {})}
            missing = [key for key in required_fields or set() if not facts.get(key)]
            if missing:
                raise ValueError(f"Required fields are missing: {', '.join(sorted(missing))}")


class StateReducer:
    """唯一修改任务状态的组件。Agent 只返回 proposal。"""

    def apply(
        self, context: TaskContext, proposal: AgentResult, calls: list[ToolCallRecord]
    ) -> None:
        for field, value in proposal.patch.items():
            if field == "extracted" and isinstance(value, dict):
                context.extracted.update(value)
            elif hasattr(context, field):
                setattr(context, field, value)
        context.missing_fields = list(proposal.missing_fields)
        if proposal.evidence:
            context.evidence = list(proposal.evidence[:3])
        if proposal.tool_requests:
            context.pending_tools = [
                item.model_dump(mode="json") for item in proposal.tool_requests
            ]
        if proposal.message_zh:
            context.reply = proposal.message_zh
        for call in calls:
            if not call.success:
                continue
            if call.name == "case_search":
                context.evidence = list(call.result or [])[:3]
            elif (
                call.name == "validate_address"
                and isinstance(call.result, dict)
                and call.result.get("valid")
            ):
                context.extracted["location"] = call.result["normalized"]
            elif call.name == "create_local_case":
                context.created_case_id = call.result["case_id"]
        if calls:
            context.pending_tools = []


class AgentRuntime:
    """The Harness loop: compile, gate, execute, reduce and checkpoint."""

    TERMINAL = {
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.WAITING_FOR_HUMAN,
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    }

    def __init__(
        self,
        agents: list[Agent],
        *,
        tools: ToolRegistry,
        skills: SkillRegistry,
        context_manager: ContextManager | None = None,
        event_log: EventLog | None = None,
        fact_ledger: FactLedger | None = None,
        memory: LayeredMemory | None = None,
        workflow: Workflow | None = None,
        budget: RunBudget | None = None,
        task_store: Any | None = None,
        checkpoint_store: Any | None = None,
        checkpoint_ttl: int = 86400,
    ) -> None:
        self.agents = {agent.name: agent for agent in agents}
        self.tools, self.skills = (tools, skills)
        self.context_manager = context_manager or ContextManager()
        self.event_log = event_log or InMemoryEventLog()
        self.fact_ledger = fact_ledger or InMemoryFactLedger()
        self.memory = memory or InMemoryLayeredMemory()
        self.workflow, self.budget = (workflow or Workflow(), budget or RunBudget())
        self.task_store, self.checkpoint_store = (task_store, checkpoint_store)
        self.checkpoint_ttl = checkpoint_ttl
        self.policy, self.reducer = (PolicyGate(), StateReducer())
        self._traces: dict[str, list[RunTrace]] = {}
        self._trace_callbacks: dict[str, Callable[[RunTrace], None]] = {}

    def start(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        on_trace: Callable[[RunTrace], None] | None = None,
    ) -> TaskContext:
        context = TaskContext(user_id=user_id, session_id=session_id, user_message=message)
        context.append_message("user", message)
        if self.task_store:
            self.task_store.create_task(context, user_id=user_id, session_id=session_id)
        self._event(context, "TASK_CREATED", {"message": _redact(message)}, actor="user")
        self._record(
            context,
            HarnessPhase.APPEND_EVENT,
            {"event_type": "TASK_CREATED", "seq": context.last_event_seq},
        )
        if _is_greeting(message):
            self._transition(context, TaskStatus.WAITING_FOR_USER, actor="greeting")
            context.reply = "你好！我是 CivicNexus，请描述你遇到的城市公共服务问题。"
            self._append_reply(context)
            self._checkpoint(context)
            return context
        return self.run(context, on_trace=on_trace)

    def add_message(
        self, context: TaskContext, message: str, on_trace: Callable[[RunTrace], None] | None = None
    ) -> TaskContext:
        context.append_message("user", message)
        self._event(context, "USER_MESSAGE", {"message": _redact(message)}, actor="user")
        self._record(
            context,
            HarnessPhase.APPEND_EVENT,
            {"event_type": "USER_MESSAGE", "seq": context.last_event_seq},
        )
        if self._handle_profile_intent(context, message):
            self._checkpoint(context)
            return context
        if context.status == TaskStatus.WAITING_FOR_USER:
            self._transition(context, TaskStatus.INTAKE, actor="harness")
        elif context.status in self.TERMINAL:
            if _is_correction(message):
                # A completed item can still receive a user correction. Keep
                # its terminal lifecycle intact, but version the corrected
                # fact so replay and the admin ledger show the new value.
                location = _correction_location(message)
                if location:
                    context.extracted["location"] = location
                    self._sync_facts(context)
                    if context.pending_profile_fact:
                        candidate = dict(context.pending_profile_fact, value=location)
                        event_id = str(self.event_log.list(context.task_id)[-1].event_id)
                        if hasattr(self.task_store, "add_fact"):
                            row = self.task_store.add_fact(
                                namespace="task",
                                namespace_id=str(context.task_id),
                                key=candidate["key"],
                                value=location,
                                status="candidate",
                                source="correction",
                                confidence=0.95,
                                source_event_ids=[event_id],
                                importance=0.9,
                            )
                            candidate["fact_id"] = row.fact_id
                        else:
                            self.fact_ledger.set(
                                context.task_id,
                                candidate["key"],
                                location,
                                source="correction",
                                confidence=0.95,
                                source_event_ids=[event_id],
                                status="candidate",
                            )
                        context.extracted[candidate["key"]] = location
                        context.pending_profile_fact = candidate
                        context.reply = (
                            f"已记录更正地点：{location}。以后需要我记住新的常用区域吗？"
                        )
                    else:
                        context.reply = (
                            f"已记录更正地点：{location}。如需重新办理，请创建新的对话事项。"
                        )
                    self._append_reply(context)
                    self._checkpoint(context)
                    return context
            context.reply = "该事项已经结束，请创建新的对话事项。"
            self._append_reply(context)
            self._checkpoint(context)
            return context
        return self.run(context, on_trace=on_trace)

    def run(
        self, context: TaskContext, *, on_trace: Callable[[RunTrace], None] | None = None
    ) -> TaskContext:
        task_key = str(context.task_id)
        if on_trace:
            self._trace_callbacks[task_key] = on_trace
        try:
            return self._run(context)
        finally:
            self._trace_callbacks.pop(task_key, None)

    def _run(self, context: TaskContext) -> TaskContext:
        steps = tool_calls = 0
        if context.status == TaskStatus.RECEIVED:
            self._transition(context, TaskStatus.INTAKE, actor="harness")
        while context.status not in self.TERMINAL:
            if steps >= self.budget.max_steps or tool_calls >= self.budget.max_tool_calls:
                self._transition(context, TaskStatus.WAITING_FOR_HUMAN, actor="budget")
                context.reply = "自动处理达到预算上限，已转人工复核。"
                break
            steps += 1
            self._record(
                context,
                HarnessPhase.APPEND_EVENT,
                {"step": steps, "source_seq": context.last_event_seq},
            )
            self._phase(context, HarnessPhase.EXTRACT_FACTS, self._sync_facts, context)
            self._phase(context, HarnessPhase.RESOLVE_FACTS, self._resolve_facts, context)
            memories = self._phase(
                context, HarnessPhase.HYDRATE_MEMORY, self._hydrate_memory, context
            )
            compiled = self._phase(
                context, HarnessPhase.COMPRESS_IF_NEEDED, self._compile_context, context, memories
            )
            agent_name = self.workflow.agent_for(context.status)
            self._record(
                context,
                HarnessPhase.SELECT_WORKFLOW_STEP,
                {
                    "agent": agent_name,
                    "skill": context.skill.get("name"),
                    "skill_version": context.skill.get("version"),
                    "skill_manifest": dict(context.skill),
                },
            )
            if not agent_name or agent_name not in self.agents:
                self._transition(context, TaskStatus.FAILED, actor="workflow")
                context.reply = "当前流程没有可执行的处理步骤。"
                break
            proposal = self._phase(
                context,
                HarnessPhase.RUN_AGENT,
                self.agents[agent_name].run,
                compiled,
                agent=agent_name,
            )
            manifest = self.skills.get(context.skill.get("name", "fallback"))
            allowed = set(manifest.allowed_tools)
            self._phase(
                context,
                HarnessPhase.VALIDATE_PROPOSAL,
                self.policy.validate,
                context,
                proposal,
                allowed,
                set(manifest.required_fields),
                agent=agent_name,
            )
            calls = self._phase(
                context,
                HarnessPhase.EXECUTE_TOOLS,
                self._execute_tools,
                context,
                proposal,
                allowed,
                agent=agent_name,
            )
            tool_calls += len(calls)
            if any((not call.success for call in calls)):
                self._event(
                    context,
                    "TOOL_FAILED",
                    {"calls": [call.model_dump(mode="json") for call in calls]},
                    actor="tool",
                )
                self._transition(context, TaskStatus.FAILED, actor="tool")
                context.reply = "工具调用失败，请稍后重试。"
                break
            previous = context.status
            self._phase(
                context,
                HarnessPhase.REDUCE_STATE,
                self.reducer.apply,
                context,
                proposal,
                calls,
                agent=agent_name,
            )
            self._event(
                context,
                "AGENT_PROPOSAL",
                proposal.model_dump(mode="json"),
                actor=agent_name,
                before=previous,
                after=proposal.next_status,
            )
            # Persist facts produced by this proposal before the checkpoint.
            # The next run-loop iteration normally syncs them at its opening,
            # but doing it here keeps the very first checkpoint replayable
            # from the SQL fact ledger as well as from the event log.
            self._sync_facts(context)
            self._transition(context, proposal.next_status, actor=agent_name)
            if context.status == TaskStatus.REVIEWING:
                context.review_round += 1
                if context.review_round > self.budget.max_review_rounds:
                    self._transition(context, TaskStatus.WAITING_FOR_HUMAN, actor="review_budget")
            if context.status == TaskStatus.CREATE_LOCAL_CASE and context.created_case_id:
                self._transition(context, TaskStatus.COMPLETED, actor="create_local_case")
            self._record(context, HarnessPhase.PERSIST_MEMORY, self._persist_memory(context))
            self._checkpoint(context)
        self._append_reply(context)
        self._checkpoint(context)
        return context

    def traces(self, task_id: str) -> list[RunTrace]:
        return list(self._traces.get(str(task_id), []))

    def rebuild_snapshot(self, context: TaskContext | str, source_seq: int | None = None):
        """Replay immutable events into a fresh context before compacting it."""
        if not isinstance(context, TaskContext):
            if self.task_store is None:
                raise KeyError(context)
            loaded = self.task_store.get_task(context)
            if loaded is None:
                raise KeyError(context)
            context = loaded
        events = self.event_log.list(context.task_id)
        latest_seq = events[-1].sequence if events else 0
        cutoff = latest_seq if source_seq is None else min(max(0, source_seq), latest_seq)
        visible = [event for event in events if event.sequence <= cutoff]
        rebuilt = TaskContext(
            task_id=context.task_id,
            user_id=context.user_id,
            session_id=context.session_id,
            user_message=context.user_message,
        )
        successful: set[tuple[str, str]] = set()
        for event in visible:
            payload = event.payload or {}
            kind = str(event.event_type).upper()
            if kind in {"TASK_CREATED", "USER_MESSAGE"}:
                text = payload.get("message", payload.get("text", ""))
                if text:
                    rebuilt.append_message("user", str(text))
            elif kind == "ASSISTANT_MESSAGE" and payload.get("message"):
                rebuilt.append_message("assistant", str(payload["message"]))
                rebuilt.reply = str(payload["message"])
            elif kind == "STATE_TRANSITION":
                target = payload.get("to", event.state_after)
                if target:
                    target = (
                        target.get("value", target.get("name"))
                        if isinstance(target, dict)
                        else target
                    )
                    rebuilt.status = TaskStatus(target)
                    rebuilt.state_history.append(
                        {
                            "from": payload.get("from", event.state_before),
                            "to": str(target),
                            "actor": event.actor,
                        }
                    )
                    if str(target) == TaskStatus.REVIEWING.value:
                        rebuilt.review_round += 1
            elif kind == "AGENT_PROPOSAL":
                self.reducer.apply(rebuilt, AgentResult.model_validate(payload), [])
            elif kind == "TOOL_CALL":
                if payload.get("success") is True:
                    call = ToolCallRecord.model_validate(payload)
                    self.reducer.apply(rebuilt, AgentResult(agent_name="replay"), [call])
                    successful.add(_tool_identity(call.name, call.arguments))
            elif kind == "MEMORY_READ":
                rebuilt.memory_hits = [
                    dict(item) for item in payload.get("hits", []) if isinstance(item, dict)
                ]
            elif kind == "MEMORY_WRITE":
                candidate = payload.get("profile_candidate")
                if isinstance(candidate, dict) and candidate.get("key"):
                    rebuilt.pending_profile_fact = dict(candidate)
            elif kind in {
                "PROFILE_CONFIRMATION",
                "PROFILE_CONFIRMED",
                "PROFILE_REJECTION",
                "PROFILE_REJECTED",
                "PROFILE_FORGET",
                "PROFILE_FORGOTTEN",
            }:
                rebuilt.pending_profile_fact = {}
            elif kind == "HARNESS_PHASE":
                detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
                if payload.get("phase") == HarnessPhase.SELECT_WORKFLOW_STEP.value:
                    manifest = detail.get("skill_manifest")
                    if isinstance(manifest, dict):
                        rebuilt.skill = dict(manifest)
                    elif detail.get("skill"):
                        rebuilt.skill = {
                            "name": detail["skill"],
                            "version": detail.get("skill_version"),
                        }
                if payload.get("phase") == HarnessPhase.CHECKPOINT.value:
                    rebuilt.context_stats = dict(detail.get("stats") or rebuilt.context_stats)
            elif kind == "CHECKPOINT":
                rebuilt.checkpoint_version = max(
                    rebuilt.checkpoint_version, int(payload.get("version", 0) or 0)
                )
        rebuilt.pending_tools = [
            item
            for item in rebuilt.pending_tools
            if _tool_identity(
                item.get("name", ""), item.get("arguments", {}), item.get("idempotency_key")
            )
            not in successful
        ]
        active = self._facts_as_of(context, cutoff)
        rebuilt.extracted = {key: _fact_value(fact) for key, fact in active.items()}
        rebuilt.active_fact_ids = [
            str(getattr(fact, "fact_id", key)) for key, fact in active.items()
        ]
        # A correction can version a task fact without emitting another
        # MEMORY_WRITE event (the original profile prompt remains in the
        # immutable log).  Reconcile the replayed candidate with the fact
        # ledger at this cutoff so a rebuilt snapshot reflects the value that
        # was actually effective then.  The candidate itself is replayed only
        # from visible events, so earlier checkpoints cannot see later fixes.
        candidate = rebuilt.pending_profile_fact
        if candidate:
            fact = active.get(candidate.get("key"))
            if fact is not None:
                rebuilt.pending_profile_fact = {
                    **candidate,
                    "value": _fact_value(fact),
                    "fact_id": str(getattr(fact, "fact_id", candidate.get("fact_id", ""))),
                }
        rebuilt.last_event_seq = cutoff
        rebuilt.checkpoint = (
            f"{context.task_id}:{rebuilt.checkpoint_version}"
            if rebuilt.checkpoint_version
            else None
        )
        ids = [str(event.event_id) for event in visible]
        refs = [str(item.get("memory_id")) for item in rebuilt.memory_hits if item.get("memory_id")]
        return self.context_manager.compact(
            rebuilt, messages=rebuilt.messages, memory_refs=refs, event_ids=ids
        )

    def _facts_as_of(self, context: TaskContext, cutoff: int) -> dict[str, Any]:
        """Select the latest fact revision whose source is visible at cutoff."""
        if hasattr(self.task_store, "facts_as_of"):
            return {row.key: row for row in self.task_store.facts_as_of(context.task_id, cutoff)}
        history = getattr(self.fact_ledger, "history", None)
        if callable(history):
            values = history(context.task_id)
            event_sequences = {
                str(event.event_id): event.sequence
                for event in self.event_log.list(context.task_id)
            }
            active: dict[str, Any] = {}
            grouped: dict[str, list[tuple[int, Any]]] = {}
            for index, fact in enumerate(values):
                refs = set(getattr(fact, "source_event_ids", []) or [])
                if refs and any(
                    event_sequences.get(str(ref)) is None or event_sequences[str(ref)] > cutoff
                    for ref in refs
                ):
                    continue
                grouped.setdefault(fact.key, []).append((index, fact))
            for key, entries in grouped.items():
                entries.sort(
                    key=lambda item: (
                        int(getattr(item[1], "revision", getattr(item[1], "version", 0))),
                        item[0],
                    )
                )
                versions = [item[1] for item in entries]
                latest = versions[-1]
                if getattr(latest, "status", "confirmed") in {
                    "candidate",
                    "confirmed",
                    "superseded",
                }:
                    active[key] = latest
                elif getattr(latest, "status", "") == "rejected" and getattr(
                    latest, "supersedes_fact_id", None
                ):
                    previous = next(
                        (
                            item
                            for item in reversed(versions[:-1])
                            if getattr(item, "fact_id", None) == latest.supersedes_fact_id
                            and getattr(item, "status", "")
                            in {"candidate", "confirmed", "superseded"}
                        ),
                        None,
                    )
                    if previous is not None:
                        active[key] = previous
            return active
        return self.fact_ledger.all(context.task_id)

    def _compile_context(
        self, context: TaskContext, memories: list[dict[str, Any]]
    ) -> CompiledContext:
        selection = self.skills.select(context.user_message, context.extracted)
        current = context.skill.get("name")
        if (
            current
            and current != "fallback"
            and self.skills.get(current).enabled
            and (selection.name == "fallback")
        ):
            selection = type(selection)(
                name=current, version=self.skills.get(current).version, reason="task skill retained"
            )
        manifest = self.skills.get(selection.name)
        context.skill = {
            **manifest.model_dump(),
            "name": selection.name,
            "matched_trigger": selection.matched_trigger,
        }
        ids = [str(item.event_id) for item in self.event_log.list(context.task_id)]
        snapshot = self.context_manager.compact(
            context,
            messages=context.messages,
            memory_refs=[item["memory_id"] for item in memories],
            event_ids=ids,
        )
        context.summary, context.context_stats, context.memory_hits = (
            snapshot.summary,
            snapshot.stats.model_dump(mode="json"),
            memories,
        )
        context.rendered_context, context.snapshot_id = (
            snapshot.rendered_context,
            str(snapshot.snapshot_id),
        )
        context.recent_event_ids, context.summary_ids = (
            snapshot.recent_event_ids,
            snapshot.summary_ids,
        )
        context.active_fact_ids = snapshot.active_fact_ids
        return self.context_manager.compile(
            context,
            memories=memories,
            active_facts=snapshot.facts,
            messages=context.messages,
            summary=snapshot.summary,
        )

    def _sync_facts(self, context: TaskContext) -> None:
        source = "correction" if _is_correction(context.user_message) else "user"
        events = self.event_log.list(context.task_id)
        source_event = next(
            (
                event
                for event in reversed(events)
                if str(event.event_type).upper() in {"USER_MESSAGE", "TASK_CREATED"}
            ),
            events[-1] if events else None,
        )
        event_ids = [str(source_event.event_id)] if source_event else []
        for key, value in context.extracted.items():
            if value is not None and (not key.startswith("_")):
                self.fact_ledger.set(
                    context.task_id,
                    key,
                    value,
                    source=source,
                    confidence=0.95,
                    source_event_ids=event_ids,
                    status="candidate",
                )

    def _resolve_facts(self, context: TaskContext) -> None:
        active = self.fact_ledger.all(context.task_id)
        context.active_fact_ids = [
            str(getattr(fact, "fact_id", key)) for key, fact in active.items()
        ]
        for key, fact in active.items():
            context.extracted[key] = fact.value

    def _hydrate_memory(self, context: TaskContext) -> list[dict[str, Any]]:
        hits = []
        for scope, reason in (
            (f"task:{context.task_id}", "task"),
            (f"session:{_session_key(context)}", "session"),
            (f"user:{context.user_id}", "profile"),
        ):
            hits.extend(
                (
                    (item, reason)
                    for item in self.memory.search(scope, context.user_message, limit=2)
                )
            )
        result, seen = ([], set())
        for item, reason in hits:
            identifier = str(item.memory_id)
            if identifier in seen:
                continue
            seen.add(identifier)
            result.append(
                {
                    "memory_id": identifier,
                    "scope_id": item.scope_id,
                    "layer": item.layer,
                    "key": item.key,
                    "content": item.content,
                    "score": item.score,
                    "reason": reason,
                    "source": item.metadata.get("source", reason),
                }
            )
        if hasattr(self.task_store, "recall_memory"):
            for item in self.task_store.recall_memory(
                user_id=context.user_id,
                session_id=_session_key(context),
                task_id=context.task_id,
                query=context.user_message,
                top_k=3,
            ):
                identifier = str(item["memory_id"])
                if identifier not in seen:
                    result.append({**item, "memory_id": identifier, "reason": "repository recall"})
                    seen.add(identifier)
        self._event(
            context,
            "MEMORY_READ",
            {
                "hits": [
                    {"memory_id": item["memory_id"], "reason": item.get("reason", "")}
                    for item in result
                ]
            },
            actor="harness",
        )
        return result[:3]

    def _persist_memory(self, context: TaskContext) -> dict[str, Any]:
        if context.status not in {TaskStatus.COMPLETED, TaskStatus.CREATE_LOCAL_CASE}:
            return {"written": 0}
        content = f"{context.extracted.get('category', '')}；{context.extracted.get('location', '')}；{context.routing.get('department', '')}"
        self.memory.put(
            f"session:{_session_key(context)}",
            MemoryLayer.EPISODIC,
            f"task:{context.task_id}",
            content,
            metadata={"task_id": str(context.task_id)},
            importance=0.7,
        )
        candidate_hint = (
            {"key": "preferred_area", "value": context.extracted["location"]}
            if context.extracted.get("location") and not context.pending_profile_fact
            else None
        )
        completion_reply = self._case_completion_reply(
            context,
            include_profile_prompt=bool(candidate_hint),
        )
        memory_event = self._event(
            context,
            "MEMORY_WRITE",
            {
                "scope": f"session:{_session_key(context)}",
                "key": f"task:{context.task_id}",
                "profile_candidate": candidate_hint,
                "reply": completion_reply,
            },
            actor="harness",
        )
        if context.extracted.get("location") and (not context.pending_profile_fact):
            candidate = {"key": "preferred_area", "value": context.extracted["location"]}
            if hasattr(self.task_store, "add_fact"):
                row = self.task_store.add_fact(
                    namespace="task",
                    namespace_id=str(context.task_id),
                    key=candidate["key"],
                    value=candidate["value"],
                    status="candidate",
                    source="user",
                    confidence=0.9,
                    importance=0.9,
                    source_event_ids=[str(memory_event.event_id)],
                )
                candidate["fact_id"] = row.fact_id
            else:
                row = self.fact_ledger.set(
                    context.task_id,
                    candidate["key"],
                    candidate["value"],
                    source="user",
                    confidence=0.9,
                    status="candidate",
                    source_event_ids=[str(memory_event.event_id)],
                    importance=0.9,
                )
                candidate["fact_id"] = str(getattr(row, "fact_id", ""))
            context.extracted[candidate["key"]] = candidate["value"]
            context.active_fact_ids = [
                str(getattr(item, "fact_id", key))
                for key, item in self.fact_ledger.all(context.task_id).items()
            ]
            context.pending_profile_fact = candidate
        context.reply = completion_reply
        return {
            "written": 1,
            "scope": f"session:{_session_key(context)}",
            "profile_candidate": bool(context.pending_profile_fact),
        }

    @staticmethod
    def _case_completion_reply(
        context: TaskContext, *, include_profile_prompt: bool
    ) -> str:
        """Explain the saved demo case and give the user an actionable next step."""
        category = str(context.extracted.get("category") or "城市公共服务问题")
        location = str(context.extracted.get("location") or "未提供")
        department = str(context.routing.get("department") or "综合受理")
        priority = str(context.routing.get("priority") or "普通")
        case_id = str(context.created_case_id or context.task_id)
        lines = [
            f"事项已创建（{case_id}）。",
            "已在 CivicNexus 本地演示系统中保存；当前版本不会提交到真实城市管理平台。",
            "",
            f"识别问题：{category}",
            f"地点：{location}",
            f"处理方向：{department}",
            f"优先级：{priority}",
            "当前状态：已完成本地演示登记。",
            "",
            "你接下来可以：",
            "1. 保留事项编号，便于后续人工核对。",
            "2. 如需补充发生时间、影响范围或现场情况，请点击“新对话”，并带上事项编号重新说明。",
            "3. 如现场存在触电、燃气泄漏、深水或其他人身危险，请先远离现场并联系当地紧急服务。",
        ]
        if include_profile_prompt:
            lines.extend(
                [
                    "",
                    f"需要我记住“{location}”作为你的常用区域吗？回复“记住”或“不需要”。",
                ]
            )
        return "\n".join(lines)

    def _handle_profile_intent(self, context: TaskContext, message: str) -> bool:
        candidate = context.pending_profile_fact
        # A correction carrying an address/position is task data, not a yes/no
        # answer to the profile prompt. Let the normal intake/terminal path
        # version the fact before asking about long-term memory again.
        if candidate and _is_correction(message) and _correction_location(message):
            return False
        if "忘记" in message:
            if hasattr(self.task_store, "forget_profile_fact"):
                self.task_store.forget_profile_fact(
                    context.user_id,
                    candidate.get("key", "preferred_area"),
                    fact_id=candidate.get("fact_id"),
                    task_id=context.task_id,
                    session_id=context.session_id,
                )
            else:
                self.memory.delete(
                    f"user:{context.user_id}", "preferred_area", layer=MemoryLayer.SEMANTIC
                )
                self._event(context, "PROFILE_FORGOTTEN", {"key": "preferred_area"}, actor="user")
            context.reply, context.pending_profile_fact = ("已忘记你的常用区域。", {})
            self._append_reply(context)
            return True
        if not candidate:
            return False
        normalized = message.strip().lower()
        negative = any((word in normalized for word in ("否", "不", "不要", "拒绝", "no")))
        affirmative = not negative and any(
            (word in normalized for word in ("是", "好", "可以", "同意", "yes", "记住"))
        )
        if not affirmative and (not negative):
            return False
        if affirmative:
            if hasattr(self.task_store, "confirm_profile_fact"):
                self.task_store.confirm_profile_fact(
                    context.user_id,
                    candidate["key"],
                    fact_id=candidate.get("fact_id"),
                    task_id=context.task_id,
                    session_id=context.session_id,
                    event_payload={
                        "key": candidate["key"],
                        "value": candidate["value"],
                        "confirmed": True,
                    },
                )
            else:
                self.memory.put(
                    f"user:{context.user_id}",
                    MemoryLayer.SEMANTIC,
                    candidate["key"],
                    str(candidate["value"]),
                    metadata={"confirmation": "explicit"},
                    importance=0.9,
                )
                self._event(context, "PROFILE_CONFIRMED", candidate, actor="user")
            context.reply = (
                f"好的，已将“{candidate['value']}”写入你的用户画像。事项仍已保留（{context.created_case_id or context.task_id}）。"
                "如需补充发生时间、影响范围或现场情况，请点击“新对话”并带上事项编号重新说明。"
            )
        else:
            if hasattr(self.task_store, "reject_profile_fact"):
                self.task_store.reject_profile_fact(
                    context.user_id,
                    candidate["key"],
                    fact_id=candidate.get("fact_id"),
                    task_id=context.task_id,
                    session_id=context.session_id,
                )
            else:
                self._event(context, "PROFILE_REJECTED", candidate, actor="user")
            context.reply = (
                f"好的，本次不会写入你的用户画像。事项仍已保留（{context.created_case_id or context.task_id}）。"
                "如需补充发生时间、影响范围或现场情况，请点击“新对话”并带上事项编号重新说明。"
            )
        context.pending_profile_fact = {}
        self._append_reply(context)
        return True

    def _execute_tools(
        self, context: TaskContext, proposal: AgentResult, allowed: set[str]
    ) -> list[ToolCallRecord]:
        tool_context = ToolContext(
            task_id=str(context.task_id), user_id=context.user_id, session_id=context.session_id
        )
        calls = [
            self.tools.execute(item, tool_context, allowed_tools=allowed)
            for item in proposal.tool_requests
        ]
        for call in calls:
            self._event(context, "TOOL_CALL", call.model_dump(mode="json"), actor="tool")
        return calls

    def _checkpoint(self, context: TaskContext) -> None:
        context.checkpoint_version += 1
        context.checkpoint = f"{context.task_id}:{context.checkpoint_version}"
        self._event(context, "CHECKPOINT", {"version": context.checkpoint_version}, actor="harness")
        self._record(
            context,
            HarnessPhase.CHECKPOINT,
            {
                "version": context.checkpoint_version,
                "source_seq": context.last_event_seq,
                "stats": context.context_stats,
            },
        )
        ids = [str(item.event_id) for item in self.event_log.list(context.task_id)]
        snapshot = self.context_manager.save_checkpoint(
            context, messages=context.messages, event_ids=ids
        )
        context.summary, context.context_stats = (
            snapshot.summary,
            snapshot.stats.model_dump(mode="json"),
        )
        context.rendered_context, context.snapshot_id = (
            snapshot.rendered_context,
            str(snapshot.snapshot_id),
        )
        context.recent_event_ids, context.summary_ids = (
            snapshot.recent_event_ids,
            snapshot.summary_ids,
        )
        context.active_fact_ids = snapshot.active_fact_ids
        if self.task_store:
            self.task_store.save_task(
                context, user_id=context.user_id, session_id=context.session_id
            )
        if self.checkpoint_store:
            self.checkpoint_store.save_context(
                context.task_id, context, ttl_seconds=self.checkpoint_ttl
            )
            if hasattr(self.checkpoint_store, "save_snapshot"):
                self.checkpoint_store.save_snapshot(
                    context.task_id, snapshot, ttl_seconds=self.checkpoint_ttl
                )

    def _append_reply(self, context: TaskContext) -> None:
        if context.reply and (
            not context.messages or context.messages[-1].content != context.reply
        ):
            context.append_message("assistant", context.reply)
            self._event(
                context, "ASSISTANT_MESSAGE", {"message": _redact(context.reply)}, actor="assistant"
            )

    def _transition(self, context: TaskContext, target: TaskStatus, *, actor: str) -> None:
        before = context.status
        assert_transition(before, target)
        context.status = target
        context.state_history.append({"from": before.value, "to": target.value, "actor": actor})
        self._event(
            context,
            "STATE_TRANSITION",
            {"from": before.value, "to": target.value},
            actor=actor,
            before=before,
            after=target,
        )

    def _event(
        self,
        context: TaskContext,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor: str,
        before: TaskStatus | None = None,
        after: TaskStatus | None = None,
    ):
        event = self.event_log.append(
            context.task_id,
            event_type,
            payload,
            actor=actor,
            state_before=before.value if before else None,
            state_after=after.value if after else None,
        )
        context.last_event_seq = event.sequence
        return event

    def _phase(
        self,
        context: TaskContext,
        phase: HarnessPhase,
        function: Callable[..., Any],
        *args: Any,
        agent: str | None = None,
    ) -> Any:
        started = time.perf_counter()
        try:
            result = function(*args)
        except Exception as exc:
            self._record(
                context,
                phase,
                {"failed": True, "error_type": type(exc).__name__, "error": _redact(str(exc))},
                agent=agent,
                duration=(time.perf_counter() - started) * 1000,
            )
            raise
        self._record(
            context,
            phase,
            _phase_detail(result),
            agent=agent,
            duration=(time.perf_counter() - started) * 1000,
        )
        return result

    def _record(
        self,
        context: TaskContext,
        phase: HarnessPhase,
        detail: dict[str, Any],
        *,
        agent: str | None = None,
        duration: float = 0.0,
    ) -> None:
        bucket = self._traces.setdefault(str(context.task_id), [])
        trace = RunTrace(
            sequence=len(bucket) + 1,
            phase=phase,
            status=context.status,
            agent=agent,
            detail=detail,
            duration_ms=round(duration, 3),
        )
        bucket.append(trace)
        context.trace.append(trace.model_dump(mode="json"))
        self._event(context, "HARNESS_PHASE", trace.model_dump(mode="json"), actor="harness")
        callback = self._trace_callbacks.get(str(context.task_id))
        if callback:
            callback(trace)


def _phase_detail(result: Any) -> dict[str, Any]:
    if isinstance(result, AgentResult):
        return {
            "next_status": result.next_status,
            "confidence": result.confidence,
            "llm_fallback": bool(result.llm_error),
            "reason": result.reason,
        }
    if isinstance(result, CompiledContext):
        return {
            "stats": result.context_stats,
            "facts": list(result.active_facts),
            "memory_hits": len(result.memories),
        }
    if isinstance(result, list):
        return {"count": len(result)}
    return {}


def _is_correction(text: str) -> bool:
    return any((marker in text for marker in ("纠正", "更正", "改为", "不是", "说错", "应为")))


def _is_greeting(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？!?、,.]+", "", text).lower()
    return normalized in {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey"}


def _session_key(context: TaskContext) -> str:
    return f"{context.user_id}:{context.session_id}"


def _correction_location(text: str) -> str | None:
    match = re.search(
        r"(?:地址|位置)(?:改为|更正为|是|为)?\s*[:：]?\s*([^，,。；;!?！？\n]+)", text
    )
    return match.group(1).strip() if match else None


def _fact_value(fact: Any) -> Any:
    return getattr(fact, "value", getattr(fact, "value_json", fact))


def _tool_identity(
    name: str, arguments: Any, idempotency_key: str | None = None
) -> tuple[str, str]:
    args = arguments if isinstance(arguments, dict) else {}
    return str(name), str(
        idempotency_key
        or args.get("idempotency_key")
        or json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    )


def _redact(text: str) -> str:
    text = re.sub("\\b1\\d{10}\\b", "<phone>", text)
    return re.sub("[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}", "<email>", text)


__all__ = ["AgentRuntime", "PolicyGate", "RunBudget", "RunTrace", "StateReducer"]
