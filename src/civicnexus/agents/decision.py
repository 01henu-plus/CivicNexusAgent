from __future__ import annotations

import json

from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult, ToolRequest
from civicnexus.llm import LLMClient, LLMError
from civicnexus.llm.schemas import LLMProposal


class DecisionEngine:
    """Runs both deterministic policy and the LLM, then fuses their proposals."""

    def __init__(
        self, llm: LLMClient, *, local_weight: float = 0.3, llm_weight: float = 0.7
    ) -> None:
        if local_weight <= 0 or llm_weight <= 0:
            raise ValueError("Decision weights must be positive.")
        self.llm = llm
        self.local_weight = local_weight
        self.llm_weight = llm_weight

    def decide(
        self,
        *,
        agent_name: str,
        context: CompiledContext,
        local: AgentResult,
        prompt: str,
    ) -> AgentResult:
        try:
            model = self.llm.complete(
                system=(
                    f"你是 CivicNexus 的 {agent_name}。只返回符合 schema 的 JSON。"
                    "不得绕过必填字段、状态机或工具权限。" + prompt
                ),
                user=_model_input(context),
                schema=LLMProposal,
            )
        except LLMError as exc:
            local.local_result = _proposal_dump(local)
            local.llm_error = str(exc)
            return local
        local_score = local.confidence * self.local_weight
        llm_score = model.confidence * self.llm_weight
        prefer_model = model.confidence > 0 and llm_score > local_score
        patch = _merge_patch(local.patch, model.patch, prefer_model=prefer_model)
        model_status = _status(model.next_status)
        model_controls_intake = agent_name == "intake" and prefer_model
        next_status = (
            model_status
            if model_controls_intake and model_status
            else local.next_status or model_status
        )
        missing_fields = (
            list(model.missing_fields) if model_controls_intake else list(local.missing_fields)
        )
        if model_controls_intake:
            extracted = patch.get("extracted")
            extracted = extracted if isinstance(extracted, dict) else {}
            missing_fields = [
                key for key in ("category", "location") if not extracted.get(key)
            ]
            next_status = (
                TaskStatus.WAITING_FOR_USER if missing_fields else TaskStatus.CASE_ANALYSIS
            )
        return AgentResult(
            agent_name=agent_name,
            accepted=local.accepted,
            patch=patch,
            missing_fields=missing_fields,
            evidence=local.evidence or model.evidence,
            tool_requests=local.tool_requests
            or [ToolRequest.model_validate(item) for item in model.tool_requests],
            next_status=next_status,
            confidence=round(local_score + llm_score, 4),
            message_zh=(
                model.message_zh if prefer_model and model.message_zh else local.message_zh
            ),
            reason=(
                f"local={local.reason}; llm={model.reason}; "
                f"selected={'llm' if prefer_model else 'local'}"
            ),
            local_result=_proposal_dump(local),
            llm_result=model.model_dump(),
        )


def _status(value: str | TaskStatus | None) -> TaskStatus | None:
    if isinstance(value, TaskStatus):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return TaskStatus(value)
    except ValueError:
        return None


def _merge_patch(
    local: dict[str, object], model: dict[str, object], *, prefer_model: bool
) -> dict[str, object]:
    """Merge model intent fields without dropping facts extracted by the other source."""
    merged = dict(local)
    for key, value in model.items():
        if key == "extracted" and isinstance(value, dict):
            local_extracted = merged.get(key)
            if isinstance(local_extracted, dict):
                extracted = dict(local_extracted)
            else:
                extracted = {}
            meaningful = {name: item for name, item in value.items() if item not in (None, "")}
            if prefer_model:
                extracted.update(meaningful)
            else:
                for name, item in meaningful.items():
                    extracted.setdefault(name, item)
            merged[key] = extracted
        elif prefer_model:
            merged[key] = value
        else:
            merged.setdefault(key, value)
    return merged


def _proposal_dump(proposal: AgentResult) -> dict[str, object]:
    return proposal.model_dump(exclude={"local_result", "llm_result"}, mode="json")


def _model_input(context: CompiledContext) -> str:
    """Send only the fields an agent needs for its current decision."""
    recent = [{"role": item.role, "content": item.content} for item in context.recent_messages[-4:]]
    payload = {
        "status": context.status,
        "latest_message": context.latest_message,
        "recent_messages": recent,
        "summary": context.summary,
        "active_facts": context.active_facts,
        "missing_fields": list(context.missing_fields),
        "evidence": list(context.evidence[:3]),
        "routing": context.routing,
        "review": context.review,
        "skill": {
            "name": context.skill.get("name"),
            "version": context.skill.get("version"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
