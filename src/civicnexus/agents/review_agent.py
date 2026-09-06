from civicnexus.agents.base import Agent
from civicnexus.agents.decision import DecisionEngine
from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult, ToolRequest


class ReviewAgent(Agent):
    name = "review"

    def __init__(self, decisions: DecisionEngine, *, max_rounds: int = 2) -> None:
        self.decisions = decisions
        self.max_rounds = max_rounds

    def run(self, context: CompiledContext) -> AgentResult:
        missing = [key for key in ("category", "location") if not context.active_facts.get(key)]
        if missing:
            local = AgentResult(
                agent_name=self.name,
                missing_fields=missing,
                next_status=TaskStatus.WAITING_FOR_USER,
                confidence=1.0,
                message_zh="复核发现信息不完整，请补充具体地点。",
                reason="required facts missing",
            )
        elif not context.routing:
            local = AgentResult(
                agent_name=self.name,
                next_status=TaskStatus.ROUTING,
                confidence=0.8,
                message_zh="分派信息不完整，正在重新分析。",
                reason="routing result missing",
            )
        else:
            args = {
                "category": context.active_facts["category"],
                "location": context.active_facts["location"],
                "department": context.routing.get("department", "综合受理"),
                "priority": context.routing.get("priority", "普通"),
                "summary": context.latest_message,
            }
            local = AgentResult(
                agent_name=self.name,
                tool_requests=[
                    ToolRequest(
                        name="create_local_case",
                        arguments=args,
                        idempotency_key=f"{context.task_id}:create_local_case",
                    )
                ],
                next_status=TaskStatus.CREATE_LOCAL_CASE,
                confidence=0.96,
                message_zh="事项已通过复核，正在创建本地演示事项。",
                reason="required facts and routing verified",
            )
        return self.decisions.decide(
            agent_name=self.name,
            context=context,
            local=local,
            prompt=context.skill.get("prompt", ""),
        )
