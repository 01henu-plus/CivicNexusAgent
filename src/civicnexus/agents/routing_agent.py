from civicnexus.agents.base import Agent
from civicnexus.agents.decision import DecisionEngine
from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult


class RoutingAgent(Agent):
    name = "routing"

    def __init__(self, decisions: DecisionEngine) -> None:
        self.decisions = decisions

    def run(self, context: CompiledContext) -> AgentResult:
        defaults = context.skill.get("route_defaults", {})
        routing = {
            "department": defaults.get("department", "综合受理"),
            "priority": "紧急"
            if context.active_facts.get("urgency") == "紧急"
            else defaults.get("priority", "普通"),
            "skill": context.skill.get("name", "fallback"),
            "reason": "根据事项类型、历史案例和动态 Skill 分派",
        }
        local = AgentResult(
            agent_name=self.name,
            patch={"routing": routing},
            next_status=TaskStatus.REVIEWING,
            confidence=0.92 if context.skill.get("name") != "fallback" else 0.75,
            message_zh="分派建议已生成，正在复核。",
            reason="skill route defaults",
        )
        return self.decisions.decide(
            agent_name=self.name,
            context=context,
            local=local,
            prompt=context.skill.get("prompt", ""),
        )
