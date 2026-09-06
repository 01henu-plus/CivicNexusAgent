from civicnexus.agents.base import Agent
from civicnexus.agents.decision import DecisionEngine
from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult, ToolRequest


class CaseAnalysisAgent(Agent):
    name = "case_analysis"

    def __init__(self, decisions: DecisionEngine) -> None:
        self.decisions = decisions

    def run(self, context: CompiledContext) -> AgentResult:
        local = AgentResult(
            agent_name=self.name,
            tool_requests=[
                ToolRequest(
                    name="case_search", arguments={"query": context.latest_message, "top_k": 3}
                )
            ],
            next_status=TaskStatus.ROUTING,
            confidence=0.9,
            message_zh="历史案例分析完成，正在生成分派建议。",
            reason="retrieve top three cases",
        )
        return self.decisions.decide(
            agent_name=self.name,
            context=context,
            local=local,
            prompt=context.skill.get("prompt", ""),
        )
