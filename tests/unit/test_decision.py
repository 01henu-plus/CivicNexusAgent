from uuid import uuid4

from civicnexus.agents.decision import DecisionEngine
from civicnexus.agents.intake_agent import IntakeAgent, _category, _location
from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult
from civicnexus.llm.client import FakeLLMClient


def _context(message: str) -> CompiledContext:
    return CompiledContext(
        task_id=uuid4(),
        user_id="user",
        session_id="session",
        status=TaskStatus.INTAKE,
        latest_message=message,
    )


def test_llm_weight_can_replace_local_intake_decision() -> None:
    llm = FakeLLMClient(
        responder=lambda *_: {
            "patch": {
                "extracted": {
                    "category": "排水、积水与洪涝",
                    "location": "我家这条路",
                }
            },
            "missing_fields": [],
            "next_status": "CASE_ANALYSIS",
            "confidence": 0.95,
            "message_zh": "已识别积水地点，正在查询相关案例。",
        }
    )
    local = AgentResult(
        agent_name="intake",
        patch={"extracted": {"category": None, "location": None, "urgency": "普通"}},
        missing_fields=["category", "location"],
        next_status=TaskStatus.WAITING_FOR_USER,
        confidence=0.78,
        message_zh="请补充事项类型和具体地点。",
    )

    result = DecisionEngine(llm).decide(
        agent_name="intake",
        context=_context("我家这条路积水了"),
        local=local,
        prompt="",
    )

    assert result.patch["extracted"] == {
        "category": "排水、积水与洪涝",
        "location": "我家这条路",
        "urgency": "普通",
    }
    assert result.missing_fields == []
    assert result.next_status == TaskStatus.CASE_ANALYSIS
    assert result.message_zh == "已识别积水地点，正在查询相关案例。"


def test_intake_uses_llm_to_fill_natural_location_and_category() -> None:
    llm = FakeLLMClient(
        responder=lambda *_: {
            "patch": {
                "extracted": {
                    "category": "排水、积水与洪涝",
                    "location": "附近这条路",
                }
            },
            "missing_fields": [],
            "next_status": "CASE_ANALYSIS",
            "confidence": 0.9,
        }
    )
    result = IntakeAgent(DecisionEngine(llm)).run(_context("我家这条路积水了"))

    assert result.patch["extracted"]["category"] == "排水、积水与洪涝"
    assert result.patch["extracted"]["location"] == "附近这条路"
    assert result.missing_fields == ["location"]
    assert result.next_status == TaskStatus.WAITING_FOR_USER


def test_intake_does_not_skip_missing_required_fact_when_llm_status_is_wrong() -> None:
    llm = FakeLLMClient(
        responder=lambda *_: {
            "patch": {"extracted": {"category": "排水、积水与洪涝"}},
            "missing_fields": [],
            "next_status": "CASE_ANALYSIS",
            "confidence": 0.95,
        }
    )
    result = IntakeAgent(DecisionEngine(llm)).run(_context("积水了"))

    assert result.missing_fields == ["location"]
    assert result.next_status == TaskStatus.WAITING_FOR_USER


def test_local_intake_handles_common_road_and_location_phrases() -> None:
    assert _category("我家这条路积水了") == "排水、积水与洪涝"
    assert _location("我家这条路积水了") == "我家这条路"
    assert _location("小区门口有积水") == "小区门口"


def test_leaking_water_is_recognized_but_relative_location_requires_address() -> None:
    llm = FakeLLMClient(
        responder=lambda *_: {
            "patch": {"extracted": {"category": "污水与下水道", "location": "我家附近"}},
            "missing_fields": ["category", "location"],
            "next_status": "CASE_ANALYSIS",
            "confidence": 0.95,
            "message_zh": "请补充事项类型和地点。",
        }
    )

    result = IntakeAgent(DecisionEngine(llm)).run(_context("我家附近漏水了"))

    assert _category("我家附近漏水了") == "污水与下水道"
    assert _location("我家附近漏水了") == "我家附近"
    assert result.patch["extracted"]["category"] == "污水与下水道"
    assert result.patch["extracted"]["location"] == "我家附近"
    assert result.missing_fields == ["location"]
    assert result.next_status == TaskStatus.WAITING_FOR_USER
    assert "事项类型" not in (result.message_zh or "")
    assert "具体地址" in (result.message_zh or "")


def test_exact_address_allows_leaking_water_intake_to_continue() -> None:
    result = IntakeAgent(DecisionEngine(FakeLLMClient())).run(_context("人民路 1 号漏水了"))

    assert result.patch["extracted"]["category"] == "污水与下水道"
    assert result.patch["extracted"]["location"] == "人民路 1 号"
    assert result.missing_fields == []
    assert result.next_status == TaskStatus.CASE_ANALYSIS
