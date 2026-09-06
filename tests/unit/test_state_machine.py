from civicnexus.domain.enums import TaskStatus
from civicnexus.runtime.engine import AgentRuntime
from civicnexus.runtime.state_machine import assert_transition, can_transition
from civicnexus.skills import SkillRegistry
from civicnexus.tools import ToolRegistry


def test_normal_task_transition_is_allowed() -> None:
    assert can_transition(TaskStatus.RECEIVED, TaskStatus.INTAKE)


def test_terminal_task_cannot_transition() -> None:
    assert not can_transition(TaskStatus.COMPLETED, TaskStatus.INTAKE)


def test_invalid_transition_raises() -> None:
    try:
        assert_transition(TaskStatus.RECEIVED, TaskStatus.ROUTING)
    except ValueError:
        return
    raise AssertionError("Expected invalid transition to raise ValueError")


def test_greeting_returns_without_running_business_agents() -> None:
    runtime = AgentRuntime([], tools=ToolRegistry(), skills=SkillRegistry("configs/skills"))
    context = runtime.start(user_id="u1", session_id="s1", message="你好！")
    assert context.status == TaskStatus.WAITING_FOR_USER
    assert "城市公共服务问题" in context.reply
