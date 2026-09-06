from civicnexus.domain.enums import TaskStatus


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.RECEIVED: {TaskStatus.INTAKE, TaskStatus.WAITING_FOR_USER, TaskStatus.CANCELLED},
    TaskStatus.INTAKE: {TaskStatus.WAITING_FOR_USER, TaskStatus.CASE_ANALYSIS, TaskStatus.FAILED},
    TaskStatus.WAITING_FOR_USER: {TaskStatus.INTAKE, TaskStatus.CANCELLED},
    TaskStatus.CASE_ANALYSIS: {TaskStatus.ROUTING, TaskStatus.FAILED},
    TaskStatus.ROUTING: {TaskStatus.REVIEWING, TaskStatus.FAILED},
    TaskStatus.REVIEWING: {
        TaskStatus.ROUTING,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.WAITING_FOR_HUMAN,
        TaskStatus.CREATE_LOCAL_CASE,
        TaskStatus.FAILED,
    },
    TaskStatus.WAITING_FOR_HUMAN: {TaskStatus.REVIEWING, TaskStatus.CANCELLED},
    TaskStatus.CREATE_LOCAL_CASE: {TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.FAILED: set(),
}


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def assert_transition(current: TaskStatus, target: TaskStatus) -> None:
    if not can_transition(current, target):
        raise ValueError(f"Invalid task transition: {current} -> {target}")
