from __future__ import annotations

from pathlib import Path

import yaml

from civicnexus.domain.enums import TaskStatus


DEFAULT_WORKFLOW: dict[TaskStatus, str] = {
    TaskStatus.RECEIVED: "intake",
    TaskStatus.INTAKE: "intake",
    TaskStatus.CASE_ANALYSIS: "case_analysis",
    TaskStatus.ROUTING: "routing",
    TaskStatus.REVIEWING: "review",
}


class Workflow:
    """Small declarative state-to-agent map with a safe built-in default."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.steps = dict(DEFAULT_WORKFLOW)
        self.reload()

    def reload(self) -> None:
        if not self.path or not self.path.exists():
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        for key, value in (raw.get("steps") or {}).items():
            self.steps[TaskStatus(key)] = str(value["agent"] if isinstance(value, dict) else value)

    def agent_for(self, status: TaskStatus) -> str | None:
        self.reload()
        return self.steps.get(status)
