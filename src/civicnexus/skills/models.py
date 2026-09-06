from __future__ import annotations

from pydantic import BaseModel, Field


class SkillManifest(BaseModel):
    name: str
    version: str
    enabled: bool = True
    priority: int = 0
    triggers: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    route_defaults: dict[str, str] = Field(default_factory=dict)
    prompt: str = ""


class SkillSelection(BaseModel):
    name: str
    version: str
    matched_trigger: str | None = None
    reason: str
