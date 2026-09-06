from __future__ import annotations
import json
from pathlib import Path
from civicnexus.skills.models import SkillManifest, SkillSelection


class SkillRegistry:
    """Declarative, reloadable skill registry; manifests never execute code."""

    def __init__(self, directory: str | Path, *, fallback: SkillManifest | None = None) -> None:
        self.directory = Path(directory)
        self.fallback = fallback or SkillManifest(
            name="fallback",
            version="1.0",
            priority=-1,
            required_fields=["category", "location"],
            allowed_tools=["case_search", "validate_address", "memory_lookup", "create_local_case"],
            route_defaults={"department": "综合受理", "priority": "普通"},
        )
        self._skills: dict[str, SkillManifest] = {}
        self._mtimes: dict[Path, int] = {}
        self.reload()

    @property
    def skills(self) -> tuple[SkillManifest, ...]:
        return tuple(self._skills.values())

    def reload(self) -> bool:
        files = sorted(
            [
                *self.directory.glob("*.yaml"),
                *self.directory.glob("*.yml"),
                *self.directory.glob("*.json"),
            ]
        )
        changed = {path: path.stat().st_mtime_ns for path in files if path.exists()}
        if changed == self._mtimes and self._skills:
            return False
        loaded: dict[str, SkillManifest] = {}
        for path in files:
            raw = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".json"
                else self._yaml(path)
            )
            skill = SkillManifest.model_validate(raw)
            if skill.name in loaded:
                raise ValueError(f"Duplicate skill name: {skill.name}")
            loaded[skill.name] = skill
        self._skills = loaded
        self._mtimes = changed
        return True

    def select(self, text: str, facts: dict[str, object] | None = None) -> SkillSelection:
        self.reload()
        lowered = f"{text} {' '.join(str(value) for value in (facts or {}).values())}".lower()
        candidates = [
            (skill, trigger)
            for skill in self._skills.values()
            if skill.enabled
            for trigger in skill.triggers
            if trigger.lower() in lowered
        ]
        if not candidates:
            skill = self.fallback
            return SkillSelection(
                name=skill.name, version=skill.version, reason="no trigger matched"
            )
        skill, trigger = sorted(
            candidates, key=lambda item: (-item[0].priority, -len(item[1]), item[0].name)
        )[0]
        return SkillSelection(
            name=skill.name,
            version=skill.version,
            matched_trigger=trigger,
            reason="trigger matched",
        )

    def get(self, name: str) -> SkillManifest:
        return self._skills.get(name, self.fallback)

    @staticmethod
    def _yaml(path: Path) -> object:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
