from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from inspect import Parameter, signature
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from civicnexus.domain.context import TaskContext
from civicnexus.evaluation.gold import GOLD_CASES, GOLD_DATASET, GoldCase


class Prediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    category: str | None = None
    location: str | None = None
    skill: str | None = None
    department: str | None = None


class Metric(BaseModel):
    name: str
    label: str
    value: float
    target: float
    passed: bool
    unit: str | None = None
    comparison: Literal["gte", "lte"] = "gte"


class CaseScore(BaseModel):
    case_id: str
    prediction: Prediction
    matches: dict[str, bool]
    exact_match: bool


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dataset: str = GOLD_DATASET
    sample_count: int = 0
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: list[Metric] = Field(default_factory=list)
    cases: list[CaseScore] = Field(default_factory=list)
    passed: bool = False
    predictor: str = "deterministic_rules"


Predictor = Callable[[GoldCase], Any]
SCORED_FIELDS = ("category", "location", "skill", "department")
DEFAULT_TARGETS = {
    "category_accuracy": 1.0,
    "location_accuracy": 1.0,
    "skill_selection_accuracy": 1.0,
    "routing_accuracy": 1.0,
    "category_macro_f1": 0.9,
    "overall_exact_match": 1.0,
}
_LABELS = {
    "category_accuracy": "事项分类准确率",
    "location_accuracy": "地点抽取准确率",
    "skill_selection_accuracy": "Skill 选择准确率",
    "routing_accuracy": "部门路由准确率",
    "category_macro_f1": "事项分类 Macro-F1",
    "overall_exact_match": "全字段完全匹配率",
}


class EvaluationRunner:
    def __init__(
        self,
        cases: Sequence[GoldCase | Mapping[str, Any]] = GOLD_CASES,
        predictor: Predictor | None = None,
        *,
        targets: Mapping[str, float] | None = None,
    ) -> None:
        self.cases = tuple(
            case if isinstance(case, GoldCase) else GoldCase.model_validate(case) for case in cases
        )
        if not self.cases:
            raise ValueError("evaluation gold set must not be empty")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation gold case IDs must be unique")
        self.predictor = predictor or default_predictor
        self.targets = {**DEFAULT_TARGETS, **dict(targets or {})}

    def run(self, predictor: Predictor | None = None) -> EvaluationReport:
        active = predictor or self.predictor
        scores = [self._score(case, _invoke(active, case)) for case in self.cases]
        metrics = self._metrics(scores)
        return EvaluationReport(
            dataset=GOLD_DATASET,
            sample_count=len(scores),
            metrics=metrics,
            cases=scores,
            passed=all(item.passed for item in metrics),
            predictor=getattr(active, "__name__", active.__class__.__name__),
        )

    evaluate = run

    @staticmethod
    def _score(case: GoldCase, raw: Any) -> CaseScore:
        prediction = _coerce_prediction(raw)
        matches = {
            field: getattr(prediction, field) == getattr(case, field) for field in SCORED_FIELDS
        }
        return CaseScore(
            case_id=case.case_id,
            prediction=prediction,
            matches=matches,
            exact_match=all(matches.values()),
        )

    def _metrics(self, scores: Sequence[CaseScore]) -> list[Metric]:
        total = len(scores)
        values = {
            "category_accuracy": _accuracy(scores, "category"),
            "location_accuracy": _accuracy(scores, "location"),
            "skill_selection_accuracy": _accuracy(scores, "skill"),
            "routing_accuracy": _accuracy(scores, "department"),
            "category_macro_f1": _macro_f1(
                [case.category for case in self.cases],
                [score.prediction.category or "<missing>" for score in scores],
            ),
            "overall_exact_match": sum(score.exact_match for score in scores) / total,
        }
        return [
            Metric(
                name=name,
                label=_LABELS[name],
                value=round(value, 6),
                target=self.targets.get(name, 1.0),
                passed=value >= self.targets.get(name, 1.0),
            )
            for name, value in values.items()
        ]


def _accuracy(scores: Sequence[CaseScore], field: str) -> float:
    return sum(score.matches[field] for score in scores) / len(scores)


def _invoke(predictor: Predictor, case: GoldCase) -> Any:
    try:
        first = next(iter(signature(predictor).parameters.values()))
    except (StopIteration, TypeError, ValueError):
        first = None
    wants_text = bool(
        first
        and first.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        and (
            first.name.lower() in {"message", "text", "query", "prompt"}
            or first.annotation in (str, "str")
        )
    )
    return predictor(case.message if wants_text else case)


def _coerce_prediction(raw: Any) -> Prediction:
    if isinstance(raw, Prediction):
        return raw
    if isinstance(raw, TaskContext):
        return Prediction(
            category=raw.extracted.get("category"),
            location=raw.extracted.get("location"),
            skill=(raw.skill or {}).get("name"),
            department=(raw.routing or {}).get("department"),
        )
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if not isinstance(raw, Mapping):
        raise TypeError(
            "evaluation predictor must return a mapping, Pydantic model, or TaskContext"
        )
    patch = raw.get("patch")
    patch = patch if isinstance(patch, Mapping) else {}
    extracted = raw.get("extracted") or raw.get("facts") or patch.get("extracted") or {}
    routing = raw.get("routing") or {}
    skill = raw.get("skill_name") or raw.get("skill")
    skill = skill.get("name") if isinstance(skill, Mapping) else skill
    return Prediction(
        category=raw.get("category") or _value(extracted, "category"),
        location=raw.get("location") or _value(extracted, "location"),
        skill=skill,
        department=raw.get("department") or _value(routing, "department"),
    )


def _value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def default_predictor(case: GoldCase | str) -> Prediction:
    message = case.message if isinstance(case, GoldCase) else case
    category, skill, department = _classify(message)
    return Prediction(
        category=category, location=_location(message), skill=skill, department=department
    )


def _classify(message: str) -> tuple[str | None, str | None, str | None]:
    rules = (
        ("污水与下水道", "drainage", "排水部门", ("污水", "下水道", "排水", "积水", "内涝")),
        ("垃圾处理", "garbage", "环卫部门", ("垃圾", "清运", "回收")),
        ("道路维护", "road", "道路交通部门", ("道路", "路面", "坑洞", "交通")),
    )
    return next(
        (
            (category, skill, department)
            for category, skill, department, terms in rules
            if any(term in message for term in terms)
        ),
        (None, None, None),
    )


def _location(message: str) -> str | None:
    match = re.search(r"(?:地址|位置)(?:是|为)?\s*[:：]?\s*([^，,。；;!?！？\n]+)", message)
    return match.group(1).strip() if match else None


def _macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    labels = set(gold) | set(predicted)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(a == label == p for a, p in zip(gold, predicted, strict=True))
        fp = sum(a != label and p == label for a, p in zip(gold, predicted, strict=True))
        fn = sum(a == label and p != label for a, p in zip(gold, predicted, strict=True))
        precision, recall = (
            (tp / (tp + fp) if tp + fp else 0.0),
            (tp / (tp + fn) if tp + fn else 0.0),
        )
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


__all__ = [
    "DEFAULT_TARGETS",
    "CaseScore",
    "EvaluationReport",
    "EvaluationRunner",
    "Metric",
    "Prediction",
    "default_predictor",
]
