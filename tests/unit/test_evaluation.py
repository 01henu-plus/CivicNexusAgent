from __future__ import annotations

from types import SimpleNamespace

from civicnexus.evaluation import GOLD_CASES, EvaluationRunner, Prediction, runtime_metrics


def test_fixed_gold_runner_is_reproducible_and_scores_baseline() -> None:
    report = EvaluationRunner().run()

    assert report.dataset == "civicnexus-gold-v1"
    assert report.sample_count == len(GOLD_CASES) == 6
    assert report.passed is True
    values = {metric.name: metric.value for metric in report.metrics}
    assert values["category_accuracy"] == 1.0
    assert values["location_accuracy"] == 1.0
    assert values["category_macro_f1"] == 1.0
    assert values["overall_exact_match"] == 1.0


def test_runner_exposes_partial_failures_instead_of_hiding_them() -> None:
    def wrong_location(case):
        return Prediction(
            category=case.category,
            location="错误地点",
            skill=case.skill,
            department=case.department,
        )

    report = EvaluationRunner(predictor=wrong_location).run()
    values = {metric.name: metric.value for metric in report.metrics}
    assert values["category_accuracy"] == 1.0
    assert values["location_accuracy"] == 0.0
    assert values["overall_exact_match"] == 0.0
    assert report.passed is False
    assert all(not item.matches["location"] for item in report.cases)


def test_runner_accepts_task_context_predictions() -> None:
    def context_prediction(case):
        from civicnexus.domain.context import TaskContext

        return TaskContext(
            user_message=case.message,
            extracted={"category": case.category, "location": case.location},
            skill={"name": case.skill},
            routing={"department": case.department},
        )

    report = EvaluationRunner(predictor=context_prediction).run()
    assert report.passed is True


def test_runtime_metrics_read_events_and_context_without_placeholder_perfect_scores() -> None:
    class FakeRepository:
        def list_tasks(self, limit=500):
            return [
                SimpleNamespace(
                    id="task-1",
                    status="COMPLETED",
                    checkpoint_version=1,
                    context_json={
                        "extracted": {"category": "垃圾处理", "location": "城北小区"},
                        "context_stats": {"reduction_ratio": 0.6},
                        "memory_hits": [{"key": "last_case"}],
                        "skill": {"name": "garbage"},
                    },
                )
            ]

        def list_events(self, task_id):
            return [
                SimpleNamespace(
                    state_before="RECEIVED",
                    state_after="INTAKE",
                    event_type="STATE_TRANSITION",
                    payload_redacted={},
                ),
                SimpleNamespace(
                    state_before="INTAKE",
                    state_after="ROUTING",  # invalid edge: should lower validity
                    event_type="STATE_TRANSITION",
                    payload_redacted={},
                ),
                SimpleNamespace(
                    state_before=None,
                    state_after=None,
                    event_type="TOOL_CALL",
                    payload_redacted={"success": True},
                ),
            ]

    result = runtime_metrics(FakeRepository())
    values = {item["name"]: item["value"] for item in result["metrics"]}
    assert values["completion_rate"] == 1.0
    assert values["valid_transition_rate"] == 0.5
    assert values["compression_ratio"] == 0.6
    assert values["tool_success_rate"] == 1.0
    assert values["critical_fact_recall"] == 1.0
