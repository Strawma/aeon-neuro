"""Exercise frozen job plans, seed routing, recovery and scored coverage."""

import json
from copy import deepcopy

import numpy as np
import pytest
from aeon.classification import BaseClassifier
from aeon.classification.dummy import DummyClassifier

from aeon_neuro.experiments import (
    plan_classification_benchmark,
    run_classification_benchmark,
)
from aeon_neuro.model_selection import plan_classification_evaluation
from aeon_neuro.model_selection.tests.test_evaluation import make_dataset, profile


class AuditedClassifier(BaseClassifier):
    """Fail one training attempt without adding state to estimator parameters."""

    _tags = {"capability:multivariate": True}
    fail_next = False

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("Temporary training failure")
        return self

    def _predict(self, X):
        return np.repeat(self.classes_[0], len(X))

    def _predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


@pytest.fixture
def inputs():
    """Three people share the same two held-out-session folds across models."""
    dataset = make_dataset()
    configuration = profile("cross_session")
    configuration["metrics"] = ["accuracy", "balanced_accuracy"]
    configuration.pop("positive_label")
    evaluation = plan_classification_evaluation(
        dataset,
        configuration,
        disjoint_by=[["recording_id", "trial"]],
        source_fields=["recording_id", "start_sample", "stop_sample"],
    )
    spec = {
        "format_version": 1,
        "id": "generated",
        "problems": ["imagery"],
        "classifiers": ["prior", "audited"],
        "estimator_seeds": [42],
    }
    problems = {
        "imagery": {
            "dataset": dataset,
            "evaluation": evaluation,
            "provenance": {"scope": "generated regression"},
        }
    }
    classifiers = {
        "prior": DummyClassifier(strategy="prior"),
        "audited": AuditedClassifier(),
    }
    return spec, problems, classifiers


def test_json_plan_order_identity_and_no_fit(inputs, tmp_path, monkeypatch):
    """Freeze deterministic jobs without invoking estimator fitting."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning fitted a classifier")

    monkeypatch.setattr(DummyClassifier, "fit", forbidden)
    monkeypatch.setattr(AuditedClassifier, "fit", forbidden)
    spec, problems, classifiers = inputs
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    plan = plan_classification_benchmark(path, problems, classifiers)
    assert plan == json.loads(json.dumps(plan))
    assert plan == plan_classification_benchmark(*inputs)
    assert len(plan["jobs"]) == 12
    assert [j["classifier"] for j in plan["jobs"]] == ["prior"] * 6 + ["audited"] * 6
    assert plan["estimators"]["prior"]["42"]["parameters"]["random_state"] == 42
    assert classifiers["prior"].random_state is None
    spec["id"] = "changed"
    assert plan["specification"]["id"] == "generated"


@pytest.mark.parametrize(
    "field,value",
    [
        ("format_version", True),
        ("format_version", 2),
        ("extra", 1),
        ("problems", []),
        ("problems", ["missing"]),
        ("classifiers", ["prior", "prior"]),
        ("classifiers", ["../outside"]),
        ("estimator_seeds", [True]),
        ("estimator_seeds", [-1]),
        ("estimator_seeds", [2**32]),
    ],
)
def test_invalid_specifications_fail(inputs, field, value):
    """Reject ambiguous selections, unsafe names and invalid seeds."""
    spec, problems, classifiers = inputs
    spec[field] = value
    with pytest.raises(ValueError):
        plan_classification_benchmark(spec, problems, classifiers)


@pytest.mark.parametrize(
    "change", ["samples", "labels", "assignment", "estimator", "job", "provenance"]
)
def test_changed_inputs_rejected_before_output(inputs, tmp_path, change):
    """Bind arrays, definitions, assignments and estimator parameters before writing."""
    spec, problems, classifiers = inputs
    plan = plan_classification_benchmark(*inputs)
    prepared = problems["imagery"]
    if change == "samples":
        prepared["dataset"].X[0, 0, 0] += 1
    elif change == "labels":
        prepared["dataset"].y[0] = "right"
    elif change == "assignment":
        prepared["evaluation"]["folds"][0]["split"]["partitions"]["test"][
            "indices"
        ].reverse()
    elif change == "estimator":
        classifiers["prior"].strategy = "uniform"
    elif change == "job":
        plan["jobs"][0]["seed"] = 99
    else:
        prepared["provenance"]["scope"] = "changed"
    target = tmp_path / "suite"
    with pytest.raises(ValueError):
        run_classification_benchmark(plan, problems, classifiers, target)
    assert not target.exists()


def test_unsupported_metric_is_explicit(inputs):
    """Report unsupported scoring instead of omitting declared metrics."""
    spec, problems, classifiers = inputs
    configuration = profile("cross_session")
    dataset = problems["imagery"]["dataset"]
    problems["imagery"]["evaluation"] = plan_classification_evaluation(
        dataset,
        configuration,
        disjoint_by=[],
        source_fields=["recording_id", "start_sample", "stop_sample"],
    )
    with pytest.raises(ValueError, match="currently support"):
        plan_classification_benchmark(spec, problems, classifiers)


def test_serial_roundtrip_metrics_resume_and_corrupt_summary(inputs, tmp_path):
    """Score people equally and verify completed summaries before reuse."""
    pytest.importorskip("tsml_eval")
    spec, problems, classifiers = inputs
    spec["classifiers"] = ["prior"]
    plan = plan_classification_benchmark(*inputs)
    target = tmp_path / "suite"
    report = run_classification_benchmark(plan, problems, classifiers, target)
    assert report["coverage"] == {
        "planned": 6,
        "completed": 6,
        "failed": 0,
        "pending": 0,
    }
    assert report["status"] == "complete"
    for metric in report["summaries"][0]["metrics"].values():
        assert metric["mean"] == 0.5
        assert metric["participant_sample_sd"] == 0
        assert len(metric["participants"]) == 3
    assert (
        run_classification_benchmark(plan, problems, classifiers, target, resume=True)
        == report
    )
    with pytest.raises(FileExistsError):
        run_classification_benchmark(plan, problems, classifiers, target)
    altered = deepcopy(report)
    altered["summaries"][0]["metrics"]["accuracy"]["mean"] = 1
    receipt = target / "benchmark.json"
    receipt.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="summary changed"):
        run_classification_benchmark(plan, problems, classifiers, target, resume=True)
    assert json.loads(receipt.read_text()) == altered


def test_failed_job_keeps_coverage_and_recovery_verifies_predictions(inputs, tmp_path):
    """Retain failed coverage and reject damaged predictions after recovery."""
    pytest.importorskip("tsml_eval")
    spec, problems, classifiers = inputs
    spec["classifiers"] = ["audited"]
    plan = plan_classification_benchmark(*inputs)
    target = tmp_path / "suite"
    AuditedClassifier.fail_next = True
    with pytest.raises(RuntimeError, match="Temporary"):
        run_classification_benchmark(plan, problems, classifiers, target)
    failed = json.loads((target / "benchmark.json").read_text())
    assert failed["coverage"] == {
        "planned": 6,
        "completed": 0,
        "failed": 1,
        "pending": 5,
    }
    complete = run_classification_benchmark(
        plan, problems, classifiers, target, resume=True
    )
    assert complete["coverage"]["completed"] == 6
    receipt = json.loads((target / "job-000000/run.json").read_text())
    predictions = target / "job-000000" / receipt["artifacts"]["predictions"]["path"]
    predictions.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        run_classification_benchmark(plan, problems, classifiers, target, resume=True)


def test_seeds_expand_jobs_and_do_not_pool_as_people(inputs):
    """Keep seed repetitions explicit in the planned job matrix."""
    spec, problems, classifiers = inputs
    spec["estimator_seeds"] = [42, 7]
    result = plan_classification_benchmark(spec, problems, classifiers)
    assert len(result["jobs"]) == 24
    assert result["estimators"]["prior"]["7"]["parameters"]["random_state"] == 7
