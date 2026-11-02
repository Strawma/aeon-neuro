"""Check real result round trips, protocol guards and conservative recovery."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression

from aeon_neuro.datasets import build_classification_dataset
from aeon_neuro.experiments import _classification as adapter
from aeon_neuro.experiments import run_classification_experiment

pytest.importorskip("tsml_eval")
filelock = pytest.importorskip("filelock")


class AuditedClassifier(BaseClassifier):
    """Observe actual fitted arrays and exercise failure/decision boundaries."""

    _tags = {"capability:multivariate": True}
    fits = []
    fail_next = False
    source_to_mutate = None

    def __init__(self, probability="normal"):
        self.probability = probability
        super().__init__()

    def _fit(self, X, y):
        type(self).fits.append((X.copy(), y.copy()))
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("Temporary training failure")
        if type(self).source_to_mutate is not None:
            type(self).source_to_mutate.write_text("changed")
        return self

    def _predict(self, X):
        return np.repeat(self.classes_[-1], len(X))

    def _predict_proba(self, X):
        if self.probability == "invalid":
            return np.tile([2.0, -1.0], (len(X), 1))
        return np.tile([0.75, 0.25], (len(X), 1))


@pytest.fixture(autouse=True)
def reset_audit():
    """Keep failure simulation and observed training scope independent per test."""
    AuditedClassifier.fits = []
    AuditedClassifier.fail_next = False
    AuditedClassifier.source_to_mutate = None


@pytest.fixture
def dataset():
    """Use actual public cases with an explicit within-person session policy."""
    metadata = [
        {
            "recording_id": session,
            "channel_names": ["C3", "C4"],
            "sampling_frequency": 4,
        }
        for session in ("T", "E")
    ]
    rows = [
        {
            "case_id": f"{session}-{trial}",
            "recording_id": session,
            "start_sample": trial * 6,
            "stop_sample": trial * 6 + 6,
            "label": "left" if trial % 2 == 0 else "right",
            "group_id": session,
            "participant_id": "P1",
            "session": session,
            "source_identity": session,
            "support_start": trial * 6,
            "support_stop": trial * 6 + 6,
        }
        for session in ("T", "E")
        for trial in range(4)
    ]
    return build_classification_dataset(np.arange(96).reshape(2, 2, 24), metadata, rows)


def run(dataset, output, **kwargs):
    """Supply every protocol choice explicitly, permitting targeted overrides."""
    options = {
        "dataset_name": "GeneratedImagery",
        "provenance": {"question": "left/right", "score_unit": "person-session"},
        "disjoint_by": [("participant_id", "session")],
        "source_fields": ("source_identity", "support_start", "support_stop"),
    }
    classifier = kwargs.pop("classifier", AuditedClassifier())
    partitions = kwargs.pop("partitions", {"train": [1, 0, 3, 2], "test": [5, 4, 7, 6]})
    options.update(kwargs)
    return run_classification_experiment(
        dataset, partitions, classifier, output, **options
    )


def test_roundtrip_fit_scope_class_meanings_and_input_preservation(dataset, tmp_path):
    """Use the real tsml writer/reader and retain original prediction-row identity."""
    before = deepcopy(dataset)
    model = AuditedClassifier()
    report = run(dataset, tmp_path / "run", classifier=model)
    assert report["status"] == "complete"
    assert report["class_order"] == ["left", "right"]
    assert report["accuracy"] == 0.5
    assert not model.is_fitted
    assert len(AuditedClassifier.fits) == 1
    np.testing.assert_array_equal(AuditedClassifier.fits[0][0], dataset.X[[1, 0, 3, 2]])
    np.testing.assert_array_equal(AuditedClassifier.fits[0][1], [1, 0, 1, 0])
    mapping = json.loads(
        (tmp_path / "run" / report["artifacts"]["case_map"]["path"]).read_text()
    )
    assert [row["case_id"] for row in mapping["rows"]] == ["E-1", "E-0", "E-3", "E-2"]
    assert not list((tmp_path / "run").rglob("trainResample*.csv"))
    np.testing.assert_array_equal(dataset.X, before.X)
    np.testing.assert_array_equal(dataset.y, before.y)
    assert dataset.cases == before.cases


def test_explicit_same_person_and_held_out_person_policies(dataset, tmp_path):
    """Writing results must not silently replace the required independence unit."""
    with pytest.raises(ValueError, match="crosses"):
        run(dataset, tmp_path / "invalid", disjoint_by=[("participant_id",)])
    assert not AuditedClassifier.fits
    for row in dataset.cases[4:]:
        row["participant_id"] = "P2"
    assert (
        run(dataset, tmp_path / "valid", disjoint_by=[("participant_id",)])["status"]
        == "complete"
    )


def test_filter_support_overlap_is_checked_separately(dataset, tmp_path):
    """Distinct trial identities cannot hide overlapping original filter support."""
    for row in dataset.cases:
        row["source_identity"] = "shared-original"
    with pytest.raises(ValueError, match="overlap"):
        run(dataset, tmp_path / "run")
    assert not AuditedClassifier.fits


def test_inner_scope_cannot_include_outer_test_cases(dataset, tmp_path):
    """Validate against the supplied scope rather than all available arrays."""
    with pytest.raises(ValueError, match="cover"):
        run(dataset, tmp_path / "bad", eligible_indices=[0, 1, 2, 3])
    report = run(
        dataset,
        tmp_path / "good",
        eligible_indices=[0, 1, 2, 3],
        partitions={"train": [0, 1], "test": [2, 3]},
        disjoint_by=[("case_id",)],
    )
    assert len(AuditedClassifier.fits[-1][0]) == 2
    assert report["identity"]["split"]["eligible_indices"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "change", ["X", "case", "split", "estimator", "provenance", "runtime"]
)
def test_resume_rejects_stale_identity(dataset, tmp_path, monkeypatch, change):
    """A plausible completed CSV cannot substitute for a different experiment."""
    output = tmp_path / "run"
    run(dataset, output)
    options = {"resume": True}
    if change == "X":
        dataset.X[0, 0, 0] += 1
    elif change == "case":
        dataset.cases[0]["review"] = "different"
    elif change == "split":
        options["partitions"] = {"train": [0, 1, 2, 3], "test": [4, 5, 6, 7]}
    elif change == "estimator":
        options["classifier"] = AuditedClassifier(probability="other")
    elif change == "provenance":
        options["provenance"] = {"question": "changed"}
    else:
        original = adapter.version
        monkeypatch.setattr(
            adapter, "version", lambda name: original(name) + ".changed"
        )
    with pytest.raises(ValueError, match="identity differs"):
        run(dataset, output, **options)
    assert len(AuditedClassifier.fits) == 1


def test_intact_completed_resume_does_not_refit(dataset, tmp_path):
    """Explicit resume verifies saved artifacts and returns the same receipt."""
    output = tmp_path / "run"
    first = run(dataset, output)
    with pytest.raises(FileExistsError):
        run(dataset, output)
    assert run(dataset, output, resume=True) == first
    assert len(AuditedClassifier.fits) == 1


@pytest.mark.parametrize("state", ["failed", "running"])
def test_failed_or_interrupted_resume_retains_previous_attempt(
    dataset, tmp_path, state
):
    """A released lock lets matching work restart without trusting partial CSVs."""
    output = tmp_path / "run"
    AuditedClassifier.fail_next = True
    with pytest.raises(RuntimeError, match="Temporary"):
        run(dataset, output)
    path = output / "run.json"
    first = json.loads(path.read_text())
    assert first["status"] == "failed"
    marker = output / "attempts" / first["attempts"][0] / "partial.txt"
    marker.write_text("retained")
    first["status"] = state
    path.write_text(json.dumps(first))
    report = run(dataset, output, resume=True)
    assert report["status"] == "complete" and len(report["attempts"]) == 2
    assert marker.read_text() == "retained"
    diagnostic = json.loads((marker.parent / "attempt.json").read_text())
    assert diagnostic["status"] == "failed"
    assert diagnostic["error"]["type"] == "RuntimeError"


def test_corrupt_complete_artifact_is_not_overwritten(dataset, tmp_path):
    """Corruption is surfaced instead of silently replacing previous evidence."""
    output = tmp_path / "run"
    report = run(dataset, output)
    path = output / report["artifacts"]["predictions"]["path"]
    path.write_text("broken")
    with pytest.raises(ValueError, match="artifact changed"):
        run(dataset, output, resume=True)
    assert path.read_text() == "broken" and len(AuditedClassifier.fits) == 1


@pytest.mark.parametrize("field", ["class_order", "accuracy", "format_version"])
def test_completed_summary_corruption_is_rejected(dataset, tmp_path, field):
    """A valid CSV cannot certify changed class meanings or reported scores."""
    output = tmp_path / "run"
    report = run(dataset, output)
    report[field] = {
        "class_order": ["changed"],
        "accuracy": 0.123,
        "format_version": 99,
    }[field]
    (output / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="summary changed|receipt format"):
        run(dataset, output, resume=True)
    assert len(AuditedClassifier.fits) == 1


def test_busy_experiment_does_not_start_another_fit(dataset, tmp_path):
    """Use a real competing filesystem lock, not a mocked concurrency flag."""
    output = tmp_path / "run"
    with filelock.FileLock(str(tmp_path / "run.lock")):
        with pytest.raises(filelock.Timeout):
            run(dataset, output, resume=True)
    assert not AuditedClassifier.fits and not output.exists()


def test_source_identity_and_changes_during_fit(dataset, tmp_path):
    """Check acquired files again after fitting, not merely when a run starts."""
    source = tmp_path / "source"
    source.write_text("original")
    manifest = {source: hashlib.sha256(source.read_bytes()).hexdigest()}
    AuditedClassifier.source_to_mutate = source
    with pytest.raises(ValueError, match="Source checksum changed"):
        run(dataset, tmp_path / "run", source_manifest=manifest)
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "kind",
    ["ragged", "nonfinite", "sklearn", "invalid_probabilities", "unstable_provenance"],
)
def test_unsupported_inputs_fail_without_complete_receipt(dataset, tmp_path, kind):
    """Reject implicit flattening, invalid probabilities and unstable identities."""
    options = {}
    if kind == "ragged":
        dataset.X = list(dataset.X)
    elif kind == "nonfinite":
        dataset.X = dataset.X.astype(float)
        dataset.X[0, 0, 0] = np.nan
    elif kind == "sklearn":
        options["classifier"] = LogisticRegression()
    elif kind == "invalid_probabilities":
        options["classifier"] = AuditedClassifier(probability="invalid")
    else:
        options["provenance"] = {"unstable": object()}
    output = tmp_path / "run"
    with pytest.raises((TypeError, ValueError, AssertionError)):
        run(dataset, output, **options)
    if (output / "run.json").exists():
        assert json.loads((output / "run.json").read_text())["status"] == "failed"
