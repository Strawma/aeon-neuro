"""Exercise evaluation semantics with multiple people, sessions and trial windows."""

from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import build_classification_dataset
from aeon_neuro.model_selection import (
    plan_classification_evaluation,
    validate_classification_evaluation,
)


def make_dataset():
    """Three people, two sessions, eight trials and two windows per trial."""
    metadata, signals, cases = [], [], []
    for person in range(3):
        for session in range(2):
            recording = f"P{person}-S{session}"
            metadata.append(
                {
                    "recording_id": recording,
                    "channel_names": ["C3", "C4"],
                    "sampling_frequency": 2,
                }
            )
            signals.append(np.arange(64, dtype=float).reshape(2, 32))
            for trial in range(8):
                for window in range(2):
                    start = trial * 4 + window * 2
                    cases.append(
                        {
                            "case_id": f"{recording}-T{trial}-W{window}",
                            "recording_id": recording,
                            "start_sample": start,
                            "stop_sample": start + 2,
                            "label": "left" if trial % 2 == 0 else "right",
                            "group_id": f"{recording}-T{trial}",
                            "person": f"P{person}",
                            "session": str(session),
                            "trial": trial,
                        }
                    )
    return build_classification_dataset(signals, metadata, cases)


@pytest.fixture
def dataset():
    """Provide fresh mutable cases for each planning test."""
    return make_dataset()


def profile(protocol="within_session"):
    """Use explicit field mappings, seed, positive class and averaging policy."""
    splitters = {
        "within_session": {
            "name": "stratified_kfold",
            "n_splits": 2,
            "random_state": 42,
        },
        "cross_subject": {"name": "leave_one_participant_out"},
        "cross_session": {"name": "leave_one_session_out"},
    }
    return {
        "format_version": 1,
        "id": protocol,
        "protocol": protocol,
        "identities": {
            "participant": ["person"],
            "session": ["session"],
            "trial": ["trial"],
        },
        "splitter": splitters[protocol],
        "metrics": ["accuracy", "balanced_accuracy", "roc_auc"],
        "positive_label": "right",
        "aggregation": "equal_participant",
    }


def plan(dataset, configuration=None, **kwargs):
    """Retain the original source grid and required trial rule."""
    return plan_classification_evaluation(
        dataset,
        profile() if configuration is None else configuration,
        source_fields=("recording_id", "start_sample", "stop_sample"),
        disjoint_by=kwargs.pop("disjoint_by", [("recording_id", "trial")]),
        **kwargs,
    )


@pytest.mark.parametrize(
    "protocol,n_folds",
    [("within_session", 12), ("cross_subject", 3), ("cross_session", 6)],
)
def test_presets_have_correct_scope_and_complete_test_coverage(
    dataset, protocol, n_folds
):
    """Every case is tested once and the named independence rule holds."""
    result = plan(dataset, profile(protocol))
    assert len(result["folds"]) == n_folds
    tested = []
    for fold in result["folds"]:
        train, test = (
            fold["split"]["partitions"][p]["indices"] for p in ("train", "test")
        )
        tested.extend(test)
        left = {dataset.cases[i]["person"] for i in train}
        right = {dataset.cases[i]["person"] for i in test}
        if protocol == "cross_subject":
            assert left.isdisjoint(right) and len(right) == 1
        else:
            assert left == right and len(right) == 1
            train_sessions = {dataset.cases[i]["session"] for i in train}
            test_sessions = {dataset.cases[i]["session"] for i in test}
            assert len(test_sessions) == 1
            if protocol == "cross_session":
                assert train_sessions.isdisjoint(test_sessions)
            else:
                assert train_sessions == test_sessions

        def trial(i):
            return (dataset.cases[i]["recording_id"], dataset.cases[i]["trial"])

        assert {trial(i) for i in train}.isdisjoint({trial(i) for i in test})
    assert sorted(tested) == list(range(len(dataset.y)))
    assert validate_classification_evaluation(dataset, result) == result


def test_explicit_holdout_preserves_saved_order_and_scope(dataset):
    """Replay an existing holdout without invoking a new random split."""
    config = profile()
    config["splitter"] = {"name": "explicit"}
    scope = list(range(16))
    assignment = {
        "train": list(reversed(range(8))),
        "test": list(reversed(range(8, 16))),
    }
    result = plan(dataset, config, eligible_indices=scope, partitions=[assignment])
    assert (
        result["folds"][0]["split"]["partitions"]["train"]["indices"]
        == assignment["train"]
    )
    assert validate_classification_evaluation(dataset, result) == result
    with pytest.raises(ValueError, match="exactly one holdout"):
        plan(
            dataset, config, eligible_indices=scope, partitions=[assignment, assignment]
        )


def test_inner_scope_and_case_discovery_order(dataset):
    """Canonical case IDs drive generation within the supplied training scope."""
    result = plan(dataset, eligible_indices=list(range(12)))
    assert all(
        set(f["split"]["eligible_indices"]) == set(range(12)) for f in result["folds"]
    )
    original = plan(dataset)
    reordered = deepcopy(dataset)
    permutation = np.arange(len(dataset.y))[::-1]
    for name in ("X", "y", "groups"):
        setattr(reordered, name, getattr(dataset, name)[permutation])
    reordered.cases = [dataset.cases[i] for i in permutation]
    other = plan(reordered)
    for a, b in zip(original["folds"], other["folds"]):
        assert (
            a["split"]["partitions"]["test"]["case_ids"]
            == b["split"]["partitions"]["test"]["case_ids"]
        )
    with pytest.raises(ValueError):
        validate_classification_evaluation(reordered, original)


def test_problem_constraints_and_processing_support_cannot_be_weakened(dataset):
    """A convenient preset cannot override stricter scientific requirements."""
    with pytest.raises(ValueError, match="crosses split"):
        plan(dataset, disjoint_by=[("person",)])
    for case in dataset.cases:
        case.update(start_sample=0, stop_sample=4)
    with pytest.raises(ValueError, match="overlap"):
        plan(dataset)


def test_planning_does_not_read_signal_values(dataset):
    """Case identities and labels suffice; arrays are bound by experiment receipts."""

    class UnreadSignals:
        def __len__(self):
            return len(dataset.cases)

        def __getitem__(self, key):
            raise AssertionError("Planner must not access signal values")

    dataset.X = UnreadSignals()
    result = plan(dataset)
    assert len(result["folds"]) == 12


@pytest.mark.parametrize(
    "change",
    [
        "person",
        "session",
        "trial",
        "mixed_trial",
        "few_trials",
        "class_missing",
        "one_person",
        "one_session",
    ],
)
def test_unresolved_or_insufficient_cases_fail_before_fitting(dataset, change):
    """Reject missing identities and infeasible scopes instead of relaxing rules."""
    config, scope = profile(), None
    if change in {"person", "session", "trial"}:
        dataset.cases[0][change] = "n/a"
    elif change == "mixed_trial":
        dataset.y[0] = dataset.cases[0]["label"] = "right"
    elif change == "few_trials":
        config["splitter"]["n_splits"] = 5
    elif change == "class_missing":
        for i in range(16):
            dataset.y[i] = dataset.cases[i]["label"] = "left"
    elif change == "one_person":
        config, scope = profile("cross_subject"), list(range(32))
    else:
        config, scope = profile("cross_session"), list(range(16))
    with pytest.raises(ValueError):
        plan(dataset, config, eligible_indices=scope)


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "version",
        "protocol",
        "splitter",
        "seed",
        "folds",
        "positive",
        "metric",
        "averaging",
    ],
)
def test_malformed_profile_never_silently_defaults(dataset, change):
    """Fail unknown settings, ambiguous metrics and invalid splitter parameters."""
    config = profile()
    if change == "unknown":
        config["typo"] = True
    elif change == "version":
        config["format_version"] = True
    elif change == "protocol":
        config["protocol"] = "automatic"
    elif change == "splitter":
        config["splitter"] = {"name": "leave_one_participant_out"}
    elif change == "seed":
        config["splitter"]["random_state"] = None
    elif change == "folds":
        config["splitter"]["n_splits"] = True
    elif change == "positive":
        del config["positive_label"]
    elif change == "metric":
        config["metrics"] = ["made_up"]
    else:
        config["aggregation"] = "pool_all_trials"
    with pytest.raises(ValueError):
        plan(dataset, config)


@pytest.mark.parametrize(
    "change", ["case", "label", "metadata", "fold", "profile", "runtime", "checksum"]
)
def test_saved_plan_detects_mutation(dataset, change):
    """A seed or valid individual split cannot replace the saved full plan."""
    result = plan(dataset)
    if change == "case":
        dataset.cases[0]["review"] = "changed"
    elif change == "label":
        dataset.y[0] = dataset.cases[0]["label"] = "right"
    elif change == "metadata":
        dataset.metadata["reference"] = "different"
    elif change == "fold":
        result["folds"].pop()
    elif change == "profile":
        result["profile"]["id"] = "different"
    elif change == "runtime":
        result["runtime"]["scikit-learn"] = "different"
    else:
        result["plan_id"] = "different"
    with pytest.raises(ValueError):
        validate_classification_evaluation(dataset, result)
