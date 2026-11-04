"""Translate small evaluation profiles into fully checked case assignments."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from aeon_neuro.datasets import ClassificationDataset, validate_classification_split
from aeon_neuro.datasets._classification_split import (
    _case_indices,
    _field_names,
    _identity,
)


def _json(value):
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {key: _json(item) for key, item in value.items()}
    raise ValueError("Evaluation records must contain ordinary JSON values")


def _digest(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _key(case, fields):
    # JSON preserves integer/string identity distinctions and avoids ambiguous
    # delimiter concatenation when participant or trial names contain punctuation.
    return json.dumps([_identity(case.get(f), f) for f in fields])


def _fields(*roles):
    return list(dict.fromkeys(field for role in roles for field in role))


def _profile(profile, labels):
    required = {
        "format_version",
        "id",
        "protocol",
        "identities",
        "splitter",
        "metrics",
        "aggregation",
    }
    if (
        not isinstance(profile, dict)
        or not required <= profile.keys()
        or (profile.keys() - required - {"positive_label"})
    ):
        raise ValueError("Supply an evaluation profile with supported fields")
    profile = _json(deepcopy(profile))
    if type(profile["format_version"]) is not int or profile["format_version"] != 1:
        raise ValueError("Unsupported evaluation profile format_version")
    if not isinstance(profile["id"], str) or not profile["id"].strip():
        raise ValueError("An evaluation profile needs a nonempty id")
    protocols = {"within_session", "cross_subject", "cross_session"}
    if not isinstance(profile["protocol"], str) or profile["protocol"] not in protocols:
        raise ValueError("Unsupported evaluation protocol")
    identities = profile["identities"]
    if not isinstance(identities, dict) or set(identities) != {
        "participant",
        "session",
        "trial",
    }:
        raise ValueError("Declare participant, session and trial identity fields")
    for fields in identities.values():
        _field_names(fields, "identity fields")
    metrics = _field_names(profile["metrics"], "metrics")
    if not set(metrics) <= {"accuracy", "balanced_accuracy", "roc_auc"}:
        raise ValueError("Unsupported evaluation metric")
    if profile["aggregation"] != "equal_participant":
        raise ValueError("Only explicit equal_participant aggregation is supported")
    if len(labels) < 2:
        raise ValueError("Evaluation requires at least two classes")
    if "roc_auc" in metrics:
        positive = _identity(profile.get("positive_label"), "positive_label")
        if len(labels) != 2 or positive not in labels:
            raise ValueError("Binary roc_auc needs an explicit positive_label")
    elif "positive_label" in profile:
        raise ValueError("positive_label is only used by roc_auc")
    splitter = profile["splitter"]
    if not isinstance(splitter, dict):
        raise ValueError("Supply an explicit splitter configuration")
    name = splitter.get("name")
    allowed = {
        "within_session": {"stratified_kfold", "explicit"},
        "cross_subject": {"leave_one_participant_out"},
        "cross_session": {"leave_one_session_out", "explicit"},
    }
    if not isinstance(name, str) or name not in allowed[profile["protocol"]]:
        raise ValueError("Splitter does not match the evaluation protocol")
    if name == "stratified_kfold":
        if set(splitter) != {"name", "n_splits", "random_state"}:
            raise ValueError("Stratified folds require n_splits and random_state")
        count, seed = splitter["n_splits"], splitter["random_state"]
        if (
            type(count) is not int
            or count < 2
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError(
                "Use n_splits >= 2 and an integer random_state in [0, 2**32)"
            )
    elif set(splitter) != {"name"}:
        raise ValueError("Unexpected splitter parameters")
    return profile


def _inventory(dataset, scope):
    return {
        "all_case_ids": [case["case_id"] for case in dataset.cases],
        "scope": [
            {
                "case": dataset.cases[i],
                "label": dataset.y[i],
                "group": dataset.groups[i],
            }
            for i in scope
        ],
        "metadata": dataset.metadata,
    }


def _grouped(scope, cases, fields):
    groups = {}
    for index in scope:
        groups.setdefault(_key(cases[index], fields), []).append(index)
    return dict(sorted(groups.items()))


def _splits(dataset, scope, profile, partitions):
    ids = profile["identities"]
    people = ids["participant"]
    sessions = _fields(people, ids["session"])
    trial_fields = _fields(sessions, ids["trial"])
    name = profile["splitter"]["name"]
    if name == "explicit":
        if not isinstance(partitions, (list, tuple)) or not partitions:
            raise ValueError(
                "Explicit evaluation requires a list of saved train/test assignments"
            )
        yield from partitions
        return
    if partitions is not None:
        raise ValueError("Supplied partitions require the explicit splitter")
    if name == "stratified_kfold":
        for pool in _grouped(scope, dataset.cases, sessions).values():
            trials = list(_grouped(pool, dataset.cases, trial_fields).values())
            labels = []
            for rows in trials:
                values = {dataset.y[i] for i in rows}
                if len(values) != 1:
                    raise ValueError(
                        "Trial-level stratification requires one label per trial"
                    )
                labels.append(next(iter(values)))
            count = profile["splitter"]["n_splits"]
            if min(Counter(labels).values()) < count:
                raise ValueError("Too few independent trials per class for n_splits")
            cv = StratifiedKFold(
                n_splits=count,
                shuffle=True,
                random_state=profile["splitter"]["random_state"],
            )
            for train, test in cv.split(np.zeros((len(trials), 1)), labels):
                yield {
                    "train": [i for t in train for i in trials[t]],
                    "test": [i for t in test for i in trials[t]],
                }
        return
    pools = (
        [scope]
        if name == "leave_one_participant_out"
        else _grouped(scope, dataset.cases, people).values()
    )
    group_fields = people if name == "leave_one_participant_out" else sessions
    for pool in pools:
        groups = [_key(dataset.cases[i], group_fields) for i in pool]
        if len(set(groups)) < 2:
            raise ValueError(
                "Protocol needs at least two participants or sessions in each scope"
            )
        for train, test in LeaveOneGroupOut().split(
            np.zeros((len(pool), 1)), groups=groups
        ):
            yield {"train": [pool[i] for i in train], "test": [pool[i] for i in test]}


def plan_classification_evaluation(
    dataset,
    profile,
    *,
    disjoint_by,
    source_fields,
    eligible_indices=None,
    partitions=None,
    partition_values=None,
):
    """Resolve an evaluation profile into saved, prevalidated case assignments.

    Parameters
    ----------
    dataset : ClassificationDataset
        Cases with explicit participant, session, trial and source identities.
        Signals are not inspected or fitted by this planner.
    profile : dict
        Version 1 profile: id, protocol, identity-field mappings, splitter,
        metrics and equal_participant aggregation. Binary roc_auc also requires
        positive_label. See the evaluation profile guide for supported presets.
    disjoint_by : sequence of sequences of str
        Existing problem constraints. Protocol rules are added without weakening
        these constraints. Supply an empty list if the problem adds none.
    source_fields : sequence of three str
        Original source and full processing-support bounds, passed to the public
        split validator. Source identity and coordinate assertions are trusted.
    eligible_indices : sequence of int or None, default=None
        Explicit global case scope. For nested planning this must be the parent
        training pool; this function does not discover that pool.
    partitions : sequence of dict or None, default=None
        Saved train/test assignments for an explicit preset. Supply exactly one
        complete holdout per participant/session for within_session, or per
        participant for cross_session.
    partition_values : mapping or None, default=None
        Explicit partition-to-field allowed values, passed to the split
        validator. Use these to retain prescribed training/evaluation session
        roles; disjoint sessions alone do not establish the intended direction.

    Returns
    -------
    dict
        JSON-compatible profile, case inventory hash, runtime, effective rules
        and all validated folds with global indices and ordered case IDs.

    Notes
    -----
    Generated within-session folds stratify independent single-label trials,
    then expand to their windows. Other presets leave out one participant or
    one session within each person. All folds are checked before returning.
    Class coverage is required in train/test and each scored test session.
    This does not inspect earlier preprocessing or estimator-internal tuning.
    """
    if not isinstance(dataset, ClassificationDataset):
        raise ValueError("dataset must be a ClassificationDataset")
    size = len(dataset.cases)
    if (
        len(dataset.X) != size
        or np.shape(dataset.y) != (size,)
        or np.shape(dataset.groups) != (size,)
    ):
        raise ValueError("Dataset arrays and cases are not aligned")
    case_ids = [case.get("case_id") for case in dataset.cases]
    if (
        any(not isinstance(c, str) or not c.strip() for c in case_ids)
        or len(set(case_ids)) != size
    ):
        raise ValueError("Case IDs must be unique nonempty strings")
    scope = _case_indices(
        np.arange(size) if eligible_indices is None else eligible_indices, size
    )
    scope = sorted(scope, key=lambda i: case_ids[i])
    labels = {_identity(dataset.y[i], "label") for i in scope}
    profile = _profile(profile, labels)
    identities = profile["identities"]
    people = identities["participant"]
    sessions = _fields(people, identities["session"])
    trials = _fields(sessions, identities["trial"])
    person_groups = _grouped(scope, dataset.cases, people)
    session_groups = _grouped(scope, dataset.cases, sessions)
    _grouped(scope, dataset.cases, trials)
    if not isinstance(disjoint_by, (list, tuple)):
        raise ValueError("disjoint_by must contain explicit problem identity rules")
    rules = [list(_field_names(rule, "disjoint_by")) for rule in disjoint_by]
    required = [trials]
    if profile["protocol"] == "cross_subject":
        required.append(people)
    elif profile["protocol"] == "cross_session":
        required.append(sessions)
    for rule in required:
        if rule not in rules:
            rules.append(rule)
    folds, explicit_units = [], []
    for number, split in enumerate(_splits(dataset, scope, profile, partitions)):
        if not isinstance(split, dict) or set(split) != {"train", "test"}:
            raise ValueError("Every fold must declare exactly train and test")
        rows = {name: _case_indices(indices, size) for name, indices in split.items()}
        union = rows["train"] + rows["test"]
        if not set(union) <= set(scope):
            raise ValueError("Fold contains cases outside the eligible scope")
        protocol = profile["protocol"]
        if protocol == "within_session":
            units = {_key(dataset.cases[i], sessions) for i in union}
            if len(units) != 1:
                raise ValueError(
                    "Within-session folds must stay within one participant/session"
                )
            unit = next(iter(units))
            pool = session_groups[unit]
            explicit_units.append(unit)
        elif protocol == "cross_subject":
            if len({_key(dataset.cases[i], people) for i in rows["test"]}) != 1:
                raise ValueError(
                    "Cross-subject folds must test exactly one participant"
                )
            pool = scope
        else:
            units = {_key(dataset.cases[i], people) for i in union}
            if len(units) != 1:
                raise ValueError("Cross-session folds must stay within one participant")
            pool = person_groups[next(iter(units))]
            if len({_key(dataset.cases[i], sessions) for i in rows["test"]}) != 1:
                raise ValueError("Cross-session folds must test exactly one session")
            explicit_units.append(next(iter(units)))
        for values in rows.values():
            if {dataset.y[i] for i in values} != labels:
                raise ValueError(
                    "Every fold partition must contain the evaluation class vocabulary"
                )
        for values in _grouped(rows["test"], dataset.cases, sessions).values():
            if {dataset.y[i] for i in values} != labels:
                raise ValueError(
                    "Every scored test session must contain all evaluation classes"
                )
        checked = validate_classification_split(
            dataset,
            rows,
            eligible_indices=pool,
            disjoint_by=rules,
            source_fields=source_fields,
            partition_values=partition_values,
            require_all_classes=("train", "test"),
        )
        folds.append({"id": f"fold-{number:04}", "split": checked})
    if profile["splitter"]["name"] == "explicit":
        # A prescribed cross-session benchmark scores one direction per person;
        # generating the reverse fold would change its scientific question.
        units = (
            person_groups if profile["protocol"] == "cross_session" else session_groups
        )
        if Counter(explicit_units) != Counter(units.keys()):
            raise ValueError(
                "Explicit mode needs exactly one holdout for each evaluation unit"
            )
    if not folds:
        raise ValueError("Evaluation has no folds")
    report = {
        "format_version": 1,
        "profile": profile,
        "eligible_indices": scope,
        "case_inventory_sha256": _digest(_inventory(dataset, scope)),
        "problem_disjoint_by": _json(disjoint_by),
        "source_fields": _json(source_fields),
        "folds": folds,
        "runtime": {"scikit-learn": version("scikit-learn"), "numpy": version("numpy")},
        "planner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    report = _json(report)
    report["plan_id"] = _digest(report)
    return report


def validate_classification_evaluation(dataset, plan):
    """Rebuild and compare a saved evaluation plan before executing its folds.

    Parameters
    ----------
    dataset : ClassificationDataset
        Current cases, labels, identities and metadata.
    plan : dict
        Saved result from plan_classification_evaluation.

    Returns
    -------
    dict
        Independently regenerated plan when every assignment and identity agrees.

    Raises
    ------
    ValueError
        If the plan is malformed, changed or stale for the current cases/runtime.
    """
    try:
        partitions = None
        if plan["profile"]["splitter"]["name"] == "explicit":
            partitions = [
                {
                    name: part["indices"]
                    for name, part in fold["split"]["partitions"].items()
                }
                for fold in plan["folds"]
            ]
        checked = plan_classification_evaluation(
            dataset,
            plan["profile"],
            disjoint_by=plan["problem_disjoint_by"],
            source_fields=plan["source_fields"],
            eligible_indices=plan["eligible_indices"],
            partitions=partitions,
            partition_values=plan["folds"][0]["split"]["policy"]["partition_values"],
        )
        if plan != checked:
            raise ValueError("Evaluation plan changed or is stale")
        return checked
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed evaluation plan") from exc
