"""Train and test classifiers on a dataset and score them.

Within subject: k-fold inside each participant (grouped by recording when every
recording has one label, so windows of one file never sit on both sides).
Held blocks: two declared test rounds inside each participant.
Between subject: participant k-fold, leave one participant out, or a held-out
test set. Folds are checked with aeon-neuro's ``validate_group_split`` and
``validate_classification_split``. When every participant has one label
(patient vs control), between-subject results are scored per participant:
each person's cases vote for one prediction.
"""

from __future__ import annotations

import warnings
from collections import Counter
from math import ceil

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from aeon_neuro.datasets import validate_classification_split, validate_group_split
from aeon_neuro.experiments._held_blocks import (
    AGGREGATION,
    ASSIGNMENTS_FILE,
    check_assignments,
)
from aeon_neuro.experiments._held_blocks import MODE as HELD_BLOCKS
from aeon_neuro.experiments._selectors import (
    UNSUPERVISED_SELECTORS,
    make_unsupervised_selector,
)


class EvaluationError(ValueError):
    """A scheme that can't run on this dataset; the message says why."""


CLASSIFIERS = {
    "xdawn_lda": {"name": "xDAWN + LDA", "library": "aeon-neuro", "speed": "Fast"},
    "csp_lda": {"name": "CSP + LDA", "library": "aeon-neuro", "speed": "Fast"},
    "csp_svm": {"name": "CSP + SVM", "library": "aeon-neuro", "speed": "Fast"},
    "riemann_mdm": {"name": "Riemannian MDM", "library": "aeon-neuro", "speed": "Fast"},
    "minirocket": {"name": "MiniRocket", "library": "aeon", "speed": "Medium"},
    "hc2": {"name": "HIVE-COTE 2 (reduced)", "library": "aeon", "speed": "Slow"},
    "baseline": {
        "name": "Guessing baseline",
        "library": "scikit-learn",
        "speed": "Instant",
    },
}

FOLD_SELECTORS = {
    **UNSUPERVISED_SELECTORS,
    "riemannian": "Riemannian electrode selection",
    "bpso": "BPSO",
    "detach_rocket": "Detach-ROCKET",
}

# Within-subject testing needs this many cases of each class per participant,
# so every training fold still has several (LDA and CSP can't fit on one).
MIN_PER_CLASS = 5


def plural(n, word):
    """Format a count and its singular or plural label."""
    return f"{n} {word}{'' if n == 1 else 's'}"


def make_classifier(key, n_channels, sfreq, seed=0):
    """Construct a fresh, unfitted classifier from aeon-neuro or aeon."""
    if key in {"csp_lda", "csp_svm"}:
        from aeon_neuro.classification.feature_based import CSPClassifier

        n_components = max(2, min(4, n_channels))
        if key == "csp_lda":
            return CSPClassifier(n_components=n_components, reg="ledoit_wolf")
        from sklearn.svm import SVC

        return CSPClassifier(
            SVC(kernel="linear"),
            n_components=n_components,
            reg="ledoit_wolf",
            random_state=seed,
        )
    if key == "xdawn_lda":
        from aeon_neuro.classification.feature_based import XDAWNClassifier

        decimate = int(min(13, max(1, round(sfreq / 64))))
        return XDAWNClassifier(n_components=2, decimate=decimate, reg="ledoit_wolf")
    if key == "riemann_mdm":
        from aeon_neuro.classification.distance_based import RiemannianMDMClassifier

        return RiemannianMDMClassifier(covariance_estimator="oas")
    if key == "minirocket":
        from aeon.classification.convolution_based import MiniRocketClassifier

        return MiniRocketClassifier(random_state=seed)
    if key == "hc2":
        from aeon.classification.hybrid import HIVECOTEV2
        from aeon.classification.sklearn import RotationForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer

        # Fixed component sizes make a comparison repeatable across reruns.
        # DrCIF's aeon 1.6 Catch22 path requires float64 for these EEG arrays.
        return make_pipeline(
            FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}),
            HIVECOTEV2(
                stc_params={
                    "estimator": RotationForestClassifier(n_estimators=10),
                    "n_shapelet_samples": 128,
                    "max_shapelets": 32,
                },
                drcif_params={"n_estimators": 16},
                arsenal_params={"n_kernels": 256, "n_estimators": 8},
                tde_params={
                    "n_parameter_samples": 16,
                    "max_ensemble_size": 8,
                    "randomly_selected_params": 8,
                },
                random_state=seed,
                n_jobs=1,
            ),
        )
    if key == "baseline":
        from sklearn.dummy import DummyClassifier

        return _FlatInput(DummyClassifier(strategy="most_frequent"))
    raise EvaluationError(f"Unknown classifier: {key}")


def make_selector(method, keep, n_channels, seed=0):
    """Construct an unfitted aeon-neuro channel selector."""
    if method in UNSUPERVISED_SELECTORS:
        return make_unsupervised_selector(method, keep, n_channels, seed)

    from aeon_neuro.transformations.collection.channel_selection import (
        BPSO,
        DetachRocketChannelSelector,
        Riemannian,
    )

    proportion = min(1.0, keep / n_channels)
    if method == "riemannian":
        return Riemannian(proportion=proportion)
    if method == "bpso":
        return BPSO(proportion=proportion, random_state=seed)
    if method == "detach_rocket":
        return DetachRocketChannelSelector(proportion=proportion, random_state=seed)
    raise EvaluationError(f"Unknown channel selection: {method}")


class _SelectThenClassify:
    """Fit a selector and classifier on the training fold."""

    def __init__(self, selector, make_model):
        """Wrap a selector and classifier factory."""
        self.selector = selector
        self.make_model = make_model

    def fit(self, X, y):
        """Choose channels from the training cases, then fit the classifier on them."""
        kept = self.selector.fit_transform(X, y)
        self.model = self.make_model(kept.shape[1])
        self.model.fit(kept, y)
        return self

    def predict(self, X):
        """Predict from the channels chosen during fit."""
        return self.model.predict(self.selector.transform(X))


def _model_factory(key, dataset, selection):
    """Return a fresh model factory with optional per-fold channel selection."""
    n_channels, sfreq = dataset.X.shape[1], dataset.metadata["sampling_frequency"]
    method = (selection or {}).get("method")
    if not method or key == "baseline":
        return lambda: make_classifier(key, n_channels, sfreq)
    keep = int((selection or {}).get("keep") or 8)
    if not 1 <= keep < n_channels:
        raise EvaluationError(
            f"Channel selection: keep between 1 and {n_channels - 1} of the"
            f" {n_channels} channels."
        )
    return lambda: _SelectThenClassify(
        make_selector(method, keep, n_channels),
        lambda n: make_classifier(key, n, sfreq),
    )


class _FlatInput:
    """Lets a scikit-learn model that ignores X take aeon's 3D collections."""

    def __init__(self, model):
        """Wrap a scikit-learn classifier."""
        self.model = model

    def fit(self, X, y):
        """Fit on one dummy feature per case."""
        self.model.fit(np.zeros((len(X), 1)), y)
        return self

    def predict(self, X):
        """Predict from one dummy feature per case."""
        return self.model.predict(np.zeros((len(X), 1)))


def parse_scheme(value):
    """Resolve evaluation defaults and legacy string settings."""
    legacy = {
        "within": {"mode": "within"},
        "lopo": {"mode": "between", "scheme": "lopo"},
        "holdout": {"mode": "between", "scheme": "holdout"},
        "groupcv": {"mode": "between", "scheme": "kfold"},
        HELD_BLOCKS: {"mode": HELD_BLOCKS},
    }
    if isinstance(value, str) or value is None:
        value = legacy.get(value or "within", {"mode": "within"})
    scheme = {"mode": "within", "scheme": "kfold", "k": 5, "test_size": 0.3, "seed": 42}
    scheme.update({k: v for k, v in value.items() if v not in (None, "")})
    scheme["k"] = int(scheme["k"])
    scheme["test_size"] = float(scheme["test_size"])
    if scheme["test_size"] > 1:
        scheme["test_size"] /= 100
    scheme["seed"] = int(scheme["seed"])
    if scheme["mode"] == HELD_BLOCKS:
        scheme.update(scheme=HELD_BLOCKS, k=2)
    elif scheme["scheme"] == HELD_BLOCKS:
        raise EvaluationError(
            "Choose the held-out blocks evaluation goal for two declared block rounds."
        )
    return scheme


def describe_scheme(scheme):
    """Describe the evaluation scheme in words."""
    if scheme["mode"] == HELD_BLOCKS:
        return "within subject · held-out blocks"
    if scheme["mode"] == "within":
        return f"within subject · {scheme['k']}-fold"
    return {
        "kfold": f"between subject · {scheme['k']}-fold by participant",
        "lopo": "between subject · leave one participant out",
        "holdout": "between subject · held-out test set",
    }[scheme["scheme"]]


def participant_split(groups, y, test_size, seed):
    """Split participants with a seed and mark test cases with True."""
    rng = np.random.default_rng(seed)
    people = np.unique(groups)
    if len(people) < 2:
        raise EvaluationError("A train/test split needs at least two participants.")
    by_label = {}
    for person in people:
        values, counts = np.unique(y[groups == person], return_counts=True)
        by_label.setdefault(values[np.argmax(counts)], []).append(person)
    test = []
    for members in by_label.values():
        members = list(rng.permutation(members))
        n = (
            min(max(int(round(test_size * len(members))), 1), len(members) - 1)
            if len(members) >= 2
            else 0
        )
        test += members[:n]
    if not test:
        test = [rng.choice(people)]
    return np.isin(groups, test)


def _list_few(items):
    """Up to five (participant, class, count) entries in words."""
    shown = ", ".join(f"{person} ({n} {label})" for person, label, n in items[:5])
    return shown + (f" and {len(items) - 5} more" if len(items) > 5 else "")


def _participant_folds(idx, y, recordings, scheme, by_recording):
    """k-fold splits inside one participant; None if a class sits in one recording."""
    classes, counts = np.unique(y, return_counts=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if by_recording:
            k = min(
                scheme["k"], min(len(np.unique(recordings[y == c])) for c in classes)
            )
            if k < 2:
                return None
            splitter = StratifiedGroupKFold(
                k, shuffle=True, random_state=scheme["seed"]
            )
            return [(idx[a], idx[b]) for a, b in splitter.split(idx, y, recordings)]
        splitter = StratifiedKFold(
            min(scheme["k"], int(counts.min())),
            shuffle=True,
            random_state=scheme["seed"],
        )
        return [(idx[a], idx[b]) for a, b in splitter.split(idx, y)]


def _within_folds(dataset, scheme, notes):
    """Folds inside each participant, leaving out those with too few cases per class."""
    recordings = np.array([c["recording_id"] for c in dataset.cases])
    by_recording = all(
        len(np.unique(dataset.y[recordings == r])) == 1 for r in np.unique(recordings)
    )
    folds, too_few, one_class = [], [], []
    for person in np.unique(dataset.groups):
        idx = np.flatnonzero(dataset.groups == person)
        y = dataset.y[idx]
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            one_class.append(person)
            continue
        if counts.min() < MIN_PER_CLASS:
            too_few.append((person, classes[np.argmin(counts)], int(counts.min())))
            continue
        person_folds = _participant_folds(idx, y, recordings[idx], scheme, by_recording)
        if person_folds is None:
            notes.append(
                f"{person}: some classes come from a single recording, so windows "
                "of one recording are split between training and test."
            )
            person_folds = _participant_folds(idx, y, recordings[idx], scheme, False)
        folds += person_folds
    if not folds and too_few:
        fix = (
            "Supply more prepared cases for the classes with too few cases."
            if dataset.metadata.get("mode") == "prepared"
            else (
                "Use shorter windows or a smaller step, check the marker roles and time"
                " range, or choose between subject."
            )
        )
        raise EvaluationError(
            f"Within-subject testing needs at least {MIN_PER_CLASS} cases of every"
            " class in a participant; none has that many. Fewest:"
            f" {_list_few(too_few)}. {fix}"
        )
    if not folds:
        raise EvaluationError(
            "Within-subject testing needs every class in a participant. "
            "Here each participant has one label, so choose between subject."
        )
    if too_few:
        notes.append(
            f"{plural(len(too_few), 'participant')} left out of within-subject testing:"
            f" fewer than {MIN_PER_CLASS} cases of a class — {_list_few(too_few)}."
        )
    if one_class:
        notes.append(
            f"{plural(len(one_class), 'participant')} with only one class left out."
        )
    if (
        dataset.metadata.get("mode") != "prepared"
        and "start_sample" in dataset.cases[0]
    ):
        overlapping = sum(
            _overlaps_in_time(dataset, train, test) for train, test in folds
        )
        if overlapping:
            notes.append(
                f"In {overlapping} of {len(folds)} folds, training and test cases "
                "overlap in time; scores may be optimistic."
            )
    return folds


def _overlaps_in_time(dataset, train, test):
    """Check whether training and test windows share samples."""
    try:
        validate_classification_split(
            dataset,
            {"train": train.tolist(), "test": test.tolist()},
            disjoint_by=[("case_id",)],
            source_fields=("recording_id", "start_sample", "stop_sample"),
            eligible_indices=np.concatenate([train, test]).tolist(),
        )
    except ValueError as error:
        if "overlap" not in str(error):
            raise
        return True
    return False


def _between_folds(dataset, scheme, notes):
    """Create folds with independent participants."""
    split = np.asarray([c.get("split") or "" for c in dataset.cases])
    has_test = (split == "TEST").any() and (split == "TRAIN").any()
    if scheme["scheme"] == "holdout":
        if has_test:
            notes.append("Using the dataset's own TEST split.")
            test = split == "TEST"
        else:
            test = participant_split(
                dataset.groups, dataset.y, scheme["test_size"], scheme["seed"]
            )
        return [(np.flatnonzero(~test), np.flatnonzero(test))], len(
            np.unique(dataset.groups)
        )
    pool = np.flatnonzero(split == "TRAIN") if has_test else np.arange(len(dataset.y))
    groups = dataset.groups[pool]
    if has_test:
        notes.append("Folds use TRAIN only; TEST was not touched.")
    if scheme["scheme"] == "lopo":
        return [
            (pool[a], pool[b]) for a, b in LeaveOneGroupOut().split(pool, groups=groups)
        ], len(np.unique(groups))
    own_folds = np.asarray([dataset.cases[i].get("fold") for i in pool], dtype=object)
    if has_test and all(f is not None for f in own_folds) and len(set(own_folds)) >= 2:
        folds = [
            (pool[own_folds != f], pool[own_folds == f]) for f in sorted(set(own_folds))
        ]
        notes[-1] = (
            f"Using the dataset's {len(folds)} participant folds on TRAIN; TEST was not"
            " touched."
        )
        return folds, len(np.unique(groups))
    k = min(scheme["k"], len(np.unique(groups)))
    if k < 2:
        raise EvaluationError("Participant folds need at least two participants.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splitter = StratifiedGroupKFold(k, shuffle=True, random_state=scheme["seed"])
        folds = [
            (pool[a], pool[b]) for a, b in splitter.split(pool, dataset.y[pool], groups)
        ]
    return folds, len(np.unique(groups))


def _grouping_issue(dataset):
    # Prepared previews can display cases with placeholder IDs. Those IDs
    # cannot show that cases from the same person stay on the intended side.
    if dataset.metadata.get("grouping") == "incomplete":
        return (
            "Supply a prepared case table with a participant or group ID for every case"
            " before evaluating."
        )
    return None


def _held_block_folds(dataset):
    """Bind the declared rounds to prepared case positions, without resplitting."""
    meta = dataset.metadata
    if meta.get("mode") != "prepared":
        raise EvaluationError("Held-block testing needs one prepared evaluation pool.")
    if len(meta["splits"]) != 1:
        raise EvaluationError(
            "Supply the eligible held-block cases as one prepared .ts file."
        )
    if meta.get("grouping") != "participants":
        raise EvaluationError(
            "Held-block testing needs a participant_id for every prepared case."
        )
    if error := meta.get("block_assignments_error"):
        raise EvaluationError(error)
    assignments = meta.get("block_assignments")
    if assignments is None:
        raise EvaluationError(
            f"Supply {ASSIGNMENTS_FILE} with each case's recording, block, source"
            " samples and test fold."
        )
    positions = {row["case_id"]: index for index, row in enumerate(dataset.cases)}
    if positions.keys() != assignments.keys():
        missing = len(positions.keys() - assignments.keys())
        extra = len(assignments.keys() - positions.keys())
        raise EvaluationError(
            f"{ASSIGNMENTS_FILE} must cover every prepared case exactly once. Missing:"
            f" {missing}. Extra: {extra}."
        )
    for case_id, index in positions.items():
        assignment = assignments[case_id]
        if (
            assignment.participant_id != dataset.groups[index]
            or assignment.label != dataset.y[index]
        ):
            raise EvaluationError(
                f"{ASSIGNMENTS_FILE}: participant or label differs for case {case_id}."
            )
    try:
        memberships = check_assignments(assignments)
    except ValueError as error:
        raise EvaluationError(f"{ASSIGNMENTS_FILE}: {error}") from error
    folds = []
    for test_ids, train_ids in memberships.items():
        train = np.array(sorted(positions[case_id] for case_id in train_ids))
        test = np.array(sorted(positions[case_id] for case_id in test_ids))
        if min(Counter(dataset.y[train]).values()) < 2:
            row = assignments[dataset.cases[test[0]]["case_id"]]
            raise EvaluationError(
                f"{row.participant_id}, round {row.test_fold}: need at least two"
                " training cases per class."
            )
        folds.append((train, test))
    return folds


def plan_folds(dataset, scheme):
    """Build every train/test fold and check it before any model is fitted."""
    scheme = parse_scheme(scheme)
    if scheme["mode"] == HELD_BLOCKS:
        return _held_block_folds(dataset), [
            "Uses the two supplied block rounds within each participant."
        ]
    if issue := _grouping_issue(dataset):
        raise EvaluationError(issue)
    notes = []
    if scheme["mode"] == "within":
        return _within_folds(dataset, scheme, notes), notes
    if len(np.unique(dataset.groups)) < 2:
        raise EvaluationError(
            "Between-subject testing needs at least two participants."
        )
    folds, n_people = _between_folds(dataset, scheme, notes)
    if scheme["scheme"] != "holdout" and n_people < 5:
        notes.append(f"Only {n_people} participants; scores will be unstable.")
    for train, test in folds:
        if not len(train) or not len(test):
            raise EvaluationError("A fold came out empty; use fewer folds.")
        _, train_counts = np.unique(dataset.y[train], return_counts=True)
        if len(train_counts) < 2:
            raise EvaluationError(
                "A training fold has only one class; use fewer folds or more"
                " participants."
            )
        if train_counts.min() < 2:
            raise EvaluationError(
                "A training fold has a single case of a class, too few to fit a"
                " classifier; use fewer folds or get more cases of that class."
            )
        validate_group_split(dataset, train, test)
    return folds, notes


def _held_block_details(dataset, test):
    assignments = dataset.metadata["block_assignments"]
    held = [assignments[dataset.cases[i]["case_id"]] for i in test]
    return {
        "test_round": held[0].test_fold,
        "test_blocks": [
            {"recording_id": recording, "block_id": block}
            for recording, block in sorted(
                {(case.recording_id, case.block_id) for case in held}
            )
        ],
    }


def _check_held_block_options(scheme, shuffle_runs):
    if scheme["mode"] == HELD_BLOCKS and int(shuffle_runs or 0):
        raise EvaluationError("Turn off label shuffling for held-block testing.")


def preview_folds(dataset, scheme, *, shuffle_runs=0):
    """Summarise the evaluator's assignments without fitting any model."""
    try:
        scheme = parse_scheme(scheme)
        _check_held_block_options(scheme, shuffle_runs)
        folds, notes = plan_folds(dataset, scheme)
    except ValueError as error:
        return None, {"error": str(error)}

    classes = np.unique(dataset.y)

    def counts(indices):
        # Show zero as well, so a class absent from either side is visible.
        present = Counter(dataset.y[indices])
        return [present[label] for label in classes]

    return folds, {
        "classes": [str(label) for label in classes],
        "notes": notes,
        "folds": [
            {
                "train_cases": len(train),
                "test_cases": len(test),
                "test_groups": [
                    str(group) for group in np.unique(dataset.groups[test])
                ],
                "train_counts": counts(train),
                "test_counts": counts(test),
                **(
                    _held_block_details(dataset, test)
                    if scheme["mode"] == HELD_BLOCKS
                    else {}
                ),
            }
            for train, test in folds
        ],
    }


def _suggested_schemes(dataset, scheme):
    """Try the requested evaluation goal, with the supplied TEST partition first."""
    if scheme["mode"] == HELD_BLOCKS:
        yield scheme
        return
    k = min(20, max(2, scheme["k"]))
    if scheme["mode"] == "within":
        for count in range(k, 1, -1):
            yield {**scheme, "scheme": "kfold", "k": count}
        return

    partitions = {case.get("split") for case in dataset.cases}
    supplied = {"TRAIN", "TEST"} <= partitions
    methods = (
        ["holdout", scheme["scheme"], "kfold"]
        if supplied
        else [scheme["scheme"], "kfold", "holdout"]
    )
    groups = {
        group
        for group, case in zip(dataset.groups, dataset.cases, strict=True)
        if not supplied or case.get("split") == "TRAIN"
    }
    for method in dict.fromkeys(methods):
        if method == "kfold":
            # The planner caps k at the number of eligible groups anyway.
            for count in range(max(2, min(k, len(groups))), 1, -1):
                yield {**scheme, "scheme": method, "k": count}
        else:
            yield {**scheme, "scheme": method}


def _suggested_model(dataset, selection, n_classes, min_train):
    """Check model and selector input requirements without fitting either."""
    _, channels, timepoints = dataset.X.shape
    method = selection.get("method")
    if method:
        if not isinstance(method, str) or method not in FOLD_SELECTORS or channels < 2:
            return (
                None,
                (
                    "Turn off per-fold channel selection or choose a supported selector"
                    " with at least two channels."
                ),
            )
        try:
            keep = int(selection.get("keep") or 8)
        except (TypeError, ValueError, OverflowError):
            keep = 0
        if not 1 <= keep < channels:
            return (
                None,
                (
                    f"Channel selection must keep between 1 and {channels - 1} of the"
                    f" {channels} channels."
                ),
            )
        if method in ("bpso", "detach_rocket"):
            # These selectors run an internal stratified holdout and use MiniRocket.
            validation = ceil(min_train * (0.5 if method == "bpso" else 0.33))
            if timepoints < 9 or min(validation, min_train - validation) < n_classes:
                return (
                    None,
                    (
                        f"{FOLD_SELECTORS[method]} needs at least 9 timepoints and room"
                        " for every class on both sides of its validation split. Turn"
                        " it off or prepare more cases or longer windows."
                    ),
                )
        elif method != "variance" and timepoints < 2:
            return (
                None,
                f"{FOLD_SELECTORS[method]} needs at least two timepoints per case.",
            )

    # OAS regularizes case covariances even with short windows and small training
    # folds. For band-power series the evaluator only supports MiniRocket.
    if dataset.metadata.get("band_power"):
        if timepoints >= 9:
            return "minirocket", None
        return (
            None,
            (
                "MiniRocket needs at least 9 timepoints per case. Use a longer window"
                " or turn off band power."
            ),
        )
    if timepoints >= 2:
        return "riemann_mdm", None
    return (
        None,
        (
            "The suggested models need at least two timepoints per case. Use a longer"
            " window."
        ),
    )


def suggest_setup(dataset, evaluation, selection, *, shuffle_runs=0):
    """Propose a split and one classifier without changing the evaluation policy."""
    try:
        scheme = parse_scheme(evaluation)
        if scheme["mode"] not in {"within", "between", HELD_BLOCKS} or scheme[
            "scheme"
        ] not in {"kfold", "lopo", "holdout", HELD_BLOCKS}:
            raise EvaluationError("Unknown evaluation settings.")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return {
            "error": "Choose a valid evaluation goal, fold count, test share and seed."
        }
    try:
        _check_held_block_options(scheme, shuffle_runs)
    except (TypeError, ValueError) as error:
        return {"error": str(error)}
    if scheme["mode"] != HELD_BLOCKS and (issue := _grouping_issue(dataset)):
        return {"error": issue}

    labels = set(dataset.y)
    rejected = {}
    for candidate in _suggested_schemes(dataset, scheme):
        name = describe_scheme(candidate).capitalize()
        try:
            folds, notes = plan_folds(dataset, candidate)
        except ValueError as error:
            if scheme["mode"] == HELD_BLOCKS:
                return {"error": str(error)}
            rejected.setdefault(str(error), name)
            # A within-subject planner error occurs before usable assignments
            # exist and lowering k cannot restore missing participant classes.
            if scheme["mode"] == "within":
                break
            continue

        # The planner allows some folds to omit a class. A suggestion should
        # train on all classes in each fold and test all of them across the run.
        train_counts = [Counter(dataset.y[train]) for train, _ in folds]
        if any(
            counts.keys() != labels or min(counts.values()) < 2
            for counts in train_counts
        ):
            rejected.setdefault(
                "Every training fold needs at least two cases of each class. Add cases"
                " or choose a different split.",
                name,
            )
            continue
        test_labels = [set(dataset.y[test]) for _, test in folds]
        if set().union(*test_labels) != labels:
            rejected.setdefault(
                "The test cases across the planned folds do not cover every class. Add"
                " test cases or choose a different split.",
                name,
            )
            continue
        min_train = min(len(train) for train, _ in folds)
        key, issue = _suggested_model(dataset, selection, len(labels), min_train)
        if issue:
            rejected.setdefault(issue, name)
            continue
        if any(present != labels for present in test_labels):
            notes.append(
                "Some test folds contain only a subset of classes. All classes are"
                " tested across the folds."
            )
        return {
            "evaluation": candidate,
            "classifier": {
                "key": key,
                "name": CLASSIFIERS[key]["name"],
                "reason": (
                    "It supports the prepared band-power time series."
                    if key == "minirocket"
                    else (
                        "It regularizes channel covariance estimates, which can help"
                        " with short EEG segments."
                    )
                ),
            },
            "baseline": CLASSIFIERS["baseline"]["name"],
            "fold_count": len(folds),
            "min_train_cases": min_train,
            "notes": notes,
            "alternatives": [
                {"scheme": label, "reason": reason}
                for reason, label in rejected.items()
            ],
        }
    return {
        "error": "No supported setup for this evaluation goal.",
        "alternatives": [
            {"scheme": label, "reason": reason} for reason, label in rejected.items()
        ],
    }


def _predict_folds(make, dataset, folds, y, fitted=None):
    """Fit each fold and collect the test predictions."""
    predictions = np.empty(len(y), dtype=object)
    tested = np.zeros(len(y), dtype=bool)
    for train, test in folds:
        model = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(dataset.X[train], y[train])
            predictions[test] = np.asarray(model.predict(dataset.X[test]))
        tested[test] = True
        if fitted is not None:
            fitted.append(model)
    return predictions, tested


def _vote(groups, y_true, y_pred, classes):
    """Take the most-predicted class for each participant."""
    people = np.unique(groups)
    truth, voted = [], []
    for person in people:
        mine = groups == person
        counts = [(y_pred[mine] == c).sum() for c in classes]
        voted.append(classes[int(np.argmax(counts))])
        truth.append(y_true[mine][0])
    return np.asarray(truth), np.asarray(voted)


def _refusal(key, dataset, n_classes):
    """Why a classifier is not run on this dataset, or None."""
    if dataset.metadata.get("band_power") and key not in ("minirocket", "baseline"):
        return (
            "not run — band power leaves 5 band series and no channels; only MiniRocket"
            " can use it."
        )
    n_cases, _, n_times = dataset.X.shape
    needed = n_classes * n_times * n_cases * n_times * 8  # MNE's xDAWN design matrix
    if key == "xdawn_lda" and needed > 2e9:
        return (
            f"not run — xDAWN would need about {needed / 1e9:.0f} GB for "
            f"{n_cases} cases of {n_times} samples. It is meant for "
            "time-locked trials (a few hundred)."
        )
    return None


def _score_classifier(
    key,
    dataset,
    folds,
    score_of,
    by_participant,
    selection,
    progress,
    *,
    fold_predictions=None,
):
    """Cross-validate a classifier and pool the case predictions."""
    make = _model_factory(key, dataset, selection)
    fitted = []
    predictions = np.empty(len(dataset.y), dtype=object)
    tested = np.zeros(len(dataset.y), dtype=bool)
    fold_scores = []
    for k, (train, test) in enumerate(folds):
        progress(f"fold {k + 1}/{len(folds)}")
        p, t = _predict_folds(make, dataset, [(train, test)], dataset.y, fitted)
        fold_labels = p[test].astype(dataset.y.dtype)
        # Capture each fold before pooling, which keeps only the last test
        # prediction for a case if it occurs in more than one fold.
        if fold_predictions is not None:
            fold_predictions.append(list(map(str, fold_labels)))
        predictions[t] = p[t]
        tested |= t
        fold_scores.append(
            score_of(dataset.y[test], fold_labels, dataset.groups[test])[0]
        )
    y_true = dataset.y[tested]
    y_pred = predictions[tested].astype(dataset.y.dtype)
    groups = dataset.groups[tested]
    score, truth, predicted = score_of(y_true, y_pred, groups)
    entry = {
        "key": key,
        **CLASSIFIERS[key],
        "balanced_accuracy": score,
        "fold_scores": fold_scores,
        "n_tested": int(len(truth)),
        "n_cases_tested": int(tested.sum()),
        "sd": float(np.std(fold_scores)) if len(fold_scores) > 1 else None,
    }
    chosen = [
        m.selector.channels_selected_
        for m in fitted
        if isinstance(m, _SelectThenClassify)
    ]
    if chosen:
        names = dataset.metadata["channel_names"]
        counts = Counter(names[int(i)] for picks in chosen for i in picks)
        entry["channels_chosen"] = {name: n for name, n in counts.most_common()}
    if by_participant:
        entry["case_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    elif key != "baseline":
        entry["per_participant"] = {
            str(g): float(
                balanced_accuracy_score(y_true[groups == g], y_pred[groups == g])
            )
            for g in np.unique(groups)
            if len(np.unique(y_true[groups == g])) > 1
        }
    return entry, (truth, predicted)


def _shuffle_test(
    dataset, folds, key, runs, score_of, by_participant, best_score, selection, progress
):
    """Estimate how often the best classifier scores as well on shuffled labels."""
    rng = np.random.default_rng(0)
    people = np.unique(dataset.groups)
    make = _model_factory(key, dataset, selection)
    null = []
    for run in range(runs):
        shuffled = dataset.y.copy()
        if by_participant:
            label_of = dict(
                zip(
                    people,
                    rng.permutation(
                        [dataset.y[dataset.groups == g][0] for g in people]
                    ),
                )
            )
            shuffled = np.asarray(
                [label_of[g] for g in dataset.groups], dtype=dataset.y.dtype
            )
        else:
            for person in people:
                mine = np.flatnonzero(dataset.groups == person)
                shuffled[mine] = rng.permutation(shuffled[mine])
        predictions, tested = _predict_folds(make, dataset, folds, shuffled)
        null.append(
            score_of(
                shuffled[tested],
                predictions[tested].astype(dataset.y.dtype),
                dataset.groups[tested],
            )[0]
        )
        progress(run + 1)
    as_good = int(np.sum(np.asarray(null) >= best_score))
    return {"runs": runs, "as_good": as_good, "p_value": (as_good + 1) / (runs + 1)}


def _evaluation_evidence(dataset, folds, predictions, unit, *, held_blocks=False):
    """Describe the exact assignments and predictions of a completed run."""
    case_ids = [case["case_id"] for case in dataset.cases]
    evidence = {
        "format_version": 1,
        "scoring": {
            "unit": unit,
            "repeated_test_case": "last_fold",
            "participant_vote_tie_break": "class_order",
        },
        "cases": [
            {"case_id": case_id, "group_id": str(group), "true_label": str(label)}
            for case_id, group, label in zip(
                case_ids, dataset.groups, dataset.y, strict=True
            )
        ],
        "folds": [
            {
                "fold": index,
                "train_case_ids": [case_ids[i] for i in train],
                "test_case_ids": [case_ids[i] for i in test],
            }
            for index, (train, test) in enumerate(folds, start=1)
        ],
        "predictions": predictions,
    }
    if dataset.exclusions:
        evidence["exclusions"] = [
            {"case_id": row["case_id"], "reason": row["exclusion_reason"]}
            for row in dataset.exclusions
        ]
    if held_blocks:
        evidence["scoring"]["aggregation"] = AGGREGATION
        assignments = dataset.metadata["block_assignments"]
        for row in evidence["cases"]:
            case = assignments[row["case_id"]]
            row.update(
                recording_id=case.recording_id,
                block_id=case.block_id,
                test_fold=case.test_fold,
                start_sample=case.start_sample,
                stop_sample=case.stop_sample,
            )
        for fold, (_, test) in zip(evidence["folds"], folds, strict=True):
            fold.update(_held_block_details(dataset, test))
    return evidence


def evaluate(
    dataset, config, progress=lambda fraction, message: None, *, include_evidence=False
):
    """Cross-validate the chosen classifiers and the guessing baseline.

    When every participant has one label and participants are held out, each
    person's cases vote for one prediction, and scores count people.
    With include_evidence, include a versioned record of the assignments and
    the unpooled test predictions of each scored classifier.
    """
    keys = [
        k for k in config.get("classifiers", []) if k in CLASSIFIERS and k != "baseline"
    ]
    if not keys:
        raise EvaluationError("Choose at least one classifier.")
    keys.append("baseline")
    scheme = parse_scheme(config.get("evaluation"))
    shuffles = int(config.get("shuffle_runs") or 0)
    _check_held_block_options(scheme, shuffles)
    folds, notes = plan_folds(dataset, scheme)
    classes = np.unique(dataset.y)
    by_participant = scheme["mode"] == "between" and all(
        len(np.unique(dataset.y[dataset.groups == g])) == 1
        for g in np.unique(dataset.groups)
    )
    unit = "participants" if by_participant else "trials"
    held_blocks = scheme["mode"] == HELD_BLOCKS
    selection = config.get("channel_selection") or {}
    total = (len(keys) + shuffles) * len(folds)

    def score_of(y_true, y_pred, groups):
        """(balanced accuracy, truth, prediction), per participant when they vote."""
        if by_participant:
            y_true, y_pred = _vote(groups, y_true, y_pred, classes)
        if held_blocks:
            # Pool the two test rounds within each person before taking the
            # mean, so a person with more cases does not have more weight.
            scores = [
                balanced_accuracy_score(
                    y_true[groups == group], y_pred[groups == group]
                )
                for group in np.unique(groups)
            ]
            return float(np.mean(scores)), y_true, y_pred
        return float(balanced_accuracy_score(y_true, y_pred)), y_true, y_pred

    results, pooled, predictions = [], {}, []
    for i, key in enumerate(keys):
        name = CLASSIFIERS[key]["name"]
        progress(i * len(folds) / total, f"Training {name}…")
        refusal = _refusal(key, dataset, len(classes))
        if refusal:
            results.append({"key": key, **CLASSIFIERS[key], "error": refusal})
            continue
        done = i * len(folds)

        def tick(step, done=done, name=name):
            progress(done / total, f"Training {name} · {step}…")

        fold_predictions = [] if include_evidence else None
        try:
            entry, pooled[key] = _score_classifier(
                key,
                dataset,
                folds,
                score_of,
                by_participant,
                selection,
                tick,
                fold_predictions=fold_predictions,
            )
            if fold_predictions is not None:
                predictions.append(
                    {
                        "classifier": key,
                        "result_index": len(results),
                        "folds": fold_predictions,
                    }
                )
        except MemoryError:
            entry = {"key": key, **CLASSIFIERS[key], "error": "ran out of memory"}
        except Exception as error:
            entry = {"key": key, **CLASSIFIERS[key], "error": str(error)}
        results.append(entry)
    scored = [r for r in results if "balanced_accuracy" in r and r["key"] != "baseline"]
    if not scored:
        raise EvaluationError(
            "Every classifier failed: "
            + "; ".join(f"{r['name']}: {r['error']}" for r in results if r.get("error"))
        )
    best = max(scored, key=lambda r: r["balanced_accuracy"])
    truth, predicted = pooled[best["key"]]
    matrix = confusion_matrix(truth, predicted, labels=classes)
    shuffle = None
    if shuffles:
        done = len(keys) * len(folds)
        shuffle = _shuffle_test(
            dataset,
            folds,
            best["key"],
            shuffles,
            score_of,
            by_participant,
            best["balanced_accuracy"],
            selection,
            lambda run: progress(
                (done + run * len(folds)) / total, f"Shuffle test {run}/{shuffles}…"
            ),
        )
    notes += _report_notes(
        dataset, scored, best, scheme, matrix, classes, by_participant
    )
    if best.get("channels_chosen"):
        often = ", ".join(
            f"{name} ({n}/{len(folds)})"
            for name, n in list(best["channels_chosen"].items())[:8]
        )
        method = FOLD_SELECTORS.get(selection.get("method"), selection.get("method"))
        notes.append(
            f"{method} chose channels in each training fold; for {best['name']} most"
            f" often: {often}."
        )
    result = {
        "scheme": scheme,
        "scheme_label": describe_scheme(scheme),
        "unit": unit,
        "n_folds": len(folds),
        "n_trials": scored[0]["n_tested"],
        "n_cases": scored[0]["n_cases_tested"],
        "n_test_groups": len({dataset.groups[i] for _, test in folds for i in test}),
        "classes": [str(c) for c in classes],
        "chance_level": 1 / len(classes),
        "best": best["key"],
        "classifiers": results,
        "confusion": matrix.tolist(),
        "shuffle": shuffle,
        "notes": notes,
    }
    if include_evidence:
        result["evaluation_evidence"] = _evaluation_evidence(
            dataset, folds, predictions, unit, held_blocks=held_blocks
        )
    return result


def _report_notes(dataset, scored, best, scheme, matrix, classes, by_participant):
    """Plain-language caveats shown under the results."""
    notes = []
    if by_participant:
        notes.append(
            "Scored per participant: each person's cases vote for one prediction."
        )
    if scheme["mode"] == HELD_BLOCKS:
        notes.append(
            "Scores average balanced accuracy across people. The confusion matrix"
            " counts individual EEG segments."
        )
    ranked = sorted(scored, key=lambda r: -r["balanced_accuracy"])
    if len(ranked) > 1:
        first, second = ranked[0], ranked[1]
        spread = max(first["sd"] or 0, second["sd"] or 0)
        if spread and first["balanced_accuracy"] - second["balanced_accuracy"] < spread:
            notes.append(
                f"{first['name']} and {second['name']} differ by less than one SD"
                " across folds; neither clearly wins."
            )
        held_out = scheme["mode"] == "between" and scheme.get("scheme") == "holdout"
        notes.append(
            f"Picking the best on the same {'test set' if held_out else 'folds'} it is"
            " scored on makes its score slightly optimistic."
        )
    if len(np.unique(dataset.groups)) == 1:
        notes.append("One participant: says nothing about new people.")
    elif scheme["mode"] in {"within", HELD_BLOCKS}:
        notes.append("Within-participant: does not show transfer to new people.")
    chance = 1 / len(classes)
    if all(r["balanced_accuracy"] + max(r["sd"] or 0, 0.05) < chance for r in scored):
        notes.append(
            "Every classifier scores well below chance: held-out data looks like "
            "the other class. This usually means the models learned differences "
            "between recordings (drift, setup) rather than between conditions."
        )
    recall = matrix.diagonal() / np.maximum(matrix.sum(1), 1)
    if len(classes) == 2 and abs(recall[0] - recall[1]) > 0.2:
        notes.append(f"{best['name']} favours '{classes[int(np.argmax(recall))]}'.")
    return notes
