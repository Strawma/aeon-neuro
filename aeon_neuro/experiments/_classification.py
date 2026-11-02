"""Connect validated classification cases to optional tsml-eval results."""

import hashlib
import inspect
import json
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder

from aeon_neuro.datasets import validate_classification_split


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_identity(value):
    value = np.asarray(value)
    if value.dtype.kind in "OUS":
        return {"shape": list(value.shape), "values": _portable(value.tolist())}
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }


def _implementation(value):
    name = f"{value.__module__}.{value.__qualname__}"
    try:
        path = inspect.getsourcefile(value)
    except TypeError:
        path = None
    return {"name": name, "sha256": None if path is None else _file_hash(path)}


def _portable(value):
    """Reject unstable repr-based identities, including hidden callable state."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _portable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, BaseEstimator):
        return {
            "implementation": _implementation(type(value)),
            "parameters": _portable(value.get_params(deep=False)),
        }
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    raise ValueError(f"No stable JSON identity for {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(
        json.dumps(_portable(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _data_identity(dataset):
    return {
        "X": _array_identity(dataset.X),
        "y": _array_identity(dataset.y),
        "groups": _array_identity(dataset.groups),
        "cases_sha256": _digest(dataset.cases),
        "exclusions_sha256": _digest(dataset.exclusions),
        "metadata_sha256": _digest(dataset.metadata),
    }


def _sources(manifest):
    if manifest is None:
        return {}
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source_manifest must be a nonempty path-to-SHA256 mapping")
    result = {}
    for name, expected in manifest.items():
        path = Path(name).resolve()
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("Source checksums must be lowercase SHA256 strings")
        if str(path) in result:
            raise ValueError("Source manifest repeats a resolved file")
        if _file_hash(path) != expected:
            raise ValueError(f"Source checksum changed: {path}")
        result[str(path)] = expected
    return result


def _backend():
    try:
        from filelock import FileLock
        from tsml_eval.evaluation.storage import load_classifier_results
        from tsml_eval.experiments import run_classification_experiment as run_tsml
    except ModuleNotFoundError as exc:
        if exc.name in {"filelock", "tsml_eval"}:
            raise ImportError(
                "Optional experiments require Python 3.12+ and "
                "'pip install aeon-neuro[tsml]'."
            ) from exc
        raise
    return FileLock, run_tsml, load_classifier_results


def _write(path, value):
    # Replace only a complete JSON document; a killed process must not leave
    # half a receipt that appears to certify an experiment.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _predictions(path, reader, truths, n_classes):
    try:
        result = reader(str(path), verify_values=True)
    except (AssertionError, ValueError, IndexError) as exc:
        raise ValueError("Invalid tsml-eval prediction file") from exc
    probabilities = np.asarray(result.probabilities)
    if (
        probabilities.shape != (len(truths), n_classes)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1)
        or not np.array_equal(result.class_labels, truths)
        or not np.array_equal(result.predictions, probabilities.argmax(axis=1))
        or not np.isclose(result.accuracy, np.mean(result.predictions == truths))
    ):
        raise ValueError("Prediction truths, class probabilities or decisions disagree")
    return result


def _verify_complete(report, root, reader, truths, mapping):
    for artifact in report["artifacts"].values():
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root) or _file_hash(path) != artifact["sha256"]:
            raise ValueError("Completed experiment artifact changed")
    map_path = root / report["artifacts"]["case_map"]["path"]
    if json.loads(map_path.read_text()) != mapping:
        raise ValueError("Prediction-to-case mapping changed")
    result = _predictions(
        root / report["artifacts"]["predictions"]["path"],
        reader,
        truths,
        len(mapping["class_order"]),
    )
    if report.get("class_order") != mapping["class_order"] or not np.isclose(
        report.get("accuracy"), result.accuracy
    ):
        raise ValueError("Completed experiment summary changed")


def run_classification_experiment(
    dataset,
    partitions,
    classifier,
    output_dir,
    *,
    dataset_name,
    provenance,
    disjoint_by,
    source_fields,
    eligible_indices=None,
    partition_values=None,
    require_all_classes=("train", "test"),
    source_manifest=None,
    classifier_name=None,
    resume=False,
):
    """Write verified tsml-eval results from explicitly assigned EEG cases.

    Parameters
    ----------
    dataset : ClassificationDataset
        Aligned cases in a finite, dense (case, channel, time) NumPy array.
        Earlier signal processing and its source support remain caller decisions.
    partitions : dict
        Exactly ``train`` and ``test`` indices, validated against the explicit
        policy. Iterate supplied folds outside this function, cloning per call.
    classifier : aeon.classification.BaseClassifier
        Estimator or aeon pipeline to clone before fitting. Ordinary sklearn
        estimators require an explicit aeon-compatible representation first.
    output_dir : str or pathlib.Path
        One experiment directory on a filesystem supporting file locks. Existing
        output is rejected unless ``resume=True``. Acquired sources are read-only.
    dataset_name : str
        Filename-safe problem name used by tsml-eval, without path separators.
    provenance : dict
        JSON-compatible problem, binding, preprocessing and protocol information,
        including the intended score unit. This is retained, not interpreted as
        a schema or scientific validation of the question.
    disjoint_by : sequence of sequences of str
        Identity constraints passed to ``validate_classification_split``.
    source_fields : sequence of three str
        Original source identity, inclusive start and exclusive stop fields.
        Include full filter/sequence support, not only returned epoch bounds.
    eligible_indices : sequence of int or None, default=None
        Explicit case scope, for example one person or an inner training pool.
    partition_values : dict or None, default=None
        Optional explicit person/session/role restrictions for each partition.
    require_all_classes : sequence of str, default=("train", "test")
        Partitions required to contain every class in the eligible scope.
        Test labels must always exist in training.
    source_manifest : dict or None, default=None
        Paths mapped to expected SHA256 hashes, verified before and after fitting
        and on resume. Without it, identity covers the supplied case arrays and
        metadata but makes no claim about unchanged acquisition files.
    classifier_name : str or None, default=None
        Filename-safe result label; defaults to the estimator class name.
    resume : bool, default=False
        Reuse intact complete results only when all identities match. Failed or
        interrupted matching runs get a new attempt; earlier files are retained.
        Corrupt completed artifacts or changed inputs raise without overwriting.

    Returns
    -------
    dict
        Completion receipt, source/case/split/estimator identities, class order,
        accuracy and relative artifact paths with SHA256 checksums. The CSV is
        an ordinary tsml-eval result; its separate map identifies original cases.

    Notes
    -----
    No resampling, automatic train-error CV or external data transform is used.
    Estimator-internal tuning, earlier learned preprocessing and aggregation
    across people/folds remain explicit responsibilities outside this adapter.
    Stored decisions follow probability argmax, which can differ from predict.
    """
    if not isinstance(classifier, BaseClassifier):
        raise TypeError("Use an explicit aeon classifier or ClassifierPipeline")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Supply nonempty problem/protocol provenance")
    classifier_name = (
        type(classifier).__name__ if classifier_name is None else classifier_name
    )
    for name in (dataset_name, classifier_name):
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
        ):
            raise ValueError("Result names must be filename-safe identifiers")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "test"}:
        raise ValueError("Supply exactly train and test assignments")
    checked = validate_classification_split(
        dataset,
        partitions,
        disjoint_by=disjoint_by,
        source_fields=source_fields,
        eligible_indices=eligible_indices,
        partition_values=partition_values,
        require_all_classes=require_all_classes,
    )
    if (
        not isinstance(dataset.X, np.ndarray)
        or dataset.X.ndim != 3
        or dataset.X.dtype.kind not in "fiu"
        or min(dataset.X.shape) < 1
        or not np.isfinite(dataset.X).all()
    ):
        raise ValueError(
            "Only finite dense numeric (case, channel, time) inputs are supported"
        )
    train = np.asarray(checked["partitions"]["train"]["indices"])
    test = np.asarray(checked["partitions"]["test"]["indices"])
    y = np.asarray(dataset.y)
    encoder = LabelEncoder().fit(y[train])
    if len(encoder.classes_) < 2:
        raise ValueError("Training requires at least two classes")
    truths = encoder.transform(y[test])
    data_identity = _data_identity(dataset)
    sources = _sources(source_manifest)
    FileLock, run_tsml, reader = _backend()
    versions = {}
    for package in (
        "aeon-neuro",
        "aeon",
        "tsml-eval",
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "mne",
        "pyriemann",
        "filelock",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    identity = {
        "data": data_identity,
        "split": checked,
        "provenance": _portable(provenance),
        "sources": sources,
        "estimator": _portable(classifier),
        "dataset_name": dataset_name,
        "classifier_name": classifier_name,
        "runtime": {"python": platform.python_version(), "versions": versions},
        "adapter_sha256": _file_hash(__file__),
        "backend": _implementation(run_tsml),
        "decision_rule": "argmax probabilities in encoded training class order",
    }
    identity = _portable(identity)
    run_id = _digest(identity)
    mapping = {
        "class_order": _portable(encoder.classes_.tolist()),
        "rows": [
            {
                "prediction_row": row,
                "case_id": dataset.cases[int(index)]["case_id"],
                "encoded_truth": int(truths[row]),
            }
            for row, index in enumerate(test)
        ],
    }
    root = Path(output_dir).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root.with_name(root.name + ".lock")), timeout=0)
    with lock:
        receipt_path = root / "run.json"
        attempts = []
        if root.exists():
            if not resume:
                raise FileExistsError(
                    "Experiment output exists; use resume=True or a new directory"
                )
            report = json.loads(receipt_path.read_text())
            if report.get("format_version") != 1:
                raise ValueError("Unsupported experiment receipt format")
            if (
                report.get("identity_sha256") != run_id
                or report.get("identity") != identity
            ):
                raise ValueError(
                    "Existing experiment identity differs from the current inputs"
                )
            if report.get("status") == "complete":
                _verify_complete(report, root, reader, truths, mapping)
                return report
            if report.get("status") not in {"running", "failed"}:
                raise ValueError("Unknown experiment completion state")
            attempts = report["attempts"]
        else:
            root.mkdir()
        # An OS file lock is released if a process dies. A matching interrupted
        # run can therefore retain its old attempt and start a fresh one safely.
        attempt = uuid4().hex
        attempts = [*attempts, attempt]
        target = root / "attempts" / attempt
        target.mkdir(parents=True)
        report = {
            "format_version": 1,
            "status": "running",
            "identity_sha256": run_id,
            "identity": identity,
            "attempts": attempts,
        }
        _write(receipt_path, report)
        _write(target / "attempt.json", report)
        try:
            fitted = clone(classifier)
            run_tsml(
                dataset.X[train].copy(),
                y[train].copy(),
                dataset.X[test].copy(),
                y[test].copy(),
                fitted,
                str(target / "results"),
                classifier_name=classifier_name,
                dataset_name=dataset_name,
                resample_id=0,
                data_transforms=None,
                build_train_file=False,
                ignore_custom_train_estimate=True,
                benchmark_time=False,
            )
            path = (
                target
                / "results"
                / classifier_name
                / "Predictions"
                / dataset_name
                / "testResample0.csv"
            )
            result = _predictions(path, reader, truths, len(encoder.classes_))
            if _data_identity(dataset) != data_identity:
                raise ValueError("Experiment changed the supplied dataset")
            _sources(sources if sources else None)
            map_path = target / "case-map.json"
            _write(map_path, mapping)
            report.update(
                status="complete",
                class_order=mapping["class_order"],
                accuracy=float(result.accuracy),
                artifacts={
                    name: {"path": str(p.relative_to(root)), "sha256": _file_hash(p)}
                    for name, p in (("predictions", path), ("case_map", map_path))
                },
            )
            _verify_complete(report, root, reader, truths, mapping)
            _write(target / "attempt.json", report)
            _write(receipt_path, report)
            return report
        except BaseException as exc:
            report.update(
                status="failed", error={"type": type(exc).__name__, "message": str(exc)}
            )
            _write(target / "attempt.json", report)
            _write(receipt_path, report)
            raise
