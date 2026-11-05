"""Prepare an explicit Braindecode input handoff from constructed BIDS cases.

This example exports configuration and provenance without fitting. Its small
CPU settings are for interface checks, not a scientific training protocol.
"""

import argparse
import hashlib
import inspect
import json
from copy import deepcopy
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

from aeon_neuro.classification.deep_learning import BraindecodeClassifier
from aeon_neuro.datasets import (
    ClassificationDataset,
    load_bids_dataset,
    make_classification_problem,
)


def prepare_handoff(dataset, *, geometry, model_params=None):
    """Return an unfitted classifier and a detached, JSON-compatible input record.

    Parameters
    ----------
    dataset : ClassificationDataset
        Native-grid, equal-length output of the shared-info problem builder.
        Current case channels/rate must agree with every included case row.
    geometry : {"none", "standard_1005"}
        Required explicit choice. ``none`` passes no positions to this EEGNet.
        ``standard_1005`` passes current case names to the adapter, which builds
        template positions. Neither route uses or changes measured source geometry.
    model_params : dict or None, default=None
        EEGNet constructor settings. None uses small kernels/pooling for the
        bounded example. The adapter retains authority over reserved arguments.

    Returns
    -------
    classifier : BraindecodeClassifier
        Unfitted EEGNet configured for one CPU epoch with a fixed seed.
    report : dict
        Current input identity, ordered case IDs, effective parameters and the
        unchanged source/history/problem provenance. No evaluation is implied.
    """
    if geometry not in ("none", "standard_1005"):
        raise ValueError("geometry must be 'none' or 'standard_1005'")
    if not isinstance(dataset, ClassificationDataset):
        raise ValueError("Supply a constructed ClassificationDataset")
    X = dataset.X
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 3
        or 0 in X.shape
        or X.dtype.kind not in "fiu"
        or not np.isfinite(X).all()
    ):
        raise ValueError("This example requires finite dense real case arrays")
    metadata = dataset.metadata
    names = metadata.get("channel_names")
    sfreq = metadata.get("sampling_frequency")
    if (
        not isinstance(names, (list, tuple))
        or len(names) != X.shape[1]
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Current case channel_names must match the array axis")
    if (
        not isinstance(sfreq, Real)
        or isinstance(sfreq, (bool, np.bool_))
        or not isfinite(sfreq)
        or sfreq <= 0
    ):
        raise ValueError("Current case sampling_frequency must be finite and positive")
    if (
        len(dataset.cases) != len(X)
        or np.shape(dataset.y) != (len(X),)
        or np.shape(dataset.groups) != (len(X),)
    ):
        raise ValueError("Case arrays, labels, groups and rows must remain aligned")
    # Source descriptors retain the full recording axis. Only the constructed
    # case metadata describes a selected/reordered classifier channel axis.
    for index, row in enumerate(dataset.cases):
        if (
            row.get("row") != index
            or row.get("output_channel_names") != list(names)
            or row.get("source_sampling_frequency") != sfreq
            or row["stop_sample"] - row["start_sample"] != X.shape[2]
            or row["label"] != dataset.y[index]
            or row["group_id"] != dataset.groups[index]
        ):
            raise ValueError("Current case metadata disagrees with the case rows")
    provenance = metadata.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not {"source", "history", "problem", "construction"} <= provenance.keys()
    ):
        raise ValueError("Supply shared-info source/history/problem provenance")
    if model_params is None:
        model_params = {
            "kernel_length": 4,
            "depthwise_kernel_length": 4,
            "pool1_kernel_size": 2,
            "pool2_kernel_size": 2,
        }
    classifier = BraindecodeClassifier(
        model="EEGNet",
        model_params=deepcopy(model_params),
        sfreq=float(sfreq),
        channel_names=list(names) if geometry == "standard_1005" else None,
        n_epochs=1,
        batch_size=4,
        random_state=0,
        device="cpu",
    )
    report = {
        "format_version": 1,
        "fit_performed": False,
        "input": {
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
            "channel_names": list(names),
            "sampling_frequency": float(sfreq),
            "unit_policy": "native decoded values; no scaling in this handoff",
        },
        "geometry": (
            "standard_1005 template; source geometry unused"
            if geometry == "standard_1005"
            else "none; source geometry unused"
        ),
        "case_ids": [row["case_id"] for row in dataset.cases],
        "classifier_parameters": deepcopy(classifier.get_params(deep=False)),
        "provenance": deepcopy(provenance),
        "adapter_sha256": hashlib.sha256(
            Path(inspect.getfile(BraindecodeClassifier)).read_bytes()
        ).hexdigest(),
        "scope": "Input handoff only; no processing, fitting, split or score",
    }
    # Enforce portability before exporting this example's configuration record.
    # The record does not certify external array mutations or define a run plan.
    return classifier, json.loads(json.dumps(report, allow_nan=False))


def main(argv=None):
    """Build one selected local source's cases and export their unfitted handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--geometry", choices=("none", "standard_1005"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("Write the handoff outside the source dataset")
    if args.output.exists():
        raise FileExistsError(args.output)
    X, info = load_bids_dataset(args.dataset, subjects=[args.subject])
    cases = make_classification_problem(X, info, args.problem, args.binding)
    _, report = prepare_handoff(cases, geometry=args.geometry)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
