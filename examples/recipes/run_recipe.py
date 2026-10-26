"""Run a recipe and optionally check the merged CSP classifier on its fixed split."""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from aeon_neuro.datasets import load_bids_classification


def main():
    """Construct cases, retain provenance and optionally fit CSP on training only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--classifier", choices=("none", "csp"), default="none")
    args = parser.parse_args()
    start = time.monotonic()
    result = load_bids_classification(
        args.bids_root,
        args.recipe,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    summary = {
        "construction_seconds": time.monotonic() - start,
        "shape": result.report["shape"],
        "partitions": result.report["partitions"],
    }
    if args.classifier == "csp":
        from aeon_neuro.classification.feature_based import CSPClassifier

        X, y = result.dataset.X, result.dataset.y
        train, test = result.splits["train"], result.splits["test"]
        labels = result.recipe["target"]["class_order"]
        seed = result.recipe["evaluation"]["split"].get("random_seed", 42)
        parameters = {
            "n_components": min(4, X.shape[1]),
            "reg": "oas",
            "cov_est": "epoch",
            "random_state": seed,
        }
        model = CSPClassifier(**parameters)
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X[train], y[train])
            fit_seconds = time.monotonic() - start
            start = time.monotonic()
            prediction = model.predict(X[test])
            predict_seconds = time.monotonic() - start
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(train), 1)), y[train])
        dummy_prediction = baseline.predict(np.zeros((len(test), 1)))

        def metrics(pred):
            return {
                "accuracy": float(accuracy_score(y[test], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro")),
                "confusion_matrix": confusion_matrix(
                    y[test], pred, labels=labels
                ).tolist(),
            }

        summary.update(
            scope=(
                "Engineering integration run on one declared split; "
                "not a benchmark or a clinical result."
            ),
            model="Merged CSPClassifier (PR #208), default LDA",
            parameters=parameters,
            labels=labels,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            csp=metrics(prediction),
            most_frequent=metrics(dummy_prediction),
            warnings=sorted({str(item.message) for item in caught}),
            test_predictions=[
                {
                    "case_id": result.dataset.cases[int(i)]["case_id"],
                    "bids_subject": result.dataset.cases[int(i)]["bids_subject"],
                    "truth": str(y[i]),
                    "prediction": str(pred),
                }
                for i, pred in zip(test, prediction)
            ],
        )
        (args.output_dir / "classification-results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    sys.stdout.write(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("test_predictions", "warnings")
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
