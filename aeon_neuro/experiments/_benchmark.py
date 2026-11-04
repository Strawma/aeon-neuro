"""Plan and run serial suites over explicit classification problems."""

import json
import re
from pathlib import Path

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from aeon_neuro.experiments._classification import (
    _backend,
    _data_identity,
    _digest,
    _file_hash,
    _portable,
    _write,
    run_classification_experiment,
)
from aeon_neuro.model_selection import validate_classification_evaluation


def _snapshot(value):
    return json.loads(json.dumps(_portable(value), allow_nan=False))


def _identifier(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
    ):
        raise ValueError("Benchmark identifiers must be filename-safe strings")
    return value


def _specification(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    fields = {"format_version", "id", "problems", "classifiers", "estimator_seeds"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Benchmark specification needs exactly {sorted(fields)}")
    if type(value["format_version"]) is not int or value["format_version"] != 1:
        raise ValueError("Unsupported benchmark specification format")
    _identifier(value["id"])
    for name in ("problems", "classifiers", "estimator_seeds"):
        items = value[name]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{name} must be a nonempty list")
        for item in items:
            if name == "estimator_seeds":
                if type(item) is not int or not 0 <= item < 2**32:
                    raise ValueError("Estimator seeds must be unsigned 32-bit integers")
            else:
                _identifier(item)
        if len(set(items)) != len(items):
            raise ValueError(f"{name} must not repeat entries")
    return _snapshot(value)


def _seeded(classifier, seed):
    if not isinstance(classifier, BaseClassifier):
        raise ValueError("Benchmark classifiers must be explicit aeon classifiers")
    result = clone(classifier)
    # Seed every exposed estimator parameter, including sklearn pipeline steps.
    # Global RNG state and hidden randomness are deliberately outside this API.
    settings = {
        key: seed
        for key in result.get_params(deep=True)
        if key.split("__")[-1] == "random_state"
    }
    result.set_params(**settings)
    return result


def _problem(value):
    required = {"dataset", "evaluation", "provenance"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - {"source_manifest"}
    ):
        raise ValueError("Prepared problems need dataset, evaluation and provenance")
    if not isinstance(value["provenance"], dict) or not value["provenance"]:
        raise ValueError("Prepared problems need nonempty provenance")
    evaluation = validate_classification_evaluation(
        value["dataset"], value["evaluation"]
    )
    if set(evaluation["profile"]["metrics"]) - {"accuracy", "balanced_accuracy"}:
        raise ValueError(
            "Serial benchmarks currently support accuracy and balanced_accuracy"
        )
    sources = value.get("source_manifest")
    if sources is not None:
        if not isinstance(sources, dict) or not sources:
            raise ValueError(
                "source_manifest must be a nonempty path-to-SHA256 mapping"
            )
        if any(
            not isinstance(p, str)
            or not isinstance(h, str)
            or re.fullmatch(r"[0-9a-f]{64}", h) is None
            for p, h in sources.items()
        ):
            raise ValueError(
                "source_manifest must map paths to lowercase SHA256 values"
            )
    return _snapshot(
        {
            "evaluation": evaluation,
            "data": _data_identity(value["dataset"]),
            "provenance": value["provenance"],
            "source_manifest": sources,
        }
    )


def plan_classification_benchmark(specification, problems, classifiers):
    """Expand a strict version-1 specification into frozen serial fold jobs.

    Parameters
    ----------
    specification : dict, str or pathlib.Path
        JSON object or UTF-8 JSON file with exactly format_version (1), id,
        problems, classifiers and estimator_seeds. The latter three are unique,
        nonempty lists of registry names and unsigned 32-bit integer seeds.
    problems : dict
        Names mapped to dictionaries containing dataset (ClassificationDataset),
        evaluation (validated evaluation plan), nonempty provenance, and optional
        source_manifest (path-to-SHA256 mapping). Adapters prepare these inputs.
    classifiers : dict
        Names mapped to unfitted aeon classifier prototypes. Cloned prototypes
        receive the seed in every exposed random_state parameter.

    Returns
    -------
    dict
        Portable specification, input and effective estimator identities,
        evaluation assignments, ordered jobs and a content-derived plan_id.

    Notes
    -----
    Planning hashes already prepared arrays and implementation files. It does
    not open source signals, fit models, execute processing history or import
    tsml-eval. Source content is verified by the experiment adapter at execution.
    Only accuracy and balanced_accuracy profiles are currently executable.
    """
    spec = _specification(specification)
    for registry, names in (
        (problems, spec["problems"]),
        (classifiers, spec["classifiers"]),
    ):
        if not isinstance(registry, dict) or any(
            name not in registry for name in names
        ):
            raise ValueError(f"Registry does not supply all selected entries: {names}")
    inputs = {name: _problem(problems[name]) for name in spec["problems"]}
    estimators = {
        name: {
            str(seed): _portable(_seeded(classifiers[name], seed))
            for seed in spec["estimator_seeds"]
        }
        for name in spec["classifiers"]
    }
    jobs = []
    for problem in spec["problems"]:
        for classifier in spec["classifiers"]:
            for seed in spec["estimator_seeds"]:
                for fold in inputs[problem]["evaluation"]["folds"]:
                    jobs.append(
                        {
                            "id": f"job-{len(jobs):06}",
                            "problem": problem,
                            "classifier": classifier,
                            "seed": seed,
                            "fold": fold["id"],
                        }
                    )
    plan = _snapshot(
        {
            "format_version": 1,
            "specification": spec,
            "problems": inputs,
            "estimators": estimators,
            "jobs": jobs,
            "runner_sha256": _file_hash(__file__),
        }
    )
    plan["plan_id"] = _digest(plan)
    return plan


def _scores(dataset, evaluation, split, root, report, reader):
    result = reader(
        str(root / report["artifacts"]["predictions"]["path"]), verify_values=True
    )
    fields = evaluation["profile"]["identities"]["participant"]
    people = {}
    for row, index in enumerate(split["partitions"]["test"]["indices"]):
        person = json.dumps([dataset.cases[index][f] for f in fields], sort_keys=True)
        people.setdefault(person, []).append(row)
    functions = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    return {
        person: {
            metric: float(
                functions[metric](
                    np.asarray(result.class_labels)[rows],
                    np.asarray(result.predictions)[rows],
                )
            )
            for metric in evaluation["profile"]["metrics"]
        }
        for person, rows in people.items()
    }


def _summaries(rows):
    grouped = {}
    for row in rows:
        job = row["job"]
        key = (job["problem"], job["classifier"], job["seed"])
        for person, metrics in row["scores"].items():
            for metric, score in metrics.items():
                grouped.setdefault(key, {}).setdefault(metric, {}).setdefault(
                    person, []
                ).append(score)
    result = []
    for (problem, classifier, seed), metrics in grouped.items():
        values = {}
        for metric, people in metrics.items():
            participant_scores = {
                person: float(np.mean(scores)) for person, scores in people.items()
            }
            scores = list(participant_scores.values())
            values[metric] = {
                "participants": participant_scores,
                "mean": float(np.mean(scores)),
                "participant_sample_sd": (
                    None if len(scores) < 2 else float(np.std(scores, ddof=1))
                ),
            }
        result.append(
            {
                "problem": problem,
                "classifier": classifier,
                "seed": seed,
                "metrics": values,
            }
        )
    return result


def run_classification_benchmark(
    plan, problems, classifiers, output_dir, *, resume=False
):
    """Revalidate a benchmark plan and execute its jobs serially.

    Parameters
    ----------
    plan : dict
        JSON-restored output of plan_classification_benchmark.
    problems, classifiers : dict
        The prepared problem and classifier registries used during planning.
    output_dir : str or pathlib.Path
        Fresh suite directory outside source data. Jobs use their own experiment
        directories; benchmark.json records the plan and explicit job coverage.
    resume : bool, default=False
        Verify completed job artifacts through the experiment adapter and retry
        matching interrupted jobs. Changed plans and completed summaries fail.

    Returns
    -------
    dict
        Completion receipt with job identities, per-participant fold scores and
        equal-participant summaries. Fold scores are averaged within each person
        before averaging people, separately for each problem, classifier and seed.

    Notes
    -----
    Requires the optional tsml extra. The first failed job stops execution and
    leaves a failed receipt; later jobs remain pending. No tuning or concurrent
    execution is performed. Source authenticity and scientific suitability of
    adapter-supplied definitions remain explicit responsibilities.
    """
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if not isinstance(plan, dict) or "specification" not in plan:
        raise ValueError("Malformed benchmark plan")
    checked = plan_classification_benchmark(
        plan["specification"], problems, classifiers
    )
    if checked != plan:
        raise ValueError("Benchmark plan changed or is stale")
    FileLock, _, reader = _backend()
    root = Path(output_dir).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.with_name(root.name + ".lock")), timeout=0):
        previous = None
        if root.exists():
            if not resume:
                raise FileExistsError(
                    "Benchmark output exists; use resume=True or a fresh directory"
                )
            previous = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
            if previous.get("plan") != checked or previous.get("status") not in {
                "running",
                "failed",
                "complete",
            }:
                raise ValueError("Existing benchmark identity or state differs")
        else:
            root.mkdir()
        rows = []
        report = {
            "format_version": 1,
            "plan": checked,
            "status": "running",
            "jobs": rows,
            "coverage": {
                "planned": len(checked["jobs"]),
                "completed": 0,
                "failed": 0,
                "pending": len(checked["jobs"]),
            },
        }
        if previous is None or previous["status"] != "complete":
            _write(root / "benchmark.json", report)
        try:
            for job in checked["jobs"]:
                supplied = problems[job["problem"]]
                expected = checked["problems"][job["problem"]]
                # Earlier jobs must not change another prepared problem before
                # that problem's own experiment identity is calculated.
                if _problem(supplied) != expected:
                    raise ValueError("Prepared problem changed during benchmark")
                evaluation = expected["evaluation"]
                split = next(
                    f["split"] for f in evaluation["folds"] if f["id"] == job["fold"]
                )
                target = root / job["id"]
                receipt = run_classification_experiment(
                    supplied["dataset"],
                    {
                        role: rows["indices"]
                        for role, rows in split["partitions"].items()
                    },
                    _seeded(classifiers[job["classifier"]], job["seed"]),
                    target,
                    dataset_name=job["problem"],
                    classifier_name=job["classifier"],
                    provenance={
                        "benchmark_plan_id": checked["plan_id"],
                        "job": job,
                        "problem": expected["provenance"],
                    },
                    source_manifest=expected["source_manifest"],
                    eligible_indices=split["eligible_indices"],
                    resume=resume and target.exists(),
                    **split["policy"],
                )
                rows.append(
                    {
                        "job": job,
                        "identity_sha256": receipt["identity_sha256"],
                        "scores": _scores(
                            supplied["dataset"],
                            evaluation,
                            split,
                            target,
                            receipt,
                            reader,
                        ),
                    }
                )
                report["coverage"].update(
                    completed=len(rows), pending=len(checked["jobs"]) - len(rows)
                )
                if previous is None or previous["status"] != "complete":
                    _write(root / "benchmark.json", report)
            report.update(status="complete", summaries=_summaries(rows))
            if (
                previous is not None
                and previous["status"] == "complete"
                and report != previous
            ):
                raise ValueError("Completed benchmark summary changed")
            _write(root / "benchmark.json", report)
            return report
        except BaseException as exc:
            if previous is None or previous["status"] != "complete":
                report.update(
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                report["coverage"].update(
                    failed=int(len(rows) < len(checked["jobs"])),
                    pending=max(0, len(checked["jobs"]) - len(rows) - 1),
                )
                _write(root / "benchmark.json", report)
            raise
