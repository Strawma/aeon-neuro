# Serial classification benchmarks

The experimental benchmark API schedules already prepared classification
problems and validated evaluation plans. It reuses the
[evaluation planner](evaluation-profiles.md) and
[optional experiment adapter](classification-experiments.md).
Dataset adapters remain responsible for source interpretation, problem/binding
definitions, eligibility and explicit processing. Reported processing history
does not execute transformations.

## Specify a suite

Save this UTF-8 JSON as `benchmark.json`:

```json
{
  "format_version": 1,
  "id": "session-baselines",
  "problems": ["imagery"],
  "classifiers": ["prior"],
  "estimator_seeds": [42]
}
```

All five fields are required; unknown fields are rejected. Names must start
with a letter or digit and contain only letters, digits, underscores, dots or
hyphens. Lists must be nonempty and contain no duplicates. Seeds are integers
from 0 through 4294967295, excluding booleans.

The names select explicit Python registries. This first format does not import
Python objects from JSON or resolve arbitrary dataset/configuration paths.
An adapter supplies the actual constructed dataset, saved evaluation plan,
nonempty scientific provenance and optional path-to-SHA256 source manifest.
Keep source roots and output locations outside the scientific specification.

```python
from aeon.classification.dummy import DummyClassifier
from aeon_neuro.experiments import (
    plan_classification_benchmark,
    run_classification_benchmark,
)

problems = {
    "imagery": {
        "dataset": dataset,
        "evaluation": evaluation_plan,
        "provenance": {"protocol": "fixed training to evaluation session"},
    }
}
classifiers = {"prior": DummyClassifier(strategy="prior")}
plan = plan_classification_benchmark("benchmark.json", problems, classifiers)
report = run_classification_benchmark(plan, problems, classifiers, "output/suite")
```

Use the same `ClassificationDataset` boundary with either the loaded-array
problem builder or a materialized shared-info problem plan. Adapters can retain
the source/history/problem provenance in the prepared problem's provenance.

## Planning and execution

Planning enumerates problem, classifier, seed and fold combinations in declared
order. It revalidates every evaluation plan and freezes case, label, group,
metadata and sample identities, effective estimator parameters and implementation
hashes. Planning reads already prepared arrays to hash them; it does not open
source signals, fit estimators or require tsml-eval. This is distinct from
metadata-only planning before signal materialization.

Each classifier is cloned, with the seed applied to every exposed `random_state`
parameter, including nested estimators. Prototypes stay unchanged. A classifier
without such parameters has no controlled random state; hidden randomness,
global RNG state and nondeterministic numerical kernels are not controlled.
Seed repetitions remain separate in results and are never counted as people.

Execution requires `pip install '.[tsml]'` on Python 3.12+. It revalidates the
whole plan before creating output, runs jobs serially and delegates cloning,
train/test validation, source checksum verification and prediction writing to
the existing adapter. Supplied source paths are runtime locators and should be
absolute. Use an output directory outside the source dataset.

Only `accuracy` and `balanced_accuracy` evaluation profiles are supported in
this increment. Other declared metrics fail during planning. Scores use the
adapter's probability-argmax decisions. Fold scores are averaged within each
declared participant, then participants receive equal weight, separately for
each problem, classifier and seed. The receipt retains participant values and
their sample standard deviation; one participant has no sample SD. Problems
are not pooled into a cross-dataset score.

## Receipts and recovery

`benchmark.json` contains the frozen plan, ordered job receipts, scores and
planned/completed/failed/pending coverage. Each `job-NNNNNN` directory contains
ordinary experiment receipts, native tsml-eval predictions and a case map.
The first failure stops execution, preserving completed jobs and pending work.
An OS file lock protects each suite; the experiment adapter also locks jobs.

```python
report = run_classification_benchmark(
    plan, problems, classifiers, "output/suite", resume=True
)
```

Resume revalidates the plan, verifies every completed job's checksums and
prediction mapping, and retries matching interrupted jobs. It recalculates
scores from verified predictions and rejects a changed completed summary.
Changing cases, samples, provenance, effective parameters, assignments or runner
code requires a fresh plan and output directory. These checks detect stale or
changed artifacts; they do not authenticate an untrusted adapter or establish
that the scientific protocol is appropriate.

This API does not add tuning, concurrency, automatic data acquisition, a pipeline
DSL or a frontend. Those can build on the explicit problem and job boundaries.
