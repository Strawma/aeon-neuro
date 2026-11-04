# Classification experiments

`run_classification_experiment` connects an existing `ClassificationDataset`,
explicit train/test assignments and an aeon classifier to tsml-eval prediction
files. It checks source support, preserves case/class mappings and verifies
completed output before reuse.

Install from the source checkout with `pip install '.[tsml]'`. The published
backend requires Python 3.12+ and aeon below 1.6; use an isolated environment.

See the [experiment guide](../../docs/classification-experiments.md) for the API,
optional source manifests, output identities and recovery semantics.

`plan_classification_benchmark` expands a strict JSON specification over named
prepared problems, classifier prototypes, seeds and validated evaluation folds.
`run_classification_benchmark` executes these jobs serially with explicit
coverage, verified recovery and equal-participant summaries. The specification
selects registries supplied by an adapter; it does not load or preprocess data.
See the [benchmark guide](../../docs/classification-benchmarks.md).
