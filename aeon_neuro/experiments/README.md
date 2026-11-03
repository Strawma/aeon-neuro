# Classification experiments

`run_classification_experiment` connects an existing `ClassificationDataset`,
explicit train/test assignments and an aeon classifier to tsml-eval prediction
files. It checks source support, preserves case/class mappings and verifies
completed output before reuse.

Install from the source checkout with `pip install '.[tsml]'`. The published
backend requires Python 3.12+ and aeon below 1.6; use an isolated environment.

See the [experiment guide](../../docs/classification-experiments.md) for the API,
optional source manifests, output identities and recovery semantics.
