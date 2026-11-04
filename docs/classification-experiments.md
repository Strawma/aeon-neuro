# Running explicitly defined classification experiments

The optional experiment adapter accepts an existing `ClassificationDataset`,
explicit assignments and an aeon classifier. It validates the assignments,
clones the classifier and writes an ordinary tsml-eval prediction CSV with a
separate case map and completion receipt. It does not choose a problem, split
data, infer a neurophysiological target or launch a benchmark campaign.

## Install

From the source checkout:

```bash
pip install '.[tsml]'
```

The `tsml` extra requires Python 3.12 or later. The published tsml-eval 0.7
release requires `aeon<1.6`, so install this optional combination in an isolated
environment when another project uses aeon 1.6. The exercised published
combination uses Python 3.12, aeon 1.5 and scikit-learn 1.8. Core imports and
case construction remain usable without tsml-eval or filelock.

## Keep the evaluation question explicit

Start with the constructed `dataset` in
[the assignment example](classification-problems.md). That example deliberately
tests a new session from the same person. Other questions need other policies:

| Evaluation | Assignment and identity rule |
| --- | --- |
| Within one person/session | Restrict the scope to that person/session and keep whole trials independent. |
| Held-out people | Keep participant identity disjoint in the outer split and in any inner model selection. |
| New sessions from known people | Keep person/session identity disjoint and declare the intended training and testing session roles. |

The problem states the intended generalisation. A caller supplies the actual
fold assignments and relevant source identities; the validator checks those
assertions. A valid tsml-eval file does not establish that the scientific
protocol is suitable. Report scores per declared unit, for example per person,
before applying the chosen cohort aggregation. The adapter does not pool people
or treat resample file names as evidence of participant independence.

```python
from pathlib import Path

from aeon.classification.dummy import DummyClassifier
from aeon_neuro.experiments import run_classification_experiment

output = Path("outputs/generated-session-dummy")
report = run_classification_experiment(
    dataset,
    {"train": [0, 1], "test": [2, 3]},
    DummyClassifier(strategy="prior", random_state=0),
    output,
    dataset_name="GeneratedSession",
    provenance={
        "question": "Predict the class in a new session from the same person",
        "protocol": "Fixed T/E session assignment",
        "score_unit": "person-session",
    },
    disjoint_by=[("participant_id", "session")],
    source_fields=("recording_id", "start_sample", "stop_sample"),
    partition_values={"train": {"session": ["T"]}, "test": {"session": ["E"]}},
)
prediction_file = output / report["artifacts"]["predictions"]["path"]
case_map_file = output / report["artifacts"]["case_map"]["path"]
```

For an inner split, supply `eligible_indices` containing only the parent
training pool. Full filter and sequence contexts belong in `source_fields`,
even if returned epochs are shorter. Already learned preprocessing and any
estimator-internal tuning remain caller responsibilities. Split validation
cannot discover unknown aliases or recover missing original source coordinates.

## Output and identities

One call evaluates one classifier and one supplied fold. Signals must be finite,
dense arrays in `(case, channel, time)` order. Pass an explicit aeon pipeline
for sklearn models; this prevents tsml-eval from silently choosing channel
flattening or padding. No external data transform, automatic resampling,
train-error estimation or training-file CV is enabled. `resample_id=0` only names
the backend file for this already supplied assignment.

Stored decisions follow probability argmax in encoded training-class order,
which may differ from an estimator's `predict`. The reader checks truths,
probability shape/range/sums and decisions before completion. `case-map.json`
retains original class meanings and ordered prediction-row/case-ID joins.
The native CSV can feed tsml-eval's other metrics and evaluators; the receipt's
accuracy is a convenience, not a replacement for protocol-specific scoring.

Run identity covers input arrays, case/exclusion/measurement metadata, exact
ordered assignments, estimator parameters and implementation, caller provenance,
adapter/backend code and runtime versions. Metadata and parameters must have
stable JSON-compatible identities; arbitrary objects and callables are rejected.
Array parameters are identified by shape, dtype and content hash, not serialized
as an executable model. Copy or save the original configuration separately when
reconstruction requires those values.

Optional `source_manifest={Path(...): "expected_sha256", ...}` checks original
recordings, annotations or model files before and after fitting and on resume.
Without a source manifest, identity covers the supplied arrays and metadata;
it does not certify unchanged acquisition files. Hash verification reads full
files and arrays. The adapter does not create an archive of the raw data or
serialize a trained estimator.

## Completion and recovery

By default an existing experiment directory raises `FileExistsError`. With
`resume=True`, matching intact completed output is verified and reused without
fitting. Changed inputs, settings, code or runtime reject reuse. Corrupt completed
artifacts are surfaced and retained; choose a fresh output after investigating.

Failed or interrupted matching runs start a new uniquely named attempt while
retaining earlier artifacts and per-attempt diagnostics. An OS file lock prevents
two callers from updating the same run concurrently and is released if a process
dies. Use a filesystem supporting these locks; this is not a distributed job
scheduler. The final receipt is written atomically only after source, input and
result validation. A partial CSV is never considered completed work.
