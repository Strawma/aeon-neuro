# Planning neuro classification evaluations

An evaluation profile describes what should generalise, how folds are formed,
and how scores are averaged. It sits beside the task definition, so the same
left/right imagery task can have separate within-session and held-out-person
experiments. Case construction still supplies targets, eligibility and source
identities. Existing problem constraints remain mandatory.

`aeon_neuro.model_selection` plans assignments without fitting models or requiring
MOABB, tsml-eval or filelock. It returns assignments for a separate execution
layer. These presets are explicit native policies; selecting a similar name does
not reproduce every MOABB setting.

## Supported profiles

| Protocol | Splitter | Evaluation scope |
| --- | --- | --- |
| `within_session` | `stratified_kfold` | Separate folds for every participant/session; split single-label whole trials, then expand to their windows. |
| `within_session` | `explicit` | Replay exactly one saved complete train/test holdout for each participant/session in scope. |
| `cross_subject` | `leave_one_participant_out` | Train on the remaining people and test all sessions from one held-out person. |
| `cross_session` | `leave_one_session_out` | For each person, train on their other sessions and test one held-out session. |

Missing participant/session/trial identities fail. Trial identities are scoped
by participant and session; session labels need not be globally unique.
Participant mappings may contain multiple fields where the source requires a
compound identity. These mappings assert actual identities, not merely filename
prefixes. Unknown links between people or copies of recordings are not inferred.

The stratified preset counts independent trials per class, rather than windows.
Every window from one trial stays together. Trial labels must be constant for
this preset. All presets require the task's class vocabulary in every training,
test and scored test-session partition. Too few trials, people, sessions or
classes fail; no seed retry, skipped person or weaker split is substituted.
Use separately reviewed supplied splits for more complicated settings.

## Plan before fitting

For example, constructed Cho2017 cases can use these explicit field mappings:

```python
import json
from pathlib import Path

from aeon_neuro.model_selection import (
    plan_classification_evaluation,
    validate_classification_evaluation,
)

profile = {
    "format_version": 1,
    "id": "imagery-within-session-5fold",
    "protocol": "within_session",
    "identities": {
        "participant": ["bids_subject"],
        "session": ["session"],
        "trial": ["recording_id", "event_id"],
    },
    "splitter": {"name": "stratified_kfold", "n_splits": 5, "random_state": 42},
    "metrics": ["accuracy", "balanced_accuracy", "roc_auc"],
    "positive_label": "right_hand",
    "aggregation": "equal_participant",
}
plan = plan_classification_evaluation(
    dataset,
    profile,
    disjoint_by=[("recording_id", "event_id")],
    source_fields=("canonical_source", "support_start_sample", "support_stop_sample"),
)
Path("evaluation-plan.json").write_text(json.dumps(plan, indent=2))
plan = validate_classification_evaluation(dataset, plan)
```

Supply `eligible_indices` to restrict the global scope, including a parent
training pool for inner planning. All folds are materialised and validated before
the planner returns. Canonical case/trial identities drive generated ordering;
the plan saves actual global indices and ordered case IDs, not just the seed.

Replay uses `splitter={"name": "explicit"}` and
`partitions=[{"train": saved_train, "test": saved_test}, ...]`. It preserves the
saved order, covers every eligible participant/session once and draws no new
holdout. Other preset/splitter combinations and unknown settings are rejected.

The plan binds its profile, scope, source-support fields, effective independence
rules, case inventory/labels/groups, measurement metadata, implementation and
splitter runtime. Revalidation regenerates and compares the full plan; changed
cases, assignments, code or runtime fail. Signal values are not inspected by the
planner. The experiment adapter separately binds and verifies the actual arrays.

## Hand off a saved plan

Each fold provides global indices and the effective split policy:

```python
for fold in plan["folds"]:
    split = fold["split"]
    partitions = {
        name: partition["indices"]
        for name, partition in split["partitions"].items()
    }
    # Pass these assignments and split["policy"] to the execution layer.
```

Revalidate the saved plan against the current cases before executing its folds.
The planner does not bind signal values, fit classifiers, write predictions or
compute metrics. The separate execution layer must bind the arrays and preserve
fold identities and prediction-row mappings.

The profile records requested accuracy, balanced accuracy and/or binary ROC AUC.
AUC requires an explicit positive class. Its declared `equal_participant` policy
means scoring each fold/test-session, then averaging folds within sessions,
sessions within people and people equally. These fields describe the reporting
contract; this module does not execute that aggregation. Probabilities from
independently fitted folds should not be pooled for AUC, and fold variation does
not supply independent replicates for a confidence interval.

## Scientific scope

These presets do not implement hyperparameter search, calibration/adaptation,
chronological prediction, repeated resampling, multiclass AUC or fitted-model
serialization. Inner planning can use the parent training scope, but no automatic
nesting or routing into estimator-internal CV is provided. Earlier learned
preprocessing, unknown identity aliases and source-coordinate assertions remain
explicit responsibilities. Planning cannot undo source conversion that used
labels or data across evaluation boundaries.

Compare with [MOABB's evaluation concepts](https://moabb.neurotechx.com/docs/api.html#evaluations)
and [scikit-learn's grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
MOABB compatibility should compare the complete case population, preprocessing,
assignments, fitting, metrics and aggregation under pinned versions. No MOABB
runtime dependency is needed for this planner.

An adapter may preserve MOABB-style `subject` and `session` metadata in case
records and map those fields to the participant/session roles. Original trial
identity and processing-support coordinates must also be supplied. Subject and
session columns alone cannot reconstruct those facts from anonymous arrays.
