Classification recipes: experimental prototype
===============================================

This branch implements the GDP discussion recipe ``0.3.0-draft``. It is an
experimental interface, not an accepted BIDS extension. It builds on the case
container and bounded-array validation from Pair 2's PR #241, currently an
unmerged dependency. The recipe adapter adds BIDS-table interpretation,
versioned defaults, interval reads, provenance and reproducible group splits.

Install the optional JSON Schema validator with ``pip install -e '.[recipes]'``.
The normal package can be imported without that extra; recipe use requests it.

Small author input
------------------

.. code-block:: json

   {
     "spec_version": "0.3.0-draft",
     "target": {"source": "participants.tsv", "column": "group"},
     "segmentation": {"mode": "fixed_windows", "window_seconds": 2}
   }

The caller supplies the BIDS root separately. This example deliberately uses
full-recording windows, source label strings and trusted BIDS-declared EEG
channels. It does not infer that a recording is resting or eyes closed. Missing
channel roles, missing/ambiguous participant joins and multiple unspecified
tasks are errors; explicitly select names/prefixes or tasks when appropriate.

The default partition is 80% train / 20% test, seed 42, shuffled at whole
``bids_subject`` level. All sessions and runs of a subject stay together. A BIDS
subject may identify a pair; the default does not assert that it is one person.
One available group is insufficient; there is no per-window fallback. For
validation, supply a complete fraction triple such as 0.6/0.2/0.2. Fractions
apply to group counts. Target stratification is an explicit optional setting and
requires one label per group. No seed retries or silent class/group changes are
performed. The selected backend is scikit-learn ``train_test_split`` on canonical
group IDs, holding out test groups first and then relative validation groups.

No signal filtering, resampling, re-referencing, normalisation, baseline
correction or artifact rejection is added. Models and learned transformations
must be fitted on training data only. Primary metrics and aggregation of window
predictions into participant decisions are separate scientific choices.

Usage
-----

.. code-block:: python

   from aeon_neuro.datasets import load_bids_classification
   from aeon_neuro.classification.feature_based import CSPClassifier

   result = load_bids_classification(
       "/data/bids", "recipe.json", output_dir="/results/new-run"
   )
   X, y = result.dataset.X, result.dataset.y
   train, test = result.splits["train"], result.splits["test"]
   model = CSPClassifier(n_components=4, reg="oas", cov_est="epoch")
   model.fit(X[train], y[train])
   predictions = model.predict(X[test])

``result.dataset`` is Pair 2's ``ClassificationDataset``. Original file-relative,
zero-based half-open source intervals and identities remain in ``cases``.
The adapter reads only selected intervals, validates them through the shared
builder, and writes a memory-mapped ``X.npy`` when an output directory is given.
An in-memory allocation over 512 MiB is rejected unless that limit is explicitly
overridden. A model may still materialise indexed training data; the recipe
loader's memory bounds do not establish a classifier's memory footprint.
Embedded EEGLAB sources may be preloaded by their underlying reader; this fact
is recorded. EDF/BDF layout guards run before decoding, reusing the native helper.

The new output directory must be outside raw BIDS. It retains ``X.npy``,
``y.npy``, ``cases.tsv``, ``exclusions.tsv``, ``split-assignments.tsv``,
``resolved-recipe.json`` and ``run-report.json``. Reports include actual selected
channels/rates, source label dictionaries, setting origins, source and module
hashes, versions, exclusions, adjustments and class/group counts. Planning
errors occur before output creation. An extraction/write failure retains failed
status; never consume arrays from an incomplete run. Retrying needs a new
destination. Source fingerprints are taken before metadata interpretation and
checked after planning, hashing and extraction. Source changes are rejected.
Physical signal identity is checked separately from BIDS header identity, so
two BrainVision headers referencing the same signal cannot conceal overlapping
train/test samples. These checks do not establish independence of copied or
otherwise correlated recordings; subject identity still needs source review.

Run the same utility with ``output_dir=None`` for an in-memory result with its
metadata returned instead of written. Recipes and input files remain untouched.

Included COVID and RPS examples
------------------------------

``examples/recipes/covid.classification.json`` selects 19 named scalp channels
and two-second windows within the initial eyes-closed interval, capped at
10--480 seconds. State meaning, source-specific discontinuity exclusions and
permission to truncate a short recording are explicit choices. It uses source
``participants.tsv:group`` labels and participant-group stratification. Supply
enough eligible participants from both cohorts; a one-person inspection cache
cannot provide a subject-held-out classification problem. This does not
reproduce an existing archive or establish diagnostic validity.

``examples/recipes/rps.classification.json`` selects pair 31, both ``1-`` and
``2-`` channel-name banks, and ``player1_resp`` as the target. Source strings
1/2/3 are labels, with rock/paper/scissors descriptions retained from the source
dictionary. Code 0 is explicitly excluded. Decision-onset epochs are [0,2)
seconds. Blocks are defined from original trial numbers, in bins of 40 with
origin 1, before exclusions; whole blocks stay together. This evaluates joint
EEG on held-out blocks of that pair, not future blocks or unseen people.
Prefixes alone do not establish which bank belongs to a player. Broader RPS
layouts and single-player correspondence require separate review.

From this source checkout, with its dependencies available:

.. code-block:: bash

   PYTHONPATH="$PWD" python examples/recipes/run_recipe.py \
       /data/pinned-rps-bids examples/recipes/rps.classification.json \
       /results/new-rps-run --classifier csp

The example fits Pair 3's merged CSP classifier with LDA and a most-frequent
baseline on the same train/test assignments, saving metrics and held-out case
predictions. Parameters are declared before fitting and the test is not used
for selection. This is an engineering pilot, not a general benchmark. CPU time
and memory depend on the selected data and classifier.

Additional dataset templates
----------------------------

``examples/recipes/ds004504.classification.json`` selects twelve participants
from OpenNeuro ds004504, version 1.0.9: the first four source IDs in each of
groups A, C and F. It uses original eyes-closed EEGLAB recordings and the source
``participants.tsv:Group`` labels, with two-second windows over [10,70) seconds
and participant-level target stratification. The source describes these groups
as Alzheimer's disease, controls and frontotemporal dementia. These choices
exercise three-class loading; a small split cannot establish clinical utility.
Use original participant folders rather than preprocessed derivatives.

``examples/recipes/eegmmidb.classification.json`` expects a BIDS copy prepared
from PhysioNet EEG Motor Movement/Imagery version 1.0.0, subjects S001--S006,
and only runs 04, 08 and 12. Prepare each native EDF with
``aeon_neuro.utils.bids.convert_eeg_to_bids``, using BIDS subject IDs 001--006,
task ``leftrightimagery`` and the original run number. Its EDF annotations
become ``events.tsv:trial_type``. For these particular runs, T1/T2 mean imagined
left/right fist and T0 is rest. The recipe excludes T0, takes [0,4) second
epochs and keeps all runs of each participant in one partition. Other runs
have different label meanings and must not be mixed under this mapping.

These templates use the same loader and classifier runner as COVID and RPS.
Dataset acquisition and source eligibility remain separate steps. Compare
against the saved baseline and inspect participant/group counts as well as
case counts; no test-set tuning or automatic scientific-protocol selection is
performed.

Optional alternatives
---------------------

Use ``evaluation.split`` with ``strategy: chronological`` and explicit
``order_by`` fields for a temporal holdout. Groups must have a constant ordering
key. There is no random seed or stratification in this mode.

For replay only, ``method: supplied_manifest`` accepts a relative TSV, explicit
``join_keys`` and an optional ``split_column`` (default ``split``). The file is
relative to the recipe file, not the current directory. Duplicate/missing or
unused identity keys are errors. A group-level assignment can cover multiple
cases; a manifest must agree with the selected included identities. All modes
check group isolation, class coverage and actual sample overlap across partitions.
The strict Pair 2 ``validate_group_split`` remains unchanged: its independent
recording policy is deliberately different from an explicit within-file block
holdout, so the adapter validates the latter without weakening that helper.

Schema, defaults and limits
---------------------------

The packaged ``classification_recipe.schema.json`` validates types and supported
keys; ``resolve_classification_recipe`` then supplies omission defaults explicitly.
See also :doc:`recipe_format` for the minimum author input and omission defaults.
JSON Schema default annotations do not themselves fill values. Actual source
facts are resolved at BIDS inspection. The draft supports participant/event
targets, fixed windows, event epochs, named/type/prefix channel selection and
explicit integer-bin event groups. It rejects incompatible channels/rates/case
lengths instead of intersecting, interpolating, relabelling or resampling them.
Source sample columns are all checked against onset and each other; missing
columns can be derived, contradictory ones cannot. Complete windows only are
retained, without padding. Explicit event ID columns must be unique per recording;
omit them to use source-row identities when markers share an event code.

State intervals require their source vocabulary. The first applicable state
must be the declared initial state; ambiguous or missing evidence follows the
explicit error/exclusion policy. Additional experimental markers in an intended
rest interval, cohort confounds, artifacts and scientific sufficiency remain
research-review questions. Structural validation and a successful model run do
not answer them. Failures during planning raise before writing; failed-attempt
reporting currently begins once the output array stage is planned successfully.
