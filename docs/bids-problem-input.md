# Inspect a local BIDS folder before planning cases

`inspect_bids_eeg` connects local EDF/BDF recordings and their BIDS tables to
[metadata-only problem planning](problem-planning.md). Source inspection and
scientific choices stay separate: the adapter has no problem, target mapping,
epoch, quality policy or evaluation configuration to infer.

Entity discovery uses [MNE-BIDS](https://mne.tools/mne-bids/stable/generated/mne_bids.find_matching_paths.html).
Sidecars use its [BIDSPath matching](https://mne.tools/mne-bids/stable/generated/mne_bids.BIDSPath.html)
and aeon-neuro's existing EEG JSON inheritance resolver. Supported native
headers are opened through the existing ingestion reader, without calling
`get_data` or hashing full signal files. All source handles are closed.

## Inspect and read the diagnostics

```python
from aeon_neuro.datasets import inspect_bids_eeg

report = inspect_bids_eeg("/data/nm000245", subjects=["1"])
for issue in report["issues"]:
    print(issue["severity"], issue["recording_id"], issue["message"])
```

The first supported encodings are EDF and BDF. Discovered BrainVision and
EEGLAB recordings produce a blocking diagnostic because their companion-file
snapshots are outside this increment. Other existing loaders are unchanged.
Filters `subjects`, `sessions`, `tasks`, `acqs` and `runs` use BIDS entity values
without prefixes. No matching recordings is an explicit error report.
Every supplied filter value must match within the combined selection. A partial
match such as `subjects=["1,", "2"]` is a blocking selection diagnostic, using
the same guard as `load_bids_data`, before any native header is opened.

The returned ordinary dictionary has:

| Field | Meaning |
| --- | --- |
| `version` | Inspection-report version, currently `1`; separate from the problem format version. |
| `ready` | Whether all selected sources passed this source inspection. It does not establish that a particular problem or evaluation is ready. |
| `source_ids` | Every selected recording's root-relative source path, in deterministic order. |
| `recordings` | Source descriptors when ready; `None` if any source has a blocker. |
| `events` | Original event columns with explicit recording and table-row IDs when ready; `None` if blocked. |
| `issues` | Dictionaries containing `severity`, `code`, `recording_id` and an actionable `message`. |

Malformed or ambiguous metadata, channel-order/rate conflicts, invalid event
timing and unreadable or unsupported sources block the catalogue. A broken
recording cannot silently disappear while the remaining subset looks ready.
Review the issues or select an explicitly narrower scope before trying again.

Missing optional participant/channel/event tables produce warnings. Known header
facts remain available; missing participant attributes, session identities and
event rows are not invented. An event-epoch problem still needs actual events.
Published discontinuities and duration discrepancies are reported and retained,
without automatic clipping, interpolation or source corrections. This is not
a complete BIDS validator or an artifact detector.

Descriptors retain actual channel order, rate, exact sample count, first sample,
native channel types and FIFF unit codes. Sidecars do not rescale or retype the
decoded samples. Resolved EEG JSON and field origins are retained separately,
along with dataset attribution/conversion metadata when available. Participant
attributes use literal flat keys such as `participant.cohort`; these are ordinary
binding column names rather than dotted lookup expressions.

Event rows retain their original columns, adding `recording_id`, `bids_event_id`
and zero-based `bids_event_row`. These IDs identify rows of the published table,
not recovered original acquisition trials. `onset` must be finite and on-grid;
any supplied `sample` and `onset_sample` must agree with it and one another.
Unknown `duration="n/a"` is preserved. Quality annotations require a separately
reviewed interpretation; their mere absence never means that trials are clean.

## Bind an explicit problem

For the local Cho2017 NEMAR derivative, the following choices yield raw imagery
cases. All published event rows are included; original MATLAB quality flags and
paper cohort exclusions are not applied. No model performance follows.

```python
from aeon_neuro.datasets import plan_classification_problem

if not report["ready"]:
    raise ValueError(report["issues"])

problem = {
    "format_version": 1, "id": "central-imagery", "version": "1",
    "question": "Classify left versus right hand imagery after the cue",
    "class_labels": ["left_hand", "right_hand"],
    "channels": ["C3", "Cz", "C4"],
    "segmentation": {"mode": "event_epochs", "tmin": 0.5, "tmax": 2.5},
}
binding = {
    "format_version": 1,
    "problem": {"id": "central-imagery", "version": "1"},
    "target": {
        "source": "event", "column": "trial_type",
        "mapping": {"left_hand": "left_hand", "right_hand": "right_hand"},
    },
    "group": {"source": "recording", "column": "participant_id"},
    "events": {"onset": "onset", "id": "bids_event_id"},
    "case_fields": {
        "participant_id": {"source": "recording", "column": "participant_id"},
    },
}
plan = plan_classification_problem(
    report["recordings"], problem, binding, events=report["events"]
)
```

The saved report and authored documents are sufficient for planning while
signal files are unavailable. The binding still decides targets and groups.
For a recording-level target, it can name `column="participant.cohort"` when
that participant attribute exists; the adapter does not interpret its meaning.

## Explicit selected materialization

```python
from aeon_neuro.datasets import (
    materialize_classification_problem,
    open_bids_eeg_recording,
)

chosen = [plan["cases"][0]["case_id"]]
dataset = materialize_classification_problem(
    plan,
    lambda descriptor: open_bids_eeg_recording("/data/nm000245", descriptor),
    case_ids=chosen,
)
```

Only selected recordings are opened. Saved metadata-file SHA256 checks, source
size/modification time and decoded headers are checked before returning an owned
lazy Raw handle. The materializer reads planned intervals and closes the handle.
It returns the existing `ClassificationDataset`, without changing the originals.

Source size/time and header agreement are **not full content verification**.
For trusted release checks, add `expected_sha256` to each recording descriptor
before making the plan. The opener verifies that signal checksum at explicit
materialization, reading every byte of the selected file. Inspection does not
invent a release checksum or transfer missing files.

File snapshots are local stale-input checks, not signatures or a filesystem
transaction. Keep inspected inputs immutable during use. Metadata additions do
not redefine a saved catalogue: reinspect to adopt new source files or corrected
metadata. Different paths may alias one original recording; original-source
identities and full processing supports still need an explicit evaluation policy.

## Format changes remain a separate responsibility

The adapter consumes BIDS sources and produces source facts; it does not parse
or migrate problem documents. Its report/snapshot version and the authored
`format_version` identify different contracts. New problem types can use the
same inspected facts when those inputs remain sufficient. Other modalities or
companion-file layouts need appropriate source adapters.

Generated case plans bind implementation identities and may need regeneration
after code changes. Retain the authored documents and source evidence when
replanning. Current support for format version 1 is not an automatic migration
promise for future formats.
