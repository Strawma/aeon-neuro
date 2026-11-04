# Loaded BIDS recordings and aligned metadata

`load_bids_dataset` loads local EDF/BDF recordings and returns their complete
source metadata through `EEGDatasetInfo`. It reuses the existing
[source inspector](bids-problem-input.md), snapshots and source opener. Loading
does not infer a target, change channel units/order, apply a montage or preprocess
signals. The existing `load_bids_data` interface and its format support are unchanged.

```python
from aeon_neuro.datasets import load_bids_dataset

X, info = load_bids_dataset("/data/study", subjects=["01", "02"])
print(info.source_ids, info.participant_ids)
for recording in info.recordings:
    print(recording.recording_id, recording.sfreq, recording.channel_names)
```

This explicit operation reads complete selected recordings into RAM. Equal
shapes, channel names/order and exact sampling rates produce a 3D NumPy array
in `(recording, channel, sample)` order. Otherwise the result is a list of 2D
arrays; no padding, reordering or resampling makes them stackable. Every owned
source handle is closed, including when a later recording fails to load.

For large sources, retain metadata-only inspection and selected case
materialization instead. `EEGDatasetInfo(inspect_bids_eeg(...))` creates a metadata
view without reading signals, but does not itself load or verify signal content.
The constructor requires a ready version-2 inspection report and checks selected
source/event identities and coordinates. Inspection failures remain explicit
errors, including their original blocking messages.

## Inspect source facts

The metadata views expose detached Python dictionaries and rows, preserving the
portable inspection representation. They do not require a new pandas interface.
Unknown optional facts stay `None`; source rows and `n/a` values retain their meanings.

| View | Contents |
| --- | --- |
| `info.description`, `info.readme` | Complete source dataset description and README. |
| `info.participants`, `info.participants_json` | Complete source participant table (`columns`/`rows`) and column dictionary, including unselected participants. |
| `info.dataset` | Dataset facts, file identities and selected participant scope. |
| `info.source_ids` | Source recording IDs in signal order. |
| `info.participant_ids` | One original `sub-` participant ID per recording, in signal order. Multiple recordings may share one participant. |
| `info.metadata`, `info.events` | Detached descriptors/event rows accepted by existing problem/case APIs. |
| `info.recordings` | Tuple of `RecordingInfo` views in signal order. |
| `recording.entities` | Source subject/session/task/acquisition/run values, without prefixes. |
| `recording.sfreq`, `recording.channel_names` | Decoded rate and ordered source channel names. |
| `recording.channels`, `recording.eeg_json` | Original channel rows and resolved source EEG JSON. |
| `recording.events` | Source event rows, including exact `bids_onset_sample` coordinates. |
| `recording.electrodes`, `recording.coordsystem`, `recording.scan` | Optional source geometry and matched scans row. |

`recording.bad_channels()` returns only channels explicitly marked `bad`, raising
when statuses are absent or unrecognized. `recording.channels_of_type("EEG")`
uses declared channel-table types and raises when those facts are unavailable.
Neither method certifies clean EEG or chooses channels automatically. Full value
origins, diagnostics and source snapshots remain available in `info.metadata`
and the detached version-2 report returned by `info.to_dict()`.

Changing a returned dictionary/list does not mutate the stored metadata. Source
facts are not rewritten to describe arbitrary external array operations. For
example, a source `EEGReference` remains an acquisition fact after a caller
independently rereferences an array. Use explicit history records to document
processing without changing those source facts.

## Select signals and metadata together

```python
X_selected, selected_info = info.select(X, [1, 0])  # reverse source order
X_subject, subject_info = info.select(X, [True, False])
X_empty, empty_info = info.select(X, [])
```

`select` accepts a slice, unique nonnegative integer indices, or a boolean mask
matching the recording count. Slice semantics follow Python, including reverse
slices. Negative integer indices, duplicate recordings, mixed boolean/integer
selectors and incompatible shapes raise errors. A selection copies both the
signal arrays and metadata so later edits cannot change the parent. This can
temporarily require substantial memory. Dense arrays retain their recording
axis; list/tuple inputs return a list of separate arrays.

Events follow the requested recording order, preserving their original IDs and
within-recording row order. The selected participant scope and recording-specific
diagnostics are updated; the complete source participant table is retained.
An empty view is valid and serializable, although classification construction
requires eligible observations. It cannot be used as an empty training dataset.

`info.validate_layout(X)` checks recording counts, sample/channel dimensions and
numeric types. These checks cannot detect an external permutation of equal-shaped
arrays or same-shape value edits. Keep X and info together and use paired selection;
this API does not fingerprint arrays on every operation. Source metadata inspection
also does not certify unchanged signal content without a trusted content hash.

## Make a classification problem

For sources too large to load completely, pass this same object to
`plan_classification_problem` and later materialize selected cases. The
[planning guide](problem-planning.md) describes metadata-only operation, saved
plan validation and eager/deferred provenance parity.

Scientific labels and grouping remain explicit in the ordinary problem and
dataset-binding documents. Once those choices are authored, pass the shared
metadata object to the experimental in-memory builder:

```python
from aeon_neuro.datasets import make_classification_problem

cases = make_classification_problem(
    X_selected,
    selected_info,
    "imagery.problem.json",
    "study.binding.json",
)
```

Event epochs take rows from `selected_info.events` automatically. Fixed windows
use recording facts and do not pass those events to the case builder. The
function reuses the existing problem/binding and case rules, including explicit
exclusions and exact sample-grid validation. It does not reopen BIDS files.

Binding sources remain `recording` and `event`. For a participant-level cohort,
use a recording target column such as `participant.group`, with an explicit
mapping from source groups to class names. For subject-level evaluation groups,
bind the recording column `participant_id`. These literal keys come from the
inspector's participant join; the dot is part of the key, not a path expression.
Recording entity fields include `subject`, `session`, `task`, `acq` and `run`.
An absent participant value or optional entity is not inferred.

The result is the existing `ClassificationDataset`. Its
`metadata["provenance"]` is one version-1 record with these sections:

| Field | Contents |
| --- | --- |
| `version` | Integer provenance schema version, currently `1`. |
| `source` | Selected version-2 source report, excluding its optional history extension, with original dataset facts, events and snapshots. |
| `history` | Complete reported processing records, including earlier application scopes outside a later selection. |
| `problem` | Effective problem/binding documents and their existing SHA256 configuration identities. |
| `construction` | Builder implementation identifier and aeon-neuro version. |

The configuration is stored once under `provenance.problem`. Source facts and
processing accounts retain separate meanings within this record. None of these
sections authenticates external array contents or establishes leakage-free
processing. Output signals and metadata are detached from the inputs; the result
container itself remains mutable.

`load_classification_problem(X, metadata, problem, binding, events=...)` remains
available for plain recording/event inputs, with its existing
`metadata["problem"]` output. It does not automatically include dataset-level
source facts or reported history. The new object route preserves those facts
without introducing a second parsing or segmentation implementation.

See the [problem guide and field reference](problem-format.md) for supported
choices and complete examples. This
increment does not add automatic labels, evaluation groups, DataFrame views
or companion-file source formats.

## Record reported processing

After independently performing an operation, use `with_history` to describe it.
The method returns a new metadata view; it does not mutate the parent or perform
the operation itself. For example, record an explicit scale conversion:

```python
import numpy as np

X_scaled = [recording * 1e6 for recording in X]
scaled_info = info.with_history(
    "scale",
    {"factor": 1e6, "input_units": "V", "output_units": "uV"},
    implementation="numpy.multiply",
    version=np.__version__,
    fit_scope="none",
)
print(scaled_info.history)
```

Supply the actual implementation version and effective parameters. Each record
uses the following version-1 schema:

| Field | Meaning |
| --- | --- |
| `version` | Record schema version, currently integer `1`. |
| `step` | Nonempty caller-supplied operation name. |
| `parameters` | Dictionary of effective parameters using portable JSON values. |
| `implementation`, `implementation_version` | Caller-supplied implementation identifier and its version. |
| `fit_scope` | Caller-supplied description of fitting inputs, `none` for an unfitted operation, or `unspecified` when omitted. |
| `source_ids` | Recording source identities selected when the record was appended. |

Parameters accept string-keyed dictionaries, lists, strings, finite floats,
integers, bools and `None`. Convert NumPy values explicitly. Tuples, non-string
keys, recursive structures and nonfinite numbers raise errors instead of changing
meaning during serialization. Names and versions must be nonempty strings.
Records do not add a wall-clock timestamp; callers may include one in parameters
when needed. An implementation version is required on each record.

`info.history` returns detached records in append order, initially an empty list.
`to_dict()` preserves the version-2 source report, adding the optional `history`
list only when supplied or appended. Ordinary inspector reports are unchanged.
`EEGDatasetInfo(json.loads(json.dumps(scaled_info.to_dict())))` validates and
restores both source facts and history. Loading source data does not claim any
unreported preprocessing.

Save this metadata snapshot alongside derived outputs, or retain the combined
provenance returned by `make_classification_problem`. Passing only `info.metadata`
and `info.events` to the older case APIs does not copy dataset-level history.
No `.ts` writer automatically includes the new provenance record.

Paired selection preserves complete earlier records, including source identities
outside the new subset. Narrowing those identities could conceal that processing
used a larger population. A later record identifies only the sources selected
at that later step. Empty selections preserve existing history but cannot append
a new step. The log does not automatically record selection operations.

History is a caller's account, not evidence that an operation occurred or that
fitting avoided leakage. It does not fingerprint arrays, generate derived data
identities or detect external mutation. Source channel units, reference, rate,
event coordinates and file snapshots remain source facts. A crop or resampling
record does not make the original metadata describe a changed signal layout;
`validate_layout` continues to reject incompatible shapes. Processing helpers,
derived-layout updates, automatic transformer tracking and `.ts` header formatting
remain separate features.
