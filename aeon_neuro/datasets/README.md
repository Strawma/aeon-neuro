# Classification problems on loaded recordings

For metadata-only planning and selected signal loading, use
`plan_classification_problem` followed by `materialize_classification_problem`.
Both share the eager builder's case and binding rules. See the
[planning guide](../../docs/problem-planning.md) for source-opening and memory
boundaries; planning does not fetch or read signal arrays.

Pass `EEGDatasetInfo` directly to the planner to retain source facts, its events
and reported history. Shared-info materialization produces the eager builder's
same source/history/problem provenance sections, recording the deferred entrypoint
and selected-case plan ID separately. Plain descriptor/event inputs remain usable.

`inspect_bids_eeg` reads local EDF/BDF headers and BIDS sidecars without loading
signal arrays. `load_bids_dataset` loads those sources with `EEGDatasetInfo`,
preserving dataset/participant facts, channel/event tables and source snapshots.
`info.select(X, selector)` copies signals and their metadata together.
`info.with_history(...)` appends caller-reported processing to a detached view;
it does not perform processing or update derived layouts.
See the [source inspection guide](../../docs/bids-problem-input.md) and
[loaded metadata guide](../../docs/bids-dataset-info.md). Both routes reuse the
existing source reader and preserve the legacy `load_bids_data` interface.

`make_classification_problem(X, info, problem, binding)` uses that shared object
directly for event or recording targets. It retains explicit class/group choices
and returns source, reported history and effective problem configuration together
under `dataset.metadata["provenance"]`. The plain-input loader remains available.

`load_classification_problem(recordings, metadata, problem, binding, events=...)`
constructs the existing `ClassificationDataset` from explicit problem choices.
Definitions and bindings can be dictionaries or JSON filenames. Fixed windows
use recording targets; event epochs can use event or recording targets.

The first format is experimental. It reuses the existing case builders and adds
no runtime dependencies. The [problem guide](../../docs/problem-format.md)
starts with runnable examples and lists every supported field and constraint,
including optional identities and explicit pandas target-code conversion.

Run `python examples/problem_loading.py` from a source checkout for a generated
example with two event targets, a cohort-window problem and a separately
validated participant-held-out split.
