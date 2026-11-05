# Current case metadata and the Braindecode adapter

The shared BIDS information object describes original recordings. A constructed
classification problem can select or reorder their channels, so classifier
settings must come from the **current cases**, not the full source descriptors.
For example, a 64-channel source can yield `C4, C3, Cz` cases in that order.

The source example `examples/bids_braindecode_handoff.py` validates that the
current dense arrays, labels, groups, case rows, ordered names and rate agree.
It constructs an unfitted `BraindecodeClassifier` and a detached JSON record:

```python
from bids_braindecode_handoff import prepare_handoff

classifier, report = prepare_handoff(cases, geometry="none")
```

Run this snippet with `examples/` on the Python path. `cases` is the output of
`make_classification_problem(X, info, problem, binding)`. The report includes
input shape/dtype/hash, current rate/channel order, effective classifier settings,
case IDs and the original source/history/problem provenance. Hashing the prepared
case values reads those arrays; this is not metadata-only source planning.

## Explicit geometry choice

`geometry` is required:

- `none` passes no positions to this EEGNet. Current ordered channel names are
  still retained in the handoff record.
- `standard_1005` passes current case names to the adapter, which constructs
  template electrode positions for models accepting `chs_info`. These are not
  measured source positions. Original electrodes/coordinate-system metadata
  remains preserved in the source provenance.

No automatic alias mapping, reference change, resampling or amplitude scaling is
performed. This example accepts native-grid cases only. Changed case lengths,
rates or channel order must have a supported derived-layout mapping before they
can be described as those original cases. Layout checks cannot certify arbitrary
equal-shaped array edits.

The small EEGNet settings are explicit: CPU, one training epoch if fitted, batch
size four, seed zero and small convolution/pooling sizes. They exercise the
interface; they are not a recommended scientific model or training protocol.
The CLI does **not** fit or evaluate a model.

## Export one local source's handoff

Use a supported existing EDF/BDF BIDS source and explicit problem/binding files:

```bash
python examples/bids_braindecode_handoff.py /data/study \
  --subject 01 --problem imagery.problem.json --binding study.binding.json \
  --geometry none --output /results/handoff.json
```

Loading reads the complete selected source into RAM. The output must be fresh
and outside the source dataset. No download, quality-policy inference, split or
prediction is performed. An optional later model fit needs its own explicit
evaluation and training-scope decisions.

The generated BDF acceptance tests construct two 64-channel sources, select them
together with their metadata in reversed order, and build reordered three-channel
cases. They check the metadata handoff, source/provenance ownership, stale or
missing current facts, CLI export and a single synthetic CPU training epoch.
They check probability alignment without reporting an accuracy estimate.

```bash
python -m pytest examples/tests/test_bids_braindecode_handoff.py
```
