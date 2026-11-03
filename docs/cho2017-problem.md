# A classification problem on local Cho2017 recordings

This walkthrough uses the [problem/binding format](problem-loading.md) to turn
one local recording into labelled, traceable cases. It illustrates three steps:
inspect a source once, plan from the saved metadata, then explicitly load cases.
It does not download data or fit a model.

The example uses the **Cho2017** left/right hand imagery dataset, distributed as
NEMAR **nm000245 v1.0.2**. Cho et al. provide the original study and data:
[paper](https://doi.org/10.1093/gigascience/gix034),
[source data](https://doi.org/10.5524/100295).
The [NEMAR derivative](https://github.com/nemarDatasets/nm000245/tree/v1.0.2)
declares CC BY 4.0 and conversion by MOABB 1.5.0. No participant data is bundled
with this example. Use a source checkout with aeon-neuro installed.

## Read the authored choices first

`examples/problem_definitions/cho2017.problem.json` states the question:

```json
{
  "format_version": 1,
  "id": "cho2017-central-imagery",
  "version": "1",
  "question": "Classify left versus right hand imagery from C3, Cz and C4 after the cue.",
  "class_labels": ["left_hand", "right_hand"],
  "channels": ["C3", "Cz", "C4"],
  "segmentation": {"mode": "event_epochs", "tmin": 0.5, "tmax": 2.5}
}
```

Each case is the half-open interval `[0.5, 2.5)` seconds after a published cue,
using C3, Cz and C4 in that order. At 512 Hz, each case has 1,024 samples.
These are explicit example choices; the recording itself does not select them.

`cho2017.binding.json` maps the physical dataset columns to those choices:
`trial_type` supplies the target, `onset` supplies cue seconds, and explicit
table-row IDs identify cases and trials. Participant/session identities come
from the inspected recording descriptor. The script prepares this descriptor
and the event rows; the reusable loader interprets the two authored documents.
Changing the epoch or channel list belongs in the problem JSON. Mapping another
dataset's column names or codes belongs in a new binding.

## 1. Inspect the local source

Separately obtain subject 1's BDF and matching sidecars from the
[pinned release](https://www.nemar.org/dataset/nm000245?v=v1.0.2), preserving:

```text
dataset_description.json
sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_eeg.bdf
sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_eeg.json
sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_channels.tsv
sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_events.tsv
```

Also save the small [release manifest](https://data.nemar.org/nm000245/v1.0.2/manifest.json)
locally, for example as `/data/cho2017-manifest.json`. The script trusts that
supplied manifest as the expected release identity. It verifies the small metadata
files, checks the BDF size and reads its header with `preload=False` to learn the
exact sample count, rate and channel order. It does not decode or hash the full
signal during this inspection. Header inspection requires the local BDF; this
step is not a remote metadata discovery implementation.

```bash
python examples/cho2017_problem.py inspect \
  --dataset /data/nm000245 --manifest /data/cho2017-manifest.json \
  --subject 1 --output /tmp/cho2017-catalogue.json
```

The catalogue contains source descriptors and original event columns, plus
stable IDs for rows of the supplied event table. Those IDs are not recovered
original acquisition-trial IDs. The manifest's expected BDF checksum is retained
without being treated as already verified.

## 2. Plan using only saved metadata

```bash
python examples/cho2017_problem.py plan \
  --catalogue /tmp/cho2017-catalogue.json --output /tmp/cho2017-plan.json
```

This command takes no dataset path and works while the original recording is
unavailable. It reads the catalogue and the example's authored JSON documents,
then calls `plan_classification_problem`. Use `--problem` and `--binding` to
choose edited documents. The resulting JSON exposes labels, case IDs, channel
choices, exact sample bounds and the effective problem/binding configuration.
Inspect it before loading signals.

For subject 1, the unfiltered source event table produces 200 cases, with a dense
shape of `(200, 3, 1024)` and 4,915,200 bytes of float64 case arrays. This estimate
does not describe the complete BDF size or peak process memory.

## 3. Explicitly load the planned cases

```bash
python examples/cho2017_problem.py materialize \
  --dataset /data/nm000245 --plan /tmp/cho2017-plan.json \
  --output /tmp/cho2017-cases
```

The source opener checks the BDF's complete SHA256 against the expected manifest
entry, opens it lazily and lets `materialize_classification_problem` read the
planned intervals and close the handle. Full-file hashing reads all source bytes;
selected-interval extraction does not remove that verification cost. Repeat
`--case-id` with IDs copied from the plan to extract a subset. All selected cases
here come from one recording, so subsetting cases still verifies that whole BDF.

The output directory contains raw `X.npy`, `y.npy`, `cases.json` and
`metadata.json`. Arrays use `(case, channel, time)` order and MNE-decoded volts;
source files are unchanged. Every command requires a fresh output path. These
outputs are case-construction artifacts, not a completed experiment or resumable
training run.

## Eligibility and evaluation remain explicit

This first walkthrough includes **all published events**. The exported event
tables contain no original MATLAB bad-trial annotations, and the script does
not fetch or interpret those source arrays. It records that eligibility policy
in the catalogue and output metadata. A case being included does not mean that
its EEG is artifact-free. Published cohort exclusions are not applied either.

The derivative stacks classes and already contains class-wise source demeaning;
its sample axis is not original interleaved acquisition chronology. No filtering,
rereferencing, split generation, CSP fitting or accuracy claim follows from this
example. A quality-aware benchmark needs a reviewed source-quality join and
explicit cohort, preprocessing and evaluation choices. Use
[assignment validation](classification-problems.md) when adding those stages,
including participant identities and full preprocessing support as appropriate.

Generated BDF tests exercise this walkthrough in CI without downloading or
redistributing participant recordings:

```bash
python -m pytest -q examples/tests/test_cho2017_problem.py
```
