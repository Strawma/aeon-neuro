# Executable EEG experiment recipes

`aeon_neuro.experiments` is an experimental interface for sharing experiment
settings between scripts and the EEG Console. It describes preparation,
case and label rules, models and evaluation.

```python
from aeon_neuro.experiments import load_experiment_recipe, run_experiment

recipe = load_experiment_recipe("eeg-console-setup.json")
dataset, results = run_experiment(recipe, recordings, metadata, include_evidence=True)
```

To prepare cases without fitting, call `prepare_experiment` with the same
inputs. To author a recipe in Python, pass a configuration to
`make_experiment_recipe` and write the returned object as JSON.

## File format

The standalone JSON document has `format: "aeon-neuro-experiment"`,
`format_version: 1`, `input_kind` of `raw` or `prepared`, and a
`configuration` object. Fields in that object include `labels`, `rules`,
`preprocessing`, `classifiers`, `evaluation`, `channel_selection`, and
`shuffle_runs`. The optional `describe` object carries source-adapter settings,
such as which BIDS event column supplies the codes.

The loader also accepts existing version-2 Console setup files. These files
include source and preview verification records owned by the Console. A
newer export embeds the shared recipe and includes a compatibility
configuration. The loader checks that their executable settings agree.
Case, fold and source verification can be handled by the application.

## Bind the EEG inputs

An input adapter supplies an ordered sequence of raw (channels, samples)
arrays in volts. Give one metadata mapping per recording with its
`recording_id`, `subject`, `sampling_frequency`, and `channel_names`.
Pass `events` as onset-in-seconds and code pairs for event recipes, case-label
`fields` for recording-labelled recipes, and `bad_spans` as start/stop pairs
in seconds for BAD annotation screening. A replay uses the same source
recording IDs, order, and channel names. The case and fold evidence uses
those IDs to identify what was fitted and tested.

For `prepared` recipes, give the runner a `ClassificationDataset` in place
of the raw recordings and metadata. It uses the supplied cases and
applies any fixed channel exclusions. The evaluator fits optional learned
channel selection separately in each training fold. Recipes that explicitly
include older all-case selection preserve that legacy behaviour.

The numerical choices in this interface come from the Console executor.
For example, legacy stride downsampling and the polyphase route have distinct
semantics, and the filter runs on the clock set by the preceding operations.
Keep the scientific dependency versions with a run when exact replay matters.
