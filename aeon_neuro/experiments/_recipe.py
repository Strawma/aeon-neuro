"""Encode experiment settings and execute them on a supplied EEG collection."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aeon_neuro.datasets import ClassificationDataset
from aeon_neuro.experiments._evaluation import evaluate
from aeon_neuro.experiments._preparation import (
    continuous_order,
    normalise,
    prepare_recording_cases,
    recording_plan,
    select_prepared_channels,
    transform_recording,
    transformed_metadata,
)

EXPERIMENT_FORMAT = "aeon-neuro-experiment"
_VERSION = 1
_FIELDS = {
    "describe": dict,
    "labels": dict,
    "rules": dict,
    "preprocessing": dict,
    "classifiers": list,
    "evaluation": dict,
    "channel_selection": dict,
    "shuffle_runs": int,
}


class ExperimentRecipeError(ValueError):
    """A malformed or unsupported experiment recipe."""


def make_experiment_recipe(configuration, *, input_kind="raw"):
    """Create a detached JSON-compatible recipe from experiment settings.

    Parameters
    ----------
    configuration : dict
        Execution settings for preparation, labels, case rules, classifiers,
        channel selection, and evaluation. Source-adapter options can be stored
        in ``describe``. Application-only fields are left out.
    input_kind : {"raw", "prepared"}, default="raw"
        Whether the runner consumes recordings or a ClassificationDataset.

    Returns
    -------
    dict
        The versioned recipe document.
    """
    if input_kind not in ("raw", "prepared"):
        raise ExperimentRecipeError("Experiment input_kind must be raw or prepared.")
    if not isinstance(configuration, Mapping):
        raise ExperimentRecipeError("Expected an experiment configuration object.")
    payload = {key: configuration[key] for key in _FIELDS if key in configuration}
    for key, value in payload.items():
        if type(value) is not _FIELDS[key]:
            raise ExperimentRecipeError(f"Invalid experiment field {key!r}.")
    if input_kind == "raw":
        try:
            continuous_order(payload.get("preprocessing", {}))
        except ValueError as error:
            raise ExperimentRecipeError(str(error)) from error
    try:
        payload = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ExperimentRecipeError(
            "Experiment settings must be portable JSON values."
        ) from error
    return {
        "format": EXPERIMENT_FORMAT,
        "format_version": _VERSION,
        "input_kind": input_kind,
        "configuration": payload,
    }


def load_experiment_recipe(source):
    """Load a recipe from a mapping or JSON path.

    Version-2 EEG Console setup envelopes are also supported. They can contain
    an embedded recipe and a compatibility configuration. Those settings must
    agree. The envelope's source and preview receipts remain application data.
    """
    if isinstance(source, Mapping):
        document = source
    else:
        try:
            document = json.loads(Path(source).read_text(encoding="utf-8"))
        except (TypeError, OSError, ValueError) as error:
            raise ExperimentRecipeError(
                f"Could not read the experiment recipe: {error}"
            ) from error
    if not isinstance(document, Mapping):
        raise ExperimentRecipeError("Expected an experiment recipe object.")
    if (
        document.get("format") == "eeg-console-setup"
        and document.get("format_version") == 2
    ):
        source_info = document.get("source")
        if not isinstance(source_info, Mapping):
            raise ExperimentRecipeError("The Console setup is missing its input kind.")
        legacy = make_experiment_recipe(
            document.get("configuration"), input_kind=source_info.get("kind")
        )
        if "experiment" not in document:
            return legacy
        recipe = _load_recipe(document["experiment"])
        if recipe != legacy:
            raise ExperimentRecipeError(
                "The shared recipe differs from the Console setup settings."
            )
        return recipe
    return _load_recipe(document)


def _load_recipe(document):
    if (
        not isinstance(document, Mapping)
        or document.get("format") != EXPERIMENT_FORMAT
        or type(document.get("format_version")) is not int
        or document["format_version"] != _VERSION
    ):
        raise ExperimentRecipeError(
            "Unsupported EEG experiment recipe format or version."
        )
    payload = document.get("configuration")
    if not isinstance(payload, Mapping) or set(payload).difference(_FIELDS):
        raise ExperimentRecipeError("Invalid fields in the experiment configuration.")
    return make_experiment_recipe(payload, input_kind=document.get("input_kind"))


def prepare_experiment(recipe, signals, metadata=None):
    """Build the dataset for an experiment on a supplied recording collection.

    For raw input, supply a sequence of (channels, samples) arrays in volts
    and one metadata mapping per recording. Metadata includes recording_id,
    subject, sampling_frequency, channel_names, and any events, fields or
    bad_spans required by the recipe. Use the original recording order and
    IDs when comparing to a saved setup. Readers and file layout belong to
    the input adapter.

    For prepared input, supply a ClassificationDataset with aligned cases,
    labels, and groups. Only fixed channel selection is applied in this mode.
    """
    document = load_experiment_recipe(recipe)
    config = document["configuration"]
    if document["input_kind"] == "prepared":
        if not isinstance(signals, ClassificationDataset) or metadata is not None:
            raise ExperimentRecipeError(
                "Prepared input must be a ClassificationDataset."
            )
        return select_prepared_channels(
            signals, config.get("preprocessing", {}).get("excluded_channels")
        )

    if metadata is None or len(metadata) == 0 or len(signals) != len(metadata):
        raise ExperimentRecipeError("Supply one metadata mapping per raw recording.")
    prep, labels, rules = normalise(config)
    used, target, _, _ = recording_plan(metadata, prep)
    arrays, metas = [], []
    for signal, meta in zip(signals, metadata, strict=True):
        # Match the recording cache boundary used for portable recipes.
        data = np.asarray(signal, dtype=np.float32)
        transformed, rate = transform_recording(data, meta, prep, used, target)
        arrays.append(transformed)
        metas.append(transformed_metadata(meta, used, rate))
    dataset, _ = prepare_recording_cases(arrays, metas, prep, labels, rules)
    return dataset


def run_experiment(recipe, signals, metadata=None, *, include_evidence=False):
    """Prepare and evaluate an experiment, returning its dataset and results."""
    document = load_experiment_recipe(recipe)
    dataset = prepare_experiment(document, signals, metadata)
    result = evaluate(
        dataset, document["configuration"], include_evidence=include_evidence
    )
    return dataset, result
