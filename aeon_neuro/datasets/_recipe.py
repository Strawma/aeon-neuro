"""Validate author recipes and resolve versioned omission defaults."""

import json
from copy import deepcopy
from os import PathLike
from pathlib import Path

import numpy as np


def _finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Recipe numbers must be finite.")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    if isinstance(value, list):
        for item in value:
            _finite_json(item)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate recipe JSON key: {key!r}.")
        result[key] = value
    return result


def resolve_classification_recipe(recipe, *, random_state=None):
    """Validate an author recipe and resolve the versioned omission defaults.

    Parameters
    ----------
    recipe : dict or path-like
        Recipe dictionary or UTF-8 JSON file, version ``0.3.0-draft``.
        Requires the optional ``jsonschema`` dependency (the ``recipes`` extra).
    random_state : int or None, default=None
        Optional runtime override for a generated shuffled split's seed.
        None retains the recipe seed, or its default of 42.

    Returns
    -------
    resolved : dict
        Independent effective recipe. Original inputs are never modified.
    origins : dict
        Origins of supplied settings, defaults and runtime overrides.

    Notes
    -----
    JSON Schema validates structure; this function explicitly supplies defaults.
    BIDS-derived values are resolved later by ``load_bids_classification``.
    No signal processing, case extraction or splitting is performed here.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise ModuleNotFoundError(
            "Classification recipes require jsonschema; install aeon-neuro[recipes]."
        ) from error
    if isinstance(recipe, (str, PathLike)):
        with open(recipe, encoding="utf-8") as stream:
            recipe = json.load(stream, object_pairs_hook=_unique_keys)
    if not isinstance(recipe, dict):
        raise TypeError("recipe must be a dictionary or JSON filename.")
    _finite_json(recipe)
    schema = json.loads(
        Path(__file__).with_name("classification_recipe.schema.json").read_text()
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(recipe),
        key=lambda error: str(list(error.absolute_path)),
    )
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "recipe"
        raise ValueError(f"Invalid {location}: {error.message}")
    result = deepcopy(recipe)
    origins = {}

    def supplied(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                supplied(item, f"{path}.{key}".strip("."))
        else:
            origins[path] = "supplied"

    def default(target, key, value, path, origin="default"):
        if key not in target:
            target[key] = deepcopy(value)
            origins[f"{path}.{key}".strip(".")] = origin
        return target[key]

    supplied(result)
    default(result, "selection", {}, "")
    for key in ("subjects", "sessions", "runs"):
        default(result["selection"], key, None, "selection")
    target = result["target"]
    default(target, "exclude_values", [], "target")
    default(target, "missing_value", "error", "target")
    default(result, "channels", {"types": ["eeg"]}, "")
    if "names" not in result["channels"]:
        default(result["channels"], "exclude_names", [], "channels")
    segment = result["segmentation"]
    if segment["mode"] == "fixed_windows":
        default(
            segment,
            "stride_seconds",
            segment["window_seconds"],
            "segmentation",
            "default_from:segmentation.window_seconds",
        )
        interval = default(segment, "interval", {"kind": "time_range"}, "segmentation")
        for key, value in {
            "start_seconds": 0,
            "stop_seconds": None,
            "boundary_values": [],
            "boundary_action": "error",
            "short_recording_action": "error",
        }.items():
            default(interval, key, value, "segmentation.interval")
        if interval["stop_seconds"] is not None and (
            interval["stop_seconds"] <= interval["start_seconds"]
        ):
            raise ValueError("interval.stop_seconds must exceed start_seconds.")
        if interval["kind"] == "first_state_interval":
            default(interval, "state_column", "trial_type", "segmentation.interval")
            default(
                interval, "unresolved_state_action", "error", "segmentation.interval"
            )
            if interval["initial_state"] in interval["end_states"]:
                raise ValueError("Initial and end states must be distinct.")
    else:
        default(segment, "start_seconds", 0, "segmentation")
        default(segment, "onset_column", "onset", "segmentation")
        if segment["stop_seconds"] <= segment["start_seconds"]:
            raise ValueError("Epoch stop_seconds must exceed start_seconds.")
    evaluation = default(result, "evaluation", {}, "")
    default(evaluation, "group_by", ["bids_subject"], "evaluation")
    definitions = default(evaluation, "group_definitions", {}, "evaluation")
    for name, definition in definitions.items():
        default(definition, "origin", 1, f"evaluation.group_definitions.{name}")
    known = {"bids_subject", "recording", "session", "run", "task"}
    if set(definitions) & known:
        raise ValueError("Custom group definitions cannot shadow recording identities.")
    if set(evaluation["group_by"]) - (known | set(definitions)):
        raise ValueError("group_by contains an undefined identity or custom group.")
    if definitions and segment["mode"] != "event_epochs":
        raise ValueError("Event-column group definitions require event_epochs.")
    split = default(evaluation, "split", {}, "evaluation")
    default(split, "method", "generated", "evaluation.split")
    if split["method"] == "supplied_manifest":
        default(split, "split_column", "split", "evaluation.split")
        if random_state is not None:
            raise ValueError(
                "A supplied split manifest cannot take a random_state override."
            )
    else:
        default(split, "strategy", "shuffle", "evaluation.split")
        default(
            split,
            "fractions",
            {"train": 0.8, "validation": 0, "test": 0.2},
            "evaluation.split",
        )
        if not np.isclose(sum(split["fractions"].values()), 1, rtol=0, atol=1e-12):
            raise ValueError("Split fractions must sum to one.")
        seed = None if split["strategy"] == "chronological" else 42
        default(split, "random_seed", seed, "evaluation.split")
        default(split, "stratify_by", None, "evaluation.split")
        if random_state is not None:
            if split["strategy"] != "shuffle":
                raise ValueError("Chronological splitting cannot take a random seed.")
            if (
                isinstance(random_state, bool)
                or not isinstance(random_state, int)
                or not 0 <= random_state < 2**32
            ):
                raise ValueError("random_state must be an integer in [0, 2**32).")
            split["random_seed"] = random_state
            origins["evaluation.split.random_seed"] = "runtime_override"
    return result, origins
