"""Resolve versioned author recipes and construct reproducible group splits."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Recipe numbers must be finite.")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    if isinstance(value, list):
        for item in value:
            _finite_json(item)


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
    if isinstance(recipe, (str, Path)):
        with open(recipe, encoding="utf-8") as stream:
            recipe = json.load(stream)
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


def _read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not fields:
            raise ValueError(f"Missing or duplicate TSV columns: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed TSV rows: {path}")
    return rows, fields


def _validate_partitions(cases, groups, splits):
    assignments = {}
    for name, indices in splits.items():
        for index in indices:
            if int(index) in assignments:
                raise ValueError("A case occurs in more than one partition.")
            assignments[int(index)] = name
    if len(assignments) != len(cases):
        raise ValueError("Split assignments must cover all included cases.")
    group_parts = {}
    for index, group in enumerate(groups):
        group_parts.setdefault(group, set()).add(assignments[index])
    if any(len(value) != 1 for value in group_parts.values()):
        raise ValueError("A group crosses split partitions.")
    import heapq

    records = {}
    for index, row in enumerate(cases):
        records.setdefault(row["source_identity"], []).append(
            (row["start_sample"], row["stop_sample"], assignments[index])
        )
    for intervals in records.values():
        active, counts = [], {}
        for start, stop, part in sorted(intervals):
            while active and active[0][0] <= start:
                _, expired = heapq.heappop(active)
                counts[expired] -= 1
            if any(count and name != part for name, count in counts.items()):
                raise ValueError("Source sample intervals overlap across partitions.")
            heapq.heappush(active, (stop, part))
            counts[part] = counts.get(part, 0) + 1
    classes = {row["label"] for row in cases}
    if len(classes) < 2:
        raise ValueError("Classification requires at least two retained classes.")
    for name, indices in splits.items():
        if not len(indices) or {cases[i]["label"] for i in indices} != classes:
            raise ValueError(
                f"Split {name!r} does not retain all classes. "
                "Change the declared protocol; no seed retries are performed."
            )


def _split_cases(cases, recipe, recipe_dir):
    evaluation = recipe["evaluation"]
    split = evaluation["split"]
    groups = [row["group_id"] for row in cases]
    unique = sorted(set(groups), key=lambda value: value.encode("utf-8"))
    if split["method"] == "supplied_manifest":
        if recipe_dir is None:
            raise ValueError(
                "Manifest replay requires a recipe filename to bind relative paths."
            )
        path = (recipe_dir / split["manifest"]).resolve()
        if not path.is_relative_to(recipe_dir.resolve()):
            raise ValueError("Split manifest escapes the recipe directory.")
        rows, fields = _read_tsv(path)
        keys = split["join_keys"]
        if set(keys + [split["split_column"]]) - set(fields):
            raise ValueError("Split manifest is missing join or split columns.")
        lookup = {}
        for row in rows:
            key = tuple(row[k] for k in keys)
            if key in lookup:
                raise ValueError("Duplicate split-manifest join key.")
            lookup[key] = row[split["split_column"]]
        parts, used = [], set()
        for row in cases:
            key = tuple(str(row[k]) for k in keys)
            if key not in lookup:
                raise ValueError("Split manifest does not cover an included case.")
            parts.append(lookup[key])
            used.add(key)
        if set(lookup) != used or set(parts) - {"train", "validation", "test"}:
            raise ValueError(
                "Split manifest has unused identities or unknown partitions."
            )
        if not {"train", "test"}.issubset(parts):
            raise ValueError("Split manifest needs train and test partitions.")
        splits = {
            name: np.flatnonzero(np.asarray(parts) == name)
            for name in ("train", "validation", "test")
            if name in parts
        }
    else:
        fractions = split["fractions"]
        names = [
            name for name in ("train", "validation", "test") if fractions[name] > 0
        ]
        if len(unique) < len(names):
            raise ValueError(
                "Insufficient independent groups for the requested split; "
                "explicitly choose an appropriate grouping, "
                "never a per-window fallback."
            )
        if split["strategy"] == "chronological":
            order = {}
            for row in cases:
                value = tuple(row[k] for k in split["order_by"])
                if row["group_id"] in order and order[row["group_id"]] != value:
                    raise ValueError(
                        "Chronological ordering must be constant within a group."
                    )
                order[row["group_id"]] = value
            unique.sort(key=lambda key: (order[key], key))
            n_test = int(np.ceil(len(unique) * fractions["test"]))
            n_val = int(np.ceil(len(unique) * fractions["validation"]))
            n_train = len(unique) - n_test - n_val
            if n_train < 1:
                raise ValueError(
                    "Insufficient groups for chronological split fractions."
                )
            by_part = {"train": unique[:n_train], "test": unique[n_train + n_val :]}
            if n_val:
                by_part["validation"] = unique[n_train : n_train + n_val]
        else:
            group_labels = None
            if split["stratify_by"] == "target":
                labels = {group: set() for group in unique}
                for row in cases:
                    labels[row["group_id"]].add(row["label"])
                if any(len(values) != 1 for values in labels.values()):
                    raise ValueError(
                        "Target stratification requires a single label per group."
                    )
                group_labels = {
                    group: next(iter(values)) for group, values in labels.items()
                }

            def shuffle(values, fraction):
                stratify = (
                    None if group_labels is None else [group_labels[g] for g in values]
                )
                return train_test_split(
                    values,
                    test_size=fraction,
                    random_state=split["random_seed"],
                    stratify=stratify,
                )

            train, test = shuffle(unique, fractions["test"])
            by_part = {"train": train, "test": test}
            if fractions["validation"]:
                train, validation = shuffle(
                    train, fractions["validation"] / (1 - fractions["test"])
                )
                by_part.update(train=train, validation=validation)
        membership = {
            group: name for name, values in by_part.items() for group in values
        }
        splits = {
            name: np.asarray(
                [i for i, group in enumerate(groups) if membership[group] == name],
                dtype=int,
            )
            for name in names
        }
    _validate_partitions(cases, groups, splits)
    for name, indices in splits.items():
        for index in indices:
            cases[index]["split"] = name
    return splits
