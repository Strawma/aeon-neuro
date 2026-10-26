"""Load classification recipes from BIDS using bounded native-signal reads."""

import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from pathlib import Path

import numpy as np
from mne._fiff.constants import FIFF
from mne_bids import find_matching_paths, read_raw_bids

from aeon_neuro.datasets._classification import (
    ClassificationDataset,
    build_classification_dataset,
)
from aeon_neuro.datasets._recipe import (
    _read_tsv,
    _split_cases,
    resolve_classification_recipe,
)
from aeon_neuro.utils.data_io import _check_edf_layout


@dataclass
class ClassificationRecipeResult:
    """Recipe dataset, partition indices and reproducibility records.

    Parameters
    ----------
    dataset : ClassificationDataset
        Cases, labels, group IDs and original source-sample provenance.
        X is memory-mapped when an output directory was supplied.
    splits : dict of str to np.ndarray
        Train/test and optional validation case indices. No estimator is fitted.
    recipe : dict
        Effective recipe after defaults and data-dependent selection.
    report : dict
        Completed run record including setting origins, source hashes,
        versions, exclusions and per-partition class/group counts.
    """

    dataset: ClassificationDataset
    splits: dict
    recipe: dict
    report: dict


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path, suffix, extension):
    result = path.find_matching_sidecar(
        suffix=suffix, extension=extension, on_error="ignore"
    )
    return None if result is None else Path(result)


def _open_raw(path):
    extension = path.extension.lower()
    if extension in (".edf", ".bdf"):
        _check_edf_layout(path.fpath, 2 if extension == ".edf" else 3)
    return read_raw_bids(path, extra_params={"preload": False}, verbose=False)


def _number(value, context, *, integer=False):
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context} must be numeric, not {value!r}.") from error
    if not np.isfinite(result) or (integer and not result.is_integer()):
        raise ValueError(
            f"{context} must be finite{' and integral' if integer else ''}."
        )
    return int(result) if integer else result


def _sample(seconds, sfreq):
    return int(np.floor(seconds * sfreq + 0.5))


def _event_samples(rows, fields, sfreq, segmentation):
    onset = segmentation.get("onset_column", "onset")
    if rows and onset not in fields:
        raise ValueError(f"Events are missing onset column {onset!r}.")
    requested = segmentation.get("sample_column")
    if requested is not None and requested not in fields:
        raise ValueError(f"Events are missing requested sample column {requested!r}.")
    columns = [
        name
        for name in ("sample", "onset_sample", requested)
        if name is not None and name in fields
    ]
    columns = list(dict.fromkeys(columns))
    result = []
    for index, row in enumerate(rows, start=1):
        value = _number(row[onset], f"event {index} onset") * sfreq
        candidates = [
            _number(row[name], f"event {index} {name}", integer=True)
            for name in columns
        ]
        if (
            any(abs(sample - value) > 0.5 + 1e-8 for sample in candidates)
            or len(set(candidates)) > 1
        ):
            raise ValueError(f"Event {index}: onset/sample columns disagree.")
        result.append(candidates[0] if candidates else _sample(value / sfreq, sfreq))
    return result


def _channel_names(raw, selector, path):
    names = list(raw.ch_names)
    if "names" in selector:
        selected = selector["names"]
    elif "include_prefixes" in selector:
        prefixes = selector["include_prefixes"]
        if any(
            not any(name.startswith(prefix) for name in names) for prefix in prefixes
        ):
            raise ValueError(
                "Every requested channel prefix must match a source channel."
            )
        selected = [
            name
            for name in names
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
    else:
        channels = _sidecar(path, "channels", ".tsv")
        if channels is None:
            raise ValueError(
                "Type-based channel selection needs a BIDS channels.tsv; "
                "explicitly select names/prefixes rather than reader-inferred roles."
            )
        rows, fields = _read_tsv(channels)
        if not {"name", "type"}.issubset(fields):
            raise ValueError("channels.tsv needs name and type columns.")
        types = {row["name"]: row["type"].lower() for row in rows}
        if len(types) != len(rows) or set(names) - types.keys():
            raise ValueError("BIDS channel identities are missing or ambiguous.")
        wanted = {name.lower() for name in selector["types"]}
        selected = [name for name in names if types[name] in wanted]
    selected = [
        name for name in selected if name not in selector.get("exclude_names", [])
    ]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(names)
    ):
        raise ValueError("Selected channel names must be present, nonempty and unique.")
    return list(selected)


def _label(row, target, context):
    column = target["column"]
    if column not in row:
        raise ValueError(f"{context}: missing target column {column!r}.")
    value = row[column]
    if value in target["exclude_values"]:
        return None, "excluded_target_value"
    if value in ("", "n/a"):
        if target["missing_value"] == "drop_and_report":
            return None, "missing_target_value"
        raise ValueError(f"{context}: missing target value.")
    if "mapping" in target:
        if value not in target["mapping"]:
            raise ValueError(f"{context}: unmapped target {value!r}.")
        value = target["mapping"][value]
    return str(value), ""


def _group(row, event, evaluation):
    for name, definition in evaluation["group_definitions"].items():
        column = definition["column"]
        if column not in event:
            raise ValueError(f"Missing group-definition column {column!r}.")
        value = _number(event[column], f"group column {column}", integer=True)
        if value < definition["origin"]:
            raise ValueError("Group value is below the declared integer-bin origin.")
        row[name] = (value - definition["origin"]) // definition["bin_size"] + 1
    values = []
    for key in evaluation["group_by"]:
        if row.get(key) is None:
            raise ValueError(f"Missing split group identity {key!r}.")
        values.append(row[key])
    row["group_id"] = json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _interval(raw, rows, fields, samples, interval):
    column = interval.get("state_column", "trial_type")
    if interval["boundary_values"]:
        if column not in fields:
            raise ValueError(
                "Declared boundary events require the selected state column."
            )
        if any(row[column] in interval["boundary_values"] for row in rows):
            if interval["boundary_action"] == "exclude_recording":
                return None, "declared_boundary_event"
            raise ValueError("Recording contains a declared boundary event.")
    sfreq = raw.info["sfreq"]
    first = _sample(interval["start_seconds"], sfreq)
    last = (
        raw.n_times
        if interval["stop_seconds"] is None
        else _sample(interval["stop_seconds"], sfreq)
    )
    truncated = last > raw.n_times
    if truncated and interval["short_recording_action"] == "error":
        raise ValueError(
            "Requested interval exceeds the recording; explicitly allow truncation."
        )
    last = min(last, raw.n_times)
    if interval["kind"] == "first_state_interval":
        initial = interval["initial_state"]
        states = {initial, *interval["end_states"]}
        transitions = sorted(
            (sample, row.get(column))
            for row, sample in zip(rows, samples)
            if row.get(column) in states
        )
        conflict = any(
            left[0] == right[0] and left[1] != right[1]
            for left, right in zip(transitions, transitions[1:])
        )
        if not transitions or transitions[0][1] != initial or conflict:
            if interval["unresolved_state_action"] == "exclude_recording":
                return None, "unresolved_initial_state"
            raise ValueError("Initial state is missing, ambiguous or inconsistent.")
        first = max(first, transitions[0][0])
        ends = [
            sample for sample, value in transitions if value in interval["end_states"]
        ]
        if ends:
            last = min(last, ends[0])
    return (first, last), "truncated_recording" if truncated else ""


def _discover(root, recipe):
    selection = recipe["selection"]
    paths = find_matching_paths(
        root,
        **{key: value for key, value in selection.items() if key != "tasks"},
        tasks=selection.get("tasks"),
        datatypes=["eeg"],
        extensions=[".vhdr", ".edf", ".bdf", ".set"],
        check=False,
    )
    paths = sorted(
        paths, key=lambda path: str(path.fpath.relative_to(root)).encode("utf-8")
    )
    if not paths:
        raise ValueError("No matching BIDS EEG recordings.")
    if "tasks" not in selection:
        tasks = {path.task for path in paths}
        if len(tasks) != 1:
            raise ValueError(
                "Multiple tasks found; explicitly select tasks (or null for all)."
            )
        selection["tasks"] = [next(iter(tasks))]
    return paths


def _plan(root, recipe):
    paths = _discover(root, recipe)
    participants, files = {}, set()
    participant_table = root / "participants.tsv"
    if participant_table.exists():
        rows, fields = _read_tsv(participant_table)
        if "participant_id" not in fields:
            raise ValueError("participants.tsv is missing participant_id.")
        for row in rows:
            if row["participant_id"] in participants:
                raise ValueError("Duplicate participant identity in participants.tsv.")
            participants[row["participant_id"]] = row
        files.add(participant_table)
    for path in root.glob("*.json"):
        files.add(path)
    cases, exclusions, recordings, adjustments = [], [], [], []
    output_names, output_sfreq, output_size = None, None, None
    for path in paths:
        raw = _open_raw(path)
        try:
            recording = path.fpath.relative_to(root).as_posix()
            subject = None if path.subject is None else f"sub-{path.subject}"
            info = {
                "recording_id": recording,
                "filepath": recording,
                "source_identity": str(path.fpath.resolve()),
                "bids_subject": subject,
                "session": path.session,
                "run": path.run,
                "task": path.task,
                "n_samples": int(raw.n_times),
                "sampling_frequency": float(raw.info["sfreq"]),
                "channel_names": _channel_names(raw, recipe["channels"], path),
                "bad_channels": list(raw.info["bads"]),
            }
            units = {channel["ch_name"]: channel["unit"] for channel in raw.info["chs"]}
            if any(units[name] != FIFF.FIFF_UNIT_V for name in info["channel_names"]):
                raise ValueError(
                    "Selected classifier channels must have decoded voltage units."
                )
            files.add(Path(path.fpath))
            files.update(Path(value) for value in raw.filenames if value is not None)
            if path.extension == ".vhdr":
                files.update(path.fpath.parent.glob(path.fpath.stem + ".vmrk"))
            for parent in (path.fpath.parent, *path.fpath.parent.parents):
                if not parent.is_relative_to(root):
                    break
                files.update(parent.glob("*.json"))
            events_path = _sidecar(path, "events", ".tsv")
            rows, fields = ([], []) if events_path is None else _read_tsv(events_path)
            if events_path is not None:
                files.add(events_path)
            channels_path = _sidecar(path, "channels", ".tsv")
            if channels_path is not None:
                files.add(channels_path)
            segment, target = recipe["segmentation"], recipe["target"]
            dictionary_path = (
                root / "participants.json"
                if target["source"] == "participants.tsv"
                else _sidecar(path, "events", ".json")
            )
            if dictionary_path is not None and dictionary_path.is_file():
                dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
                info["target_dictionary"] = dictionary.get(target["column"], {})
                files.add(dictionary_path)
            info["reader_preloaded_source"] = bool(raw.preload)
            samples = _event_samples(rows, fields, info["sampling_frequency"], segment)
            recording_label, reason = None, ""
            if target["source"] == "participants.tsv":
                if subject not in participants:
                    raise ValueError(f"Missing participant join for {subject!r}.")
                recording_label, reason = _label(
                    participants[subject], target, recording
                )
                if reason:
                    exclusions.append(
                        {"recording": recording, "exclusion_reason": reason}
                    )
                    continue
            candidates = []
            if segment["mode"] == "fixed_windows":
                bounds, decision = _interval(
                    raw, rows, fields, samples, segment["interval"]
                )
                if bounds is None:
                    exclusions.append(
                        {"recording": recording, "exclusion_reason": decision}
                    )
                    continue
                if decision:
                    adjustments.append({"recording": recording, "decision": decision})
                first, last = bounds
                size = _sample(segment["window_seconds"], info["sampling_frequency"])
                step = _sample(segment["stride_seconds"], info["sampling_frequency"])
                if size < 1 or step < 1:
                    raise ValueError("Window and stride must span at least one sample.")
                for start in range(first, max(first, last), step):
                    candidates.append(
                        (
                            {
                                "start_sample": start,
                                "stop_sample": start + size,
                                "label": recording_label,
                                "included": start + size <= last,
                            },
                            {},
                        )
                    )
            else:
                if (
                    events_path is None
                    or target["source"] == "events.tsv"
                    and target["column"] not in fields
                ):
                    raise ValueError("Missing event table or target column.")
                start = _sample(segment["start_seconds"], info["sampling_frequency"])
                stop = _sample(segment["stop_seconds"], info["sampling_frequency"])
                if stop <= start:
                    raise ValueError("Epoch offsets must span at least one sample.")
                ids = set()
                for index, (event, onset) in enumerate(zip(rows, samples), start=1):
                    label, reason = (
                        (recording_label, "")
                        if target["source"] == "participants.tsv"
                        else _label(event, target, f"{recording} event {index}")
                    )
                    id_column = segment.get("event_id_column")
                    event_id = str(index) if id_column is None else event.get(id_column)
                    if event_id in (None, "", "n/a") or event_id in ids:
                        raise ValueError(
                            "Explicit event identifiers must be present and "
                            "unique within a recording."
                        )
                    ids.add(event_id)
                    candidates.append(
                        (
                            {
                                "start_sample": onset + start,
                                "stop_sample": onset + stop,
                                "label": label,
                                "event_id": event_id,
                                "source_event_row": index,
                                "included": not reason,
                                "exclusion_reason": reason,
                            },
                            event,
                        )
                    )
            for index, (candidate, event) in enumerate(candidates):
                row = {
                    key: info[key]
                    for key in (
                        "recording_id",
                        "source_identity",
                        "bids_subject",
                        "session",
                        "run",
                        "task",
                    )
                }
                row.update(recording=recording, **candidate)
                row["case_id"] = f"{recording}::candidate-{index}"
                _group(row, event, recipe["evaluation"])
                keep = (
                    row["included"]
                    and 0 <= row["start_sample"] < row["stop_sample"] <= raw.n_times
                )
                if not keep:
                    row["exclusion_reason"] = (
                        row.get("exclusion_reason") or "incomplete_window"
                    )
                    row["included"] = False
                    exclusions.append(row)
                    continue
                row.pop("exclusion_reason", None)
                size = row["stop_sample"] - row["start_sample"]
                if output_names is None:
                    output_names, output_sfreq, output_size = (
                        info["channel_names"],
                        info["sampling_frequency"],
                        size,
                    )
                if (
                    info["channel_names"] != output_names
                    or info["sampling_frequency"] != output_sfreq
                    or size != output_size
                ):
                    raise ValueError(
                        "Cases must have identical selected channels/order, rate "
                        "and length; explicitly align the input selection."
                    )
                cases.append(row)
            recordings.append((path, info))
        finally:
            raw.close()
    if not cases:
        raise ValueError(
            f"No included cases remain; recording/candidate exclusions: {exclusions!r}"
        )
    classes = sorted(
        {row["label"] for row in cases}, key=lambda value: value.encode("utf-8")
    )
    if "class_order" in recipe["target"] and set(
        recipe["target"]["class_order"]
    ) != set(classes):
        raise ValueError(
            "Explicit class_order must cover exactly the retained classes."
        )
    recipe["target"].setdefault("class_order", classes)
    return (
        recordings,
        cases,
        exclusions,
        adjustments,
        files,
        (len(cases), len(output_names), output_size),
    )


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_cases(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def load_bids_classification(
    bids_root, recipe, *, output_dir=None, random_state=None, max_memory_mb=512
):
    """Construct and split BIDS cases from a versioned classification recipe.

    Parameters
    ----------
    bids_root : path-like
        Existing BIDS root. Signals and source metadata are never modified.
    recipe : dict or path-like
        Author recipe, version ``0.3.0-draft``. Requires the ``recipes`` extra.
    output_dir : path-like or None, default=None
        New directory outside raw BIDS. Writes X.npy as a memory map plus case,
        exclusion, assignment, resolved-recipe and run-report files. Existing
        destinations are rejected. None returns an in-memory result only.
    random_state : int or None, default=None
        Optional override of the recipe's generated shuffle seed.
    max_memory_mb : float, default=512
        Maximum in-memory case-array allocation in MiB when output_dir is None.
        Larger problems require an output directory. Native reads always request
        individual intervals. Embedded EEGLAB readers may preload their source.

    Returns
    -------
    ClassificationRecipeResult
        Dataset, disjoint partition indices, effective recipe and provenance.

    Notes
    -----
    Reuses Pair 2's classification builder for bounded-window validation and the
    shared ClassificationDataset container. Its strict independent-recording
    split validator is not used for explicitly requested within-recording block
    splits; group and source-sample separation are checked here instead.
    No filtering, resampling, artifact rejection or model fitting is applied.
    Files retain failed status if extraction fails after output creation. Plan
    errors occur before writing. Retry in a fresh destination after inspection.
    """
    root = Path(bids_root).resolve()
    if not root.is_dir():
        raise ValueError("bids_root must be an existing directory.")
    destination = None if output_dir is None else Path(output_dir).resolve()
    if destination is not None and (
        destination.exists() or destination.is_relative_to(root)
    ):
        raise ValueError("output_dir must be new and outside raw BIDS.")
    if (
        not isinstance(max_memory_mb, Real)
        or isinstance(max_memory_mb, bool)
        or not np.isfinite(max_memory_mb)
        or max_memory_mb <= 0
    ):
        raise ValueError("max_memory_mb must be finite and positive.")
    recipe_dir = (
        Path(recipe).resolve().parent if isinstance(recipe, (str, Path)) else None
    )
    resolved, origins = resolve_classification_recipe(recipe, random_state=random_state)
    recordings, cases, exclusions, adjustments, sources, shape = _plan(root, resolved)
    origins.setdefault("selection.tasks", "data:single_available_task")
    origins.setdefault("target.class_order", "data:retained_label_vocabulary")
    splits = _split_cases(cases, resolved, recipe_dir)
    nbytes = int(np.prod(shape)) * 8
    if destination is None and nbytes > max_memory_mb * 1024**2:
        raise ValueError(
            f"Case array requires {nbytes} bytes; "
            "supply output_dir for a memory-mapped result."
        )
    source_stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources
    }
    report = {
        "status": "planned",
        "spec_version": resolved["spec_version"],
        "shape": list(shape),
        "dtype": "float64",
        "units": "V",
        "sample_coordinates": "zero-based, stop-exclusive source-file samples",
        "setting_origins": origins,
        "resolved_recipe": resolved,
        "resolved_recipe_sha256": hashlib.sha256(
            json.dumps(resolved, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "recordings": [info for _, info in recordings],
        "adjustments": adjustments,
        "exclusions": exclusions,
        "processing": [],
        "python_version": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("numpy", "mne", "mne-bids", "scikit-learn", "jsonschema")
        },
        "splitting_backend": (
            "scikit-learn train_test_split on canonically ordered groups; "
            "test then relative validation; no retries"
        ),
        "partitions": {
            name: {
                "cases": len(indices),
                "groups": len({cases[i]["group_id"] for i in indices}),
                "class_counts": {
                    label: sum(cases[i]["label"] == label for i in indices)
                    for label in resolved["target"]["class_order"]
                },
            }
            for name, indices in splits.items()
        },
        "sources": [
            {"path": str(path), "sha256": _hash(path)} for path in sorted(sources)
        ],
        "implementation_hashes": {
            path.name: _hash(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("_recipe.py"),
                Path(__file__).with_name("_classification.py"),
            )
        },
    }
    if destination is not None:
        destination.mkdir(parents=True)
        _write_json(destination / "run-report.json", report)
    try:
        X = (
            np.empty(shape, dtype=np.float64)
            if destination is None
            else np.lib.format.open_memmap(
                destination / "X.npy", mode="w+", dtype=np.float64, shape=shape
            )
        )
        for path, info in recordings:
            indices = [
                index
                for index, row in enumerate(cases)
                if row["recording_id"] == info["recording_id"]
            ]
            if not indices:
                continue
            raw = _open_raw(path)
            try:
                for index in indices:
                    row = cases[index]
                    signal = raw.get_data(
                        picks=info["channel_names"],
                        start=row["start_sample"],
                        stop=row["stop_sample"],
                    )
                    local_info = {
                        "recording_id": info["recording_id"],
                        "channel_names": info["channel_names"],
                        "sampling_frequency": info["sampling_frequency"],
                    }
                    local_case = {
                        **row,
                        "start_sample": 0,
                        "stop_sample": signal.shape[1],
                    }
                    bounded = build_classification_dataset(
                        [signal], [local_info], [local_case]
                    )
                    X[index] = bounded.X[0]
                    row["row"] = index
            finally:
                raw.close()
        if isinstance(X, np.memmap):
            X.flush()
        if any(
            (path.stat().st_size, path.stat().st_mtime_ns) != original
            for path, original in source_stats.items()
        ):
            raise ValueError("Source files changed during recipe construction.")
        y = np.asarray([row["label"] for row in cases])
        groups = np.asarray([row["group_id"] for row in cases])
        metadata = {
            "recordings": [info for _, info in recordings],
            "channel_names": recordings[0][1]["channel_names"],
            "sampling_frequency": recordings[0][1]["sampling_frequency"],
            "recipe": resolved,
        }
        # Use the first retained case, not an excluded recording, for output metadata.
        retained_info = next(
            info
            for _, info in recordings
            if info["recording_id"] == cases[0]["recording_id"]
        )
        metadata.update(
            channel_names=retained_info["channel_names"],
            sampling_frequency=retained_info["sampling_frequency"],
        )
        dataset = ClassificationDataset(X, y, groups, cases, exclusions, metadata)
        report["status"] = "complete"
        if destination is not None:
            np.save(destination / "y.npy", y)
            _write_json(destination / "resolved-recipe.json", resolved)
            _write_cases(destination / "cases.tsv", cases)
            _write_cases(destination / "exclusions.tsv", exclusions)
            _write_cases(
                destination / "split-assignments.tsv",
                [
                    {
                        "case_id": row["case_id"],
                        "group_id": row["group_id"],
                        "split": row["split"],
                    }
                    for row in cases
                ],
            )
            _write_json(destination / "run-report.json", report)
        return ClassificationRecipeResult(dataset, splits, resolved, report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        if destination is not None:
            _write_json(destination / "run-report.json", report)
        raise
