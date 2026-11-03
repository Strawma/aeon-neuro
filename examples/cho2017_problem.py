"""Inspect local Cho2017 metadata, author a case plan, then load selected cases.

See docs/cho2017-problem.md for inputs, commands and the explicit quality policy.
This example uses NEMAR nm000245 v1.0.2 and never downloads data.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from aeon_neuro.datasets import (
    inspect_bids_eeg,
    materialize_classification_problem,
    open_bids_eeg_recording,
    plan_classification_problem,
)

DEFINITIONS = Path(__file__).with_name("problem_definitions")
ELIGIBILITY = "All published events; original MATLAB bad-trial flags not applied"


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_metadata(root, name, entries):
    if name not in entries:
        raise ValueError(
            f"Inspected metadata is absent from the release manifest: {name}"
        )
    path, entry = root / name, entries[name]
    content = path.read_bytes()
    if entry["checksum_algorithm"] == "git":
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    elif entry["checksum_algorithm"] == "sha256":
        actual = hashlib.sha256(content).hexdigest()
    else:
        raise ValueError("Unsupported metadata checksum algorithm")
    if len(content) != entry["size"] or actual != entry["checksum"]:
        raise ValueError(f"Metadata differs from the supplied release manifest: {name}")
    return path


def inspect_recording(root, manifest, subject):
    """Pin the release and use the shared BIDS inspector for source facts.

    This explicit inspection opens the BDF header without decoding its samples.
    Its expected content hash comes from the supplied release manifest; full
    signal hashing and selected-interval reading happen at materialization.
    """
    if type(subject) is not int or not 1 <= subject <= 52:
        raise ValueError("Use a Cho2017 subject number from 1 to 52")
    root = root.resolve()
    entries = {
        item["path"]: item for item in json.loads(manifest.read_text(encoding="utf-8"))
    }
    description = json.loads(
        _verified_metadata(root, "dataset_description.json", entries).read_text(
            encoding="utf-8-sig"
        )
    )
    if (
        description.get("DatasetDOI") != "10.82901/nemar.nm000245"
        or description.get("Version") != "1.0.2"
    ):
        raise ValueError("This example requires NEMAR nm000245 v1.0.2")
    stem = f"sub-{subject}/ses-0/eeg/sub-{subject}_ses-0_task-imagery_run-0"
    for suffix in ("_channels.tsv", "_events.tsv", "_eeg.json"):
        _verified_metadata(root, stem + suffix, entries)
    recording_id = stem + "_eeg.bdf"
    signal = entries[recording_id]
    if signal["checksum_algorithm"] != "sha256":
        raise ValueError("Expected a SHA256 signal entry in the release manifest")
    report = inspect_bids_eeg(
        root, subjects=[str(subject)], sessions=["0"], tasks=["imagery"], runs=["0"]
    )
    if not report["ready"]:
        errors = [
            f"{issue['code']}: {issue['message']}"
            for issue in report["issues"]
            if issue["severity"] == "error"
        ]
        raise ValueError("BIDS inspection blocked: " + "; ".join(errors))
    if report["source_ids"] != [recording_id]:
        raise ValueError("Expected exactly the selected Cho2017 BDF recording")
    recording = report["recordings"][0]
    files = recording["bids_source"]["files"]
    if files[recording_id]["size"] != signal["size"]:
        raise ValueError("Signal size differs from the release manifest")
    # Include inherited sidecars and participant data actually used by the
    # inspector, not only the four files named in the minimal example layout.
    for name, snapshot in files.items():
        if name != recording_id:
            path = _verified_metadata(root, name, entries)
            if _hash(path) != snapshot["sha256"]:
                raise ValueError(f"Metadata changed during inspection: {name}")
    if any(row.get("type") != "EEG" for row in recording["channel_metadata"]):
        raise ValueError("The Cho2017 example expects EEG-only channels")
    recording.update(
        units="V",
        release="NEMAR nm000245 v1.0.2",
        expected_sha256=signal["checksum"],
        source_size_bytes=signal["size"],
        eligibility_policy=ELIGIBILITY,
    )
    for row in report["events"]:
        # Retain this example's published row IDs alongside the adapter IDs.
        # Neither ID recovers original acquisition chronology or trial quality.
        row["event_id"] = f"events-row-{row['bids_event_row']}"
    report["release_manifest_sha256"] = _hash(manifest)
    return report


def make_plan(catalogue, problem, binding):
    """Plan using saved metadata and authored JSON, with no dataset path needed."""
    return plan_classification_problem(
        catalogue["recordings"], problem, binding, events=catalogue["events"]
    )


def _flagged_trials(values, count, origin):
    """Keep class-local flag coordinates explicit and reject missing evidence."""
    indices = np.asarray(values).reshape(-1)
    if indices.dtype.kind not in "iuf" or not np.isfinite(indices).all():
        raise ValueError("Source quality indices must be finite numbers")
    if np.any(indices != np.floor(indices)):
        raise ValueError("Source quality indices must be integers")
    indices = indices - origin
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("Source quality index is outside the declared trial range")
    indices = indices.astype(int)
    if len(set(indices)) != len(indices):
        raise ValueError("Source quality indices must not repeat")
    return set(indices.tolist())


def _quality_coordinates(index_origin, buffer_samples):
    """Reject implicit or malformed source-coordinate choices before reading."""
    if type(index_origin) is not int or index_origin not in (0, 1):
        raise ValueError("Choose an explicit quality index origin of 0 or 1")
    if type(buffer_samples) is not int or buffer_samples < 0:
        raise ValueError("Export buffer samples must be a nonnegative integer")


def _join_source_quality(eeg, events, *, index_origin, buffer_samples):
    """Match original class-local trials to every exported cue before flagging."""
    _quality_coordinates(index_origin, buffer_samples)
    markers = np.asarray(eeg.get("imagery_event")).reshape(-1)
    count = np.asarray(eeg.get("n_imagery_trials"))
    if (
        markers.dtype.kind not in "iuf"
        or not np.isfinite(markers).all()
        or count.ndim != 0
        or count.dtype.kind not in "iuf"
        or not np.isfinite(count)
        or count < 1
        or count != np.floor(count)
    ):
        raise ValueError("Invalid source trial count or imagery markers")
    count = int(count)
    cues = np.flatnonzero(markers)
    if len(cues) != count or len(events) != 2 * count or len(markers) % count:
        raise ValueError("Source trial count does not match the BIDS events")
    trial_width = len(markers) // count
    if not np.array_equal(cues, cues[0] + np.arange(count) * trial_width):
        raise ValueError("Source imagery markers do not match equal trial packages")
    sfreq = float(eeg["srate"])
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Invalid source sampling frequency")
    flags = eeg.get("bad_trial_indices", {})
    if not isinstance(flags, dict):
        raise ValueError("Missing source quality fields")
    voltage = flags.get("bad_trial_idx_voltage")
    emg = flags.get("bad_trial_idx_mi")
    if voltage is None or emg is None or len(voltage) != 2 or len(emg) != 2:
        raise ValueError("Expected separate left/right source quality arrays")
    joined = []
    for class_index, (label, field) in enumerate(
        (("left_hand", "imagery_left"), ("right_hand", "imagery_right"))
    ):
        if np.shape(eeg.get(field)) != (68, len(markers)):
            raise ValueError("Expected 64 EEG plus four EMG rows in the MATLAB source")
        bad_voltage = _flagged_trials(voltage[class_index], count, index_origin)
        bad_emg = _flagged_trials(emg[class_index], count, index_origin)
        offset = class_index * (len(markers) + buffer_samples)
        for trial, cue in enumerate(cues):
            row = dict(events[class_index * count + trial])
            expected_sample = offset + int(cue)
            # Counts alone would accept a shifted export and flag the wrong trial.
            if (
                row["trial_type"] != label
                or int(row["value"]) != class_index + 1
                or float(row["sample"]) != expected_sample
                or float(row["onset"]) * sfreq != expected_sample
            ):
                raise ValueError("BIDS event does not match its original MATLAB trial")
            flagged = trial in bad_voltage | bad_emg
            row.update(
                original_trial_id=f"{label}:{trial + index_origin}",
                source_trial_index=trial + index_origin,
                source_trial_start_sample=offset + trial * trial_width,
                source_trial_stop_sample=offset + (trial + 1) * trial_width,
                bad_voltage=int(trial in bad_voltage),
                bad_emg=int(trial in bad_emg),
                paper_quality="flagged" if flagged else "unflagged",
                included=not flagged,
                exclusion_reason="source_quality" if flagged else "",
            )
            joined.append(row)
    return joined


def inspect_quality_recording(
    root, manifest, subject, *, index_origin, buffer_samples, cohort
):
    """Prepare a source-quality catalogue, explicitly decoding the original MAT.

    This is separate from cheap BDF header inspection. It verifies and loads the
    selected original MATLAB file, including its signal arrays, then records its
    snapshot so materialization revalidates the evidence without decoding it again.
    """
    if cohort not in ("paper", "all"):
        raise ValueError("Choose cohort='paper' or cohort='all'")
    if cohort == "paper" and subject in (29, 34):
        raise ValueError("Subject excluded by the declared paper-cohort policy")
    _quality_coordinates(index_origin, buffer_samples)
    catalogue = inspect_recording(root, manifest, subject)
    root = root.resolve()
    entries = {
        item["path"]: item for item in json.loads(manifest.read_text(encoding="utf-8"))
    }
    name = f"sourcedata/s{subject:02}.mat"
    entry = entries.get(name, {})
    if entry.get("checksum_algorithm") != "sha256":
        raise ValueError("Expected an original MATLAB SHA256 entry in the manifest")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError("MATLAB source must remain inside the dataset root")
    stat = path.stat()
    if stat.st_size != entry["size"] or _hash(path) != entry["checksum"]:
        raise ValueError("MATLAB source differs from the release manifest")
    from scipy.io import loadmat

    eeg = loadmat(path, simplify_cells=True)["eeg"]
    recording = catalogue["recordings"][0]
    if float(eeg["srate"]) != recording["sampling_frequency"]:
        raise ValueError("MATLAB and BDF sampling frequencies disagree")
    joined = _join_source_quality(
        eeg,
        catalogue["events"],
        index_origin=index_origin,
        buffer_samples=buffer_samples,
    )
    if (
        path.stat().st_size != stat.st_size
        or path.stat().st_mtime_ns != stat.st_mtime_ns
    ):
        raise ValueError("MATLAB source changed during quality preparation")
    for row in joined:
        row["trial_group"] = json.dumps(
            [recording["recording_id"], row["original_trial_id"]], separators=(",", ":")
        )
    # The original file is required evidence at materialization, even though
    # planning itself only needs the small saved catalogue and flag witnesses.
    recording["bids_source"]["files"][name] = {
        "size": stat.st_size,
        "mtime_ns": str(stat.st_mtime_ns),
        "sha256": entry["checksum"],
    }
    recording["source_quality"] = {
        "path": name,
        "sha256": entry["checksum"],
        "join_implementation_sha256": _hash(Path(__file__)),
        "index_origin": index_origin,
        "export_buffer_samples": buffer_samples,
        "cohort": cohort,
        "excluded_subjects": [29, 34] if cohort == "paper" else [],
    }
    recording["eligibility_policy"] = (
        f"Exclude original voltage/EMG flagged trials; index origin {index_origin}; "
        f"export buffer {buffer_samples} samples; cohort {cohort}. "
        "Unflagged does not certify artifact-free EEG."
    )
    catalogue["events"] = joined
    return catalogue


def load_cases(root, plan, case_ids=None):
    """Verify the selected local source and materialize its planned raw samples."""
    return materialize_classification_problem(
        plan,
        lambda descriptor: open_bids_eeg_recording(root, descriptor),
        case_ids=case_ids,
    )


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    """Keep source inspection, metadata planning and signal loading explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read local headers and sidecars")
    inspect.add_argument("--dataset", type=Path, required=True)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--subject", type=int, default=1)
    inspect.add_argument("--output", type=Path, required=True)
    quality = commands.add_parser(
        "quality", help="Verify and decode source quality MAT"
    )
    quality.add_argument("--dataset", type=Path, required=True)
    quality.add_argument("--manifest", type=Path, required=True)
    quality.add_argument("--subject", type=int, required=True)
    quality.add_argument("--index-origin", type=int, choices=(0, 1), required=True)
    quality.add_argument("--buffer-samples", type=int, required=True)
    quality.add_argument("--cohort", choices=("paper", "all"), required=True)
    quality.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="Plan from saved metadata and JSON")
    plan.add_argument("--catalogue", type=Path, required=True)
    plan.add_argument(
        "--problem", type=Path, default=DEFINITIONS / "cho2017.problem.json"
    )
    plan.add_argument(
        "--binding", type=Path, default=DEFINITIONS / "cho2017.binding.json"
    )
    plan.add_argument("--output", type=Path, required=True)
    load = commands.add_parser("materialize", help="Verify and load selected cases")
    load.add_argument("--dataset", type=Path, required=True)
    load.add_argument("--plan", type=Path, required=True)
    load.add_argument("--case-id", action="append")
    load.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output path")
    if args.command in ("inspect", "quality"):
        if args.command == "quality":
            catalogue = inspect_quality_recording(
                args.dataset,
                args.manifest,
                args.subject,
                index_origin=args.index_origin,
                buffer_samples=args.buffer_samples,
                cohort=args.cohort,
            )
        else:
            catalogue = inspect_recording(args.dataset, args.manifest, args.subject)
        _save(args.output, catalogue)
        logging.info(
            "Saved local source descriptors and %s events", len(catalogue["events"])
        )
    elif args.command == "plan":
        plan = make_plan(
            json.loads(args.catalogue.read_text(encoding="utf-8")),
            args.problem,
            args.binding,
        )
        _save(args.output, plan)
        logging.info("Planned %s cases; no signal source opened", len(plan["cases"]))
    else:
        dataset = load_cases(
            args.dataset,
            json.loads(args.plan.read_text(encoding="utf-8")),
            args.case_id,
        )
        args.output.mkdir()
        np.save(args.output / "X.npy", dataset.X, allow_pickle=False)
        np.save(args.output / "y.npy", dataset.y, allow_pickle=False)
        _save(args.output / "cases.json", dataset.cases)
        _save(args.output / "exclusions.json", dataset.exclusions)
        _save(args.output / "metadata.json", dataset.metadata)
        logging.info(
            "Saved raw cases with shape %s; eligibility in metadata.json",
            dataset.X.shape,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
