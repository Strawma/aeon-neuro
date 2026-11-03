"""Inspect local Cho2017 metadata, author a case plan, then load selected cases.

See docs/cho2017-problem.md for inputs, commands and the explicit quality policy.
This example uses NEMAR nm000245 v1.0.2 and never downloads data.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import mne
import numpy as np

from aeon_neuro.datasets import (
    materialize_classification_problem,
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


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _verified_metadata(root, name, entries):
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
    """Read a local BDF header and verified sidecars into a portable catalogue.

    This explicit inspection opens the BDF header without decoding its samples.
    Its expected content hash comes from the supplied release manifest; full
    signal hashing and selected-interval reading happen at materialization.
    """
    if type(subject) is not int or not 1 <= subject <= 52:
        raise ValueError("Use a Cho2017 subject number from 1 to 52")
    entries = {item["path"]: item for item in json.loads(manifest.read_text())}
    description = json.loads(
        _verified_metadata(root, "dataset_description.json", entries).read_text()
    )
    if (
        description.get("DatasetDOI") != "10.82901/nemar.nm000245"
        or description.get("Version") != "1.0.2"
    ):
        raise ValueError("This example requires NEMAR nm000245 v1.0.2")
    stem = f"sub-{subject}/ses-0/eeg/sub-{subject}_ses-0_task-imagery_run-0"
    channels = _rows(_verified_metadata(root, stem + "_channels.tsv", entries))
    events = _rows(_verified_metadata(root, stem + "_events.tsv", entries))
    sidecar = json.loads(
        _verified_metadata(root, stem + "_eeg.json", entries).read_text()
    )
    recording_id = stem + "_eeg.bdf"
    signal = entries[recording_id]
    if signal["checksum_algorithm"] != "sha256":
        raise ValueError("Expected a SHA256 signal entry in the release manifest")
    if (root / recording_id).stat().st_size != signal["size"]:
        raise ValueError("Signal size differs from the release manifest")
    with mne.io.read_raw_bdf(root / recording_id, preload=False, verbose=False) as raw:
        if (
            raw.ch_names != [row["name"] for row in channels]
            or any(row["type"] != "EEG" for row in channels)
            or raw.info["sfreq"] != sidecar["SamplingFrequency"]
        ):
            raise ValueError("BDF header and BIDS channel/rate metadata disagree")
        recording = {
            "recording_id": recording_id,
            "subject": f"sub-{subject}",
            "session": "0",
            "channel_names": list(raw.ch_names),
            "sampling_frequency": float(raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
            "first_samp": int(raw.first_samp),
            "units": "V",
            "release": "NEMAR nm000245 v1.0.2",
            "expected_sha256": signal["checksum"],
            "source_size_bytes": signal["size"],
            "eligibility_policy": ELIGIBILITY,
            "conversion": description.get("GeneratedBy", []),
        }
    for index, row in enumerate(events):
        if float(row["onset"]) * recording["sampling_frequency"] != int(row["sample"]):
            raise ValueError("Event onset and sample columns disagree")
        # These IDs identify published table rows, not original trial chronology.
        # Preserve the source columns so that this distinction remains inspectable.
        row.update(recording_id=recording_id, event_id=f"events-row-{index}")
    return {"recordings": [recording], "events": events}


def make_plan(catalogue, problem, binding):
    """Plan using saved metadata and authored JSON, with no dataset path needed."""
    return plan_classification_problem(
        catalogue["recordings"], problem, binding, events=catalogue["events"]
    )


def load_cases(root, plan, case_ids=None):
    """Verify the selected local source and materialize its planned raw samples."""

    def open_recording(descriptor):
        path = (root / descriptor["recording_id"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Recording path must stay inside the local dataset")
        if (
            path.stat().st_size != descriptor["source_size_bytes"]
            or _hash(path) != descriptor["expected_sha256"]
        ):
            raise ValueError("Signal differs from the supplied release manifest")
        return mne.io.read_raw_bdf(path, preload=False, verbose=False)

    return materialize_classification_problem(plan, open_recording, case_ids=case_ids)


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
    if args.command == "inspect":
        catalogue = inspect_recording(args.dataset, args.manifest, args.subject)
        _save(args.output, catalogue)
        logging.info(
            "Saved local source descriptors and %s events", len(catalogue["events"])
        )
    elif args.command == "plan":
        plan = make_plan(
            json.loads(args.catalogue.read_text()), args.problem, args.binding
        )
        _save(args.output, plan)
        logging.info("Planned %s cases; no signal source opened", len(plan["cases"]))
    else:
        dataset = load_cases(
            args.dataset, json.loads(args.plan.read_text()), args.case_id
        )
        args.output.mkdir()
        np.save(args.output / "X.npy", dataset.X, allow_pickle=False)
        np.save(args.output / "y.npy", dataset.y, allow_pickle=False)
        _save(args.output / "cases.json", dataset.cases)
        _save(args.output / "metadata.json", dataset.metadata)
        logging.info("Saved raw cases with shape %s; %s", dataset.X.shape, ELIGIBILITY)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
