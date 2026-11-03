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
        _save(args.output / "metadata.json", dataset.metadata)
        logging.info("Saved raw cases with shape %s; %s", dataset.X.shape, ELIGIBILITY)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
