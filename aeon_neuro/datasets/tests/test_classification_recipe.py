"""Check defaults, real BIDS reads, bounded extraction and split independence."""

import csv
import json

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from aeon_neuro.datasets import _classification_recipe as module
from aeon_neuro.datasets import load_bids_classification

pytest.importorskip("jsonschema")
pytest.importorskip("pybv")


def _table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _recording(root, subject, *, task="rest", names=("Cz", "Fz", "AUX")):
    signal = (np.arange(3 * 64).reshape(3, 64) + int(subject) * 1000) * 1e-7
    raw = mne.io.RawArray(
        signal, mne.create_info(list(names), 8, ["eeg", "eeg", "misc"]), verbose=False
    )
    raw.info["line_freq"] = 50
    path = BIDSPath(root=root, subject=subject, task=task, datatype="eeg")
    return write_raw_bids(
        raw,
        path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose=False,
    )


def _fixed_recipe():
    return {
        "spec_version": "0.3.0-draft",
        "target": {"source": "participants.tsv", "column": "group"},
        "segmentation": {"mode": "fixed_windows", "window_seconds": 1},
        "evaluation": {"split": {"stratify_by": "target"}},
    }


@pytest.fixture
def fixed(tmp_path):
    """Make ten tiny BIDS recordings with balanced participant labels."""
    root = tmp_path / "bids"
    for index in range(1, 11):
        _recording(root, f"{index:02}")
    _table(
        root / "participants.tsv",
        [
            {"participant_id": f"sub-{index:02}", "group": "A" if index % 2 else "B"}
            for index in range(1, 11)
        ],
    )
    return root


@pytest.fixture
def events(tmp_path):
    """Make a single-pair event recording with four original trial blocks."""
    root = tmp_path / "events"
    path = _recording(root, "01", task="RPS", names=("1-A1", "2-A1", "AUX"))
    event_path = path.copy().update(suffix="events", extension=".tsv").fpath
    rows = [
        {
            "onset": (trial - 1) * 0.5,
            "duration": 0.5,
            "sample": (trial - 1) * 4,
            "onset_sample": (trial - 1) * 4,
            "trial_num": trial,
            "choice": "0" if trial == 1 else str(1 + (trial + 1) % 2),
        }
        for trial in range(1, 17)
    ]
    _table(event_path, rows)
    recipe = {
        "spec_version": "0.3.0-draft",
        "target": {"source": "events.tsv", "column": "choice", "exclude_values": ["0"]},
        "channels": {"include_prefixes": ["1-", "2-"]},
        "segmentation": {
            "mode": "event_epochs",
            "stop_seconds": 0.25,
            "event_id_column": "trial_num",
        },
        "evaluation": {
            "group_by": ["recording", "block"],
            "group_definitions": {
                "block": {"kind": "integer_bins", "column": "trial_num", "bin_size": 4}
            },
        },
    }
    return root, event_path, rows, recipe


def test_bids_fixed_windows_and_reproducible_groups(fixed):
    """Extract half-open windows and keep every subject in one partition."""
    first = load_bids_classification(fixed, _fixed_recipe())
    second = load_bids_classification(fixed, _fixed_recipe())
    assert first.dataset.X.shape == (80, 2, 8)
    assert first.recipe["selection"]["tasks"] == ["rest"]
    assert set(first.splits) == {"train", "test"}
    assert first.recipe["target"]["class_order"] == ["A", "B"]
    for name, indices in first.splits.items():
        np.testing.assert_array_equal(indices, second.splits[name])
        assert set(first.dataset.y[indices]) == {"A", "B"}
    assert not set(first.dataset.groups[first.splits["train"]]) & set(
        first.dataset.groups[first.splits["test"]]
    )
    case = first.dataset.cases[1]
    raw = module._open_raw(module._discover(fixed, first.recipe)[0])
    try:
        np.testing.assert_array_equal(
            first.dataset.X[1], raw.get_data(picks=["Cz", "Fz"], start=8, stop=16)
        )
    finally:
        raw.close()
    assert (case["start_sample"], case["stop_sample"]) == (8, 16)


def test_validation_and_runtime_seed(fixed):
    """Partition at group level and record the effective runtime override."""
    recipe = _fixed_recipe()
    recipe["evaluation"]["split"]["fractions"] = {
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
    }
    result = load_bids_classification(fixed, recipe, random_state=3)
    assert set(result.splits) == {"train", "validation", "test"}
    assert (
        result.report["setting_origins"]["evaluation.split.random_seed"]
        == "runtime_override"
    )
    assert sum(len(x) for x in result.splits.values()) == len(result.dataset.y)


def test_events_original_blocks_and_bounded_reads(events, monkeypatch):
    """Preserve trial identity through exclusions and request only epoch samples."""
    root, _, _, recipe = events
    original = mne.io.BaseRaw.get_data
    reads = []

    def bounded(self, *args, **kwargs):
        reads.append((kwargs.get("start"), kwargs.get("stop")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", bounded)
    result = load_bids_classification(root, recipe)
    assert result.dataset.X.shape == (15, 2, 2)
    assert result.dataset.cases[0]["event_id"] == "2"
    assert result.dataset.cases[0]["block"] == 1
    assert result.dataset.cases[3]["event_id"] == "5"
    assert result.dataset.cases[3]["block"] == 2
    assert all(stop is not None and stop - start == 2 for start, stop in reads)
    assert len(reads) == 15
    assert result.dataset.exclusions[0]["exclusion_reason"] == "excluded_target_value"


def test_no_single_subject_window_fallback(events):
    """Refuse to change the generalisation question when independent groups lack."""
    root, _, _, recipe = events
    recipe.pop("evaluation")
    with pytest.raises(ValueError, match="Insufficient independent groups"):
        load_bids_classification(root, recipe)


def test_every_sample_column_is_checked(events):
    """Prevent a valid preferred column from hiding another column's disagreement."""
    root, path, rows, recipe = events
    rows[4]["sample"] += 2
    _table(path, rows)
    with pytest.raises(ValueError, match="columns disagree"):
        load_bids_classification(root, recipe)


def test_cross_partition_sample_overlap(events):
    """Reject block-disjoint labels when epochs still share source samples."""
    root, _, _, recipe = events
    recipe["segmentation"]["stop_seconds"] = 1
    with pytest.raises(ValueError, match="overlap across partitions"):
        load_bids_classification(root, recipe)


def test_state_interval_and_truncation(fixed):
    """Respect transition markers and require explicit permission to truncate."""
    for path in fixed.rglob("*_eeg.vhdr"):
        event_path = path.with_name(path.name.replace("_eeg.vhdr", "_events.tsv"))
        _table(
            event_path,
            [
                {"onset": 0, "duration": 0, "sample": 0, "trial_type": "eyes_closed"},
                {"onset": 2, "duration": 0, "sample": 16, "trial_type": "eyes_open"},
            ],
        )
    recipe = _fixed_recipe()
    recipe["segmentation"].update(
        window_seconds=0.5,
        interval={
            "kind": "first_state_interval",
            "start_seconds": 0.5,
            "stop_seconds": 9,
            "initial_state": "eyes_closed",
            "end_states": ["eyes_open"],
        },
    )
    with pytest.raises(ValueError, match="exceeds the recording"):
        load_bids_classification(fixed, recipe)
    recipe["segmentation"]["interval"]["short_recording_action"] = "truncate_and_report"
    result = load_bids_classification(fixed, recipe)
    assert result.dataset.X.shape == (30, 2, 4)
    assert len(result.report["adjustments"]) == 10
    assert max(row["stop_sample"] for row in result.dataset.cases) == 16


def test_memory_map_and_destination_protection(fixed, tmp_path):
    """Persist results outside raw BIDS and refuse existing destinations."""
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="memory-mapped"):
        load_bids_classification(fixed, _fixed_recipe(), max_memory_mb=0.0001)
    result = load_bids_classification(
        fixed, _fixed_recipe(), output_dir=destination, max_memory_mb=0.0001
    )
    assert isinstance(result.dataset.X, np.memmap)
    np.testing.assert_array_equal(np.load(destination / "X.npy"), result.dataset.X)
    assert (
        json.loads((destination / "run-report.json").read_text())["status"]
        == "complete"
    )
    assert (destination / "cases.tsv").exists()
    assert (destination / "split-assignments.tsv").exists()
    with pytest.raises(ValueError, match="new and outside"):
        load_bids_classification(fixed, _fixed_recipe(), output_dir=destination)
    with pytest.raises(ValueError, match="new and outside"):
        load_bids_classification(fixed, _fixed_recipe(), output_dir=fixed / "results")


def test_failed_read_never_marked_complete(fixed, tmp_path, monkeypatch):
    """Keep a failed report when extraction yields nonfinite data."""
    original = mne.io.BaseRaw.get_data

    def nonfinite(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result[0, 0] = np.nan
        return result

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", nonfinite)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="non-finite"):
        load_bids_classification(fixed, _fixed_recipe(), output_dir=destination)
    report = json.loads((destination / "run-report.json").read_text())
    assert report["status"] == "failed"


def test_missing_join_and_channel_types(fixed):
    """Reject missing label or channel facts rather than guess them."""
    participants, _ = module._read_tsv(fixed / "participants.tsv")
    _table(fixed / "participants.tsv", participants[1:])
    with pytest.raises(ValueError, match="Missing participant join"):
        load_bids_classification(fixed, _fixed_recipe())
    _table(fixed / "participants.tsv", participants)
    for path in fixed.rglob("*_channels.tsv"):
        path.unlink()
    with pytest.raises(ValueError, match="channel selection needs"):
        load_bids_classification(fixed, _fixed_recipe())


def test_multiple_tasks_are_explicit(fixed):
    """Never silently combine distinct experimental paradigms."""
    _recording(fixed, "01", task="other")
    with pytest.raises(ValueError, match="Multiple tasks"):
        load_bids_classification(fixed, _fixed_recipe())
    recipe = _fixed_recipe()
    recipe["selection"] = {"tasks": ["rest"]}
    assert load_bids_classification(fixed, recipe).dataset.X.shape[0] == 80


def test_optional_subject_manifest_replay(fixed, tmp_path):
    """Apply one subject assignment to all its windows without a case-level file."""
    result = load_bids_classification(fixed, _fixed_recipe())
    assignments = {row["bids_subject"]: row["split"] for row in result.dataset.cases}
    _table(
        tmp_path / "split.tsv",
        [{"bids_subject": key, "split": value} for key, value in assignments.items()],
    )
    recipe = _fixed_recipe()
    recipe["evaluation"]["split"] = {
        "method": "supplied_manifest",
        "manifest": "split.tsv",
        "join_keys": ["bids_subject"],
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    replay = load_bids_classification(fixed, path)
    np.testing.assert_array_equal(replay.splits["train"], result.splits["train"])
    assert str(tmp_path / "split.tsv") in {
        source["path"] for source in replay.report["sources"]
    }
    with pytest.raises(ValueError, match="recipe filename"):
        load_bids_classification(fixed, recipe)
    recipe["evaluation"]["split"]["join_keys"] = ["typo"]
    _table(tmp_path / "split.tsv", [{"typo": "value", "split": "train"}])
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="join_keys must name available case fields"):
        load_bids_classification(fixed, path)


def test_chronological_blocks_are_explicit(events):
    """Hold out the final complete block when a temporal question is requested."""
    root, _, _, recipe = events
    recipe["evaluation"]["split"] = {"strategy": "chronological", "order_by": ["block"]}
    result = load_bids_classification(root, recipe)
    assert {result.dataset.cases[i]["block"] for i in result.splits["test"]} == {4}
    assert result.recipe["evaluation"]["split"]["random_seed"] is None
    with pytest.raises(ValueError, match="Chronological"):
        load_bids_classification(root, recipe, random_state=42)


def test_incompatible_channel_identity_is_not_intersected(fixed):
    """A different selected order must not silently change the feature meaning."""
    original = module._channel_names

    def reversed_first(raw, selector, path):
        names = original(raw, selector, path)
        return names[::-1] if path.subject == "01" else names

    from unittest.mock import patch

    with patch.object(module, "_channel_names", reversed_first):
        with pytest.raises(ValueError, match="identical selected channels"):
            load_bids_classification(fixed, _fixed_recipe())


def test_source_change_during_extraction_is_failed(fixed, tmp_path, monkeypatch):
    """Never associate constructed cases with a stale input-table hash."""
    original = mne.io.BaseRaw.get_data
    changed = False

    def mutate_source(self, *args, **kwargs):
        nonlocal changed
        result = original(self, *args, **kwargs)
        if not changed:
            with (fixed / "participants.tsv").open("a") as stream:
                stream.write("\n")
            changed = True
        return result

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", mutate_source)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="Source files changed"):
        load_bids_classification(fixed, _fixed_recipe(), output_dir=output)
    assert json.loads((output / "run-report.json").read_text())["status"] == "failed"


def test_headers_sharing_signal_cannot_cross_partitions(fixed):
    """Header identities cannot conceal the same physical BrainVision payload."""
    initial = load_bids_classification(fixed, _fixed_recipe())
    train = initial.dataset.cases[initial.splits["train"][0]]["recording_id"]
    test = initial.dataset.cases[initial.splits["test"][0]]["recording_id"]
    header = fixed / test
    lines = header.read_text().splitlines()
    header.write_text(
        "\n".join(
            (
                f"DataFile={(fixed / train).with_suffix('.eeg')}"
                if line.startswith("DataFile=")
                else line
            )
            for line in lines
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Source sample intervals overlap"):
        load_bids_classification(fixed, _fixed_recipe())


@pytest.mark.parametrize("stage", ["table", "split", "hash"])
def test_source_change_before_extraction_is_rejected(
    fixed, tmp_path, monkeypatch, stage
):
    """Fingerprints must precede metadata interpretation, not follow planning."""

    def mutate():
        table = fixed / "participants.tsv"
        table.write_text(table.read_text().replace("\tA", "\tZ"))

    if stage == "table":
        original = module._read_tsv

        def altered(path):
            value = original(path)
            if path.name == "participants.tsv":
                mutate()
            return value

        monkeypatch.setattr(module, "_read_tsv", altered)
    elif stage == "split":
        original = module._split_cases

        def altered(*args):
            value = original(*args)
            mutate()
            return value

        monkeypatch.setattr(module, "_split_cases", altered)
    else:
        original = module._hash

        def altered(path):
            value = original(path)
            if path.name == "participants.tsv":
                mutate()
            return value

        monkeypatch.setattr(module, "_hash", altered)
    output = tmp_path / "changed-before-extraction"
    with pytest.raises(ValueError, match="Source files changed"):
        load_bids_classification(fixed, _fixed_recipe(), output_dir=output)
    assert not output.exists()


def test_unknown_channel_exclusion_is_an_error(fixed):
    """An exclusion typo must not silently leave an unwanted channel selected."""
    recipe = _fixed_recipe()
    recipe["channels"] = {"types": ["eeg"], "exclude_names": ["ECG_typo"]}
    with pytest.raises(ValueError, match="Excluded channel names are absent"):
        load_bids_classification(fixed, recipe)


def test_unknown_chronological_order_field_is_an_error(events):
    """A case-field typo should identify the recipe choice, not raise KeyError."""
    root, _, _, recipe = events
    recipe["evaluation"]["split"] = {"strategy": "chronological", "order_by": ["typo"]}
    with pytest.raises(ValueError, match="order_by must name available case fields"):
        load_bids_classification(root, recipe)
