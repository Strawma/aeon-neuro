"""Exercise source inspection and selected reading with tiny generated BDF files."""

import hashlib
import json
from copy import deepcopy

import edfio
import mne
import numpy as np
import pytest

from aeon_neuro.datasets import _bids_input as adapter
from aeon_neuro.datasets import (
    inspect_bids_eeg,
    materialize_classification_problem,
    open_bids_eeg_recording,
    plan_classification_problem,
)


@pytest.fixture
def bids(tmp_path, request):
    """Use inherited EEG metadata, explicit subjects and distinguishable samples."""
    (tmp_path / "dataset_description.json").write_text(
        '{"Name":"Generated EEG","BIDSVersion":"1.9.0"}'
    )
    (tmp_path / "task-imagery_eeg.json").write_text(
        '{"SamplingFrequency":32,"RecordingDuration":8}'
    )
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tcohort\nsub-01\tcontrol\nsub-02\tpatient\n"
    )
    extension = getattr(request, "param", "bdf")
    signal_type, writer = (
        (edfio.BdfSignal, edfio.Bdf)
        if extension == "bdf"
        else (edfio.EdfSignal, edfio.Edf)
    )
    paths = []
    for subject in ("01", "02"):
        stem = tmp_path / f"sub-{subject}/eeg/sub-{subject}_task-imagery"
        stem.parent.mkdir(parents=True)
        path = stem.with_name(stem.name + f"_eeg.{extension}")
        values = np.random.default_rng(int(subject)).normal(size=(2, 256))
        writer(
            [
                signal_type(row, 32, label=name, physical_dimension="uV")
                for row, name in zip(values, ("C3", "C4"))
            ]
        ).write(path)
        stem.with_name(stem.name + "_channels.tsv").write_text(
            "name\ttype\tunits\nC3\tEEG\tuV\nC4\tEEG\tuV\n"
        )
        # Keep BOM reader coverage independent of the host's default encoding.
        stem.with_name(stem.name + "_events.tsv").write_text(
            "\ufeffonset\tduration\ttrial_type\tsample\n"
            "1\t2\tleft\t32\n4\t2\tright\t128\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tmp_path, paths


def problem():
    """Keep scientific choices outside the source adapter."""
    return {
        "format_version": 1,
        "id": "imagery",
        "version": "1",
        "question": "Classify the declared imagery target",
        "class_labels": ["left", "right"],
        "channels": ["C4", "C3"],
        "segmentation": {"mode": "event_epochs", "tmin": 0.5, "tmax": 1.5},
    }


def binding():
    """Name dataset columns and identities explicitly."""
    return {
        "format_version": 1,
        "problem": {"id": "imagery", "version": "1"},
        "target": {
            "source": "event",
            "column": "trial_type",
            "mapping": {"left": "left", "right": "right"},
        },
        "group": {"source": "recording", "column": "participant_id"},
        "events": {"onset": "onset", "id": "bids_event_id"},
        "case_fields": {
            "participant_id": {"source": "recording", "column": "participant_id"}
        },
    }


def test_inspect_and_plan_never_read_or_hash_signals(bids, monkeypatch):
    """Headers/tables suffice and the saved catalogue survives unavailable signals."""
    root, paths = bids
    original_hash = adapter._hash

    def metadata_hash(path):
        assert path.suffix != ".bdf", "Signal hash during inspection"
        return original_hash(path)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Sample decoding before explicit materialization")

    monkeypatch.setattr(adapter, "_hash", metadata_hash)
    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected_read)
    report = inspect_bids_eeg(root)
    assert report["ready"] and len(report["recordings"]) == 2
    first = report["recordings"][0]
    assert first["participant.cohort"] == "control"
    assert (
        first["metadata_origins"]["fields"]["SamplingFrequency"]
        == "task-imagery_eeg.json"
    )
    assert first["session"] is None
    assert report["events"][0]["sample"] == "32"
    assert all(issue["severity"] == "warning" for issue in report["issues"])
    for path in paths:
        path.unlink()
    saved = json.loads(json.dumps(report))
    plan = plan_classification_problem(
        saved["recordings"], problem(), binding(), events=saved["events"]
    )
    assert plan["dense_shape"] == [4, 2, 32]


def test_selection_reads_only_chosen_source_and_exact_samples(bids, monkeypatch):
    """The existing materializer owns one handle and preserves selected samples."""
    root, paths = bids
    report = inspect_bids_eeg(root)
    selected = report["recordings"][-1]
    selected["expected_sha256"] = adapter._hash(paths[-1])
    plan = plan_classification_problem(
        report["recordings"], problem(), binding(), events=report["events"]
    )
    case = plan["cases"][-1]
    opened = []
    original = adapter.load_eeg_to_mne

    def watched(path, **kwargs):
        opened.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(adapter, "load_eeg_to_mne", watched)
    dataset = materialize_classification_problem(
        plan,
        lambda descriptor: open_bids_eeg_recording(root, descriptor),
        case_ids=[case["case_id"]],
    )
    assert opened == [paths[-1]]
    with original(paths[-1], preload=False, verbose=False) as raw:
        np.testing.assert_array_equal(
            dataset.X[0], raw.get_data(picks=["C4", "C3"], start=144, stop=176)
        )
    subset = inspect_bids_eeg(root, subjects=["01"])
    assert subset["source_ids"] == [paths[0].relative_to(root).as_posix()]


@pytest.mark.parametrize(
    "fault",
    ["channels", "rate", "sample_alias", "ambiguous", "sidecar_tie", "unsupported"],
)
def test_blockers_return_diagnostics_without_partial_catalogue(bids, fault):
    """A broken selected recording cannot disappear into an apparently valid subset."""
    root, paths = bids
    stem = paths[-1].name.removesuffix("_eeg.bdf")
    if fault == "channels":
        paths[-1].with_name(stem + "_channels.tsv").write_text("name\nC4\nC3\n")
    elif fault == "rate":
        paths[-1].with_suffix(".json").write_text('{"SamplingFrequency":64}')
    elif fault == "sample_alias":
        paths[-1].with_name(stem + "_events.tsv").write_text(
            "onset\tduration\tsample\tonset_sample\n1\t2\t32\t33\n"
        )
    elif fault == "ambiguous":
        (root / "eeg.json").write_text('{"SamplingFrequency":32}')
    elif fault == "sidecar_tie":
        paths[-1].with_name(stem + "_channels.tsv").unlink()
        paths[-1].rename(
            paths[-1].with_name("sub-02_task-imagery_acq-a_run-01_eeg.bdf")
        )
        for name in [
            "sub-02_task-imagery_acq-a_channels.tsv",
            "sub-02_task-imagery_run-01_channels.tsv",
        ]:
            paths[-1].with_name(name).write_text("name\ttype\nC3\tEEG\nC4\tEEG\n")
    else:
        paths[-1].rename(paths[-1].with_suffix(".vhdr"))
    report = inspect_bids_eeg(root)
    assert not report["ready"]
    assert report["recordings"] is None and report["events"] is None
    assert len(report["source_ids"]) == 2
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert errors and errors[0]["message"]


def test_optional_tables_remain_unknown_instead_of_inferred(bids):
    """Recording targets can still use known headers when no event table exists."""
    root, paths = bids
    for path in paths:
        path.with_name(path.name.removesuffix("_eeg.bdf") + "_events.tsv").unlink()
    (root / "participants.tsv").unlink()
    report = inspect_bids_eeg(root)
    assert report["ready"] and report["events"] == []
    assert {issue["code"] for issue in report["issues"]} >= {
        "events_missing",
        "participants_missing",
        "quality_unreviewed",
    }
    assert "participant.cohort" not in report["recordings"][0]


@pytest.mark.parametrize("changed", ["signal", "metadata", "checksum", "version"])
def test_stale_or_untrusted_sources_fail_before_decode(bids, monkeypatch, changed):
    """Saved file snapshots and optional content checks precede the reader."""
    root, paths = bids
    descriptor = deepcopy(inspect_bids_eeg(root)["recordings"][0])
    if changed == "signal":
        with paths[0].open("ab") as stream:
            stream.write(b"x")
    elif changed == "metadata":
        (root / "participants.tsv").write_text(
            "participant_id\tcohort\nsub-01\tchanged\n"
        )
    elif changed == "checksum":
        descriptor["expected_sha256"] = hashlib.sha256(b"wrong source").hexdigest()
    else:
        descriptor["bids_source"]["version"] = 2

    def unexpected(*args, **kwargs):
        raise AssertionError("Stale source reached the decoder")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    with pytest.raises(ValueError):
        open_bids_eeg_recording(root, descriptor)


def test_no_matching_sources_is_an_explicit_blocker(bids):
    """A wrong selection is distinguishable from an empty valid dataset."""
    report = inspect_bids_eeg(bids[0], subjects=["99"])
    assert not report["ready"] and report["source_ids"] == []
    assert report["issues"][0]["code"] == "selection_failed"


def test_partial_filter_mismatch_blocks_before_header_inspection(bids, monkeypatch):
    """The adapter shares the existing loader's strict selection boundary."""

    def unexpected(*args, **kwargs):
        raise AssertionError("Partial selection reached a source header")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    report = inspect_bids_eeg(bids[0], subjects=["01", "typo"])
    assert not report["ready"] and report["recordings"] is None
    assert "typo" in report["issues"][0]["message"]


@pytest.mark.parametrize("bids", ["edf"], indirect=True)
def test_edf_sources_use_the_same_metadata_and_reader_boundary(bids):
    """Both advertised single-file encodings reach the existing case planner."""
    root, _ = bids
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"] and report["source_ids"][0].endswith(".edf")
    plan = plan_classification_problem(
        report["recordings"], problem(), binding(), events=report["events"]
    )
    dataset = materialize_classification_problem(
        plan, lambda item: open_bids_eeg_recording(root, item)
    )
    assert dataset.X.shape == (2, 2, 32)


def test_changed_header_closes_the_newly_owned_handle(bids, monkeypatch):
    """A descriptor/header mismatch must not leak its source handle."""
    root, paths = bids
    descriptor = inspect_bids_eeg(root)["recordings"][0]
    descriptor["n_samples"] += 1
    raw = adapter.load_eeg_to_mne(paths[0], preload=False, verbose=False)
    closed = []
    close = raw.close

    def watched_close():
        closed.append(True)
        close()

    monkeypatch.setattr(raw, "close", watched_close)
    monkeypatch.setattr(adapter, "load_eeg_to_mne", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="header differs"):
        open_bids_eeg_recording(root, descriptor)
    assert closed == [True]
