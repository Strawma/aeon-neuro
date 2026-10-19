"""Test metadata review, explicit choices and rejection before writer effects."""

import json
import warnings

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_array_equal

from aeon_neuro.utils.bids import convert_eeg_to_bids
from aeon_neuro.utils.metadata_conflicts import (
    MetadataConflictError,
    MetadataResolutionWarning,
    inspect_eeg_metadata,
    reconcile_eeg_metadata,
)


@pytest.fixture
def raw():
    """Provide known samples and timing alongside incomplete acquisition metadata."""
    raw = mne.io.RawArray(
        np.arange(400).reshape(2, 200) * 1e-6,
        mne.create_info(["Cz", "Eye"], 100, ["eeg", "eeg"]),
    )
    raw.set_annotations(mne.Annotations([0.1], [0.05], ["marker"]))
    return raw


def test_inspection_collects_conflicts_without_mutation(raw):
    """Report every disagreement, including representational equivalence."""
    raw.info["line_freq"] = 60
    sources = [
        {
            "source": "matched JSON/channels",
            "line_freq": 50,
            "sfreq": 100.0,
            "channel_types": {"Cz": "EEG", "Eye": "EOG"},
        }
    ]
    report = inspect_eeg_metadata(raw, metadata_sources=sources)
    blockers = [f for f in report["findings"] if f["blocking"]]
    assert [(f["field"], f["channel"]) for f in blockers] == [
        ("line_freq", None),
        ("channel_type", "Eye"),
    ]
    assert all(f["code"] == "conflict" for f in blockers)
    assert report["findings"][1]["code"] == "consistent"
    assert report["decisions"] == [] and not report["ready"]
    assert sources[0]["channel_types"]["Cz"] == "EEG"
    assert raw.info["line_freq"] == 60
    assert raw.get_channel_types() == ["eeg", "eeg"]
    json.dumps(report, allow_nan=False)


def test_unknown_frequency_is_permitted(raw):
    """Allow unknown frequency rather than inventing an acquisition fact."""
    with warnings.catch_warnings(record=True) as caught:
        prepared, report = reconcile_eeg_metadata(
            raw, metadata_sources=[{"source": "JSON", "line_freq": "n/a"}]
        )
    assert report["ready"] and report["decisions"] == []
    assert report["findings"][0]["code"] == "unknown"
    assert prepared.info["line_freq"] is None and not caught


def test_automatic_fill_retains_all_candidates(raw):
    """Fill only absent frequency when known sources agree, with a typed warning."""
    sources = [
        {"source": "JSON", "line_freq": 50},
        {"source": "protocol", "line_freq": 50.0},
    ]
    before = raw.get_data().copy()
    with pytest.warns(MetadataResolutionWarning, match="fill_missing"):
        prepared, report = reconcile_eeg_metadata(raw, metadata_sources=sources)
    assert prepared.info["line_freq"] == 50 and raw.info["line_freq"] is None
    assert len(report["findings"][0]["candidates"]) == 3
    assert report["decisions"][0]["method"] == "fill_missing"
    assert report["decisions"][0]["applied"]
    assert report["corrections"][0]["source"] == "JSON; protocol"
    assert_array_equal(prepared.get_data(), before)
    assert_array_equal(prepared.annotations.onset, raw.annotations.onset)
    json.dumps(report, allow_nan=False)


def test_known_disagreement_needs_explicit_choice(raw):
    """No source ordering or majority vote silently selects a known value."""
    raw.info["line_freq"] = 60
    sources = [
        {"source": "JSON", "line_freq": 50},
        {"source": "protocol", "line_freq": 50},
    ]
    with pytest.raises(MetadataConflictError, match="60.0.*raw.*50.0.*JSON") as error:
        reconcile_eeg_metadata(raw, metadata_sources=sources)
    assert isinstance(error.value, ValueError)
    assert error.value.report["decisions"] == []
    json.dumps(error.value.report, allow_nan=False)


def test_explicit_choices_include_retaining_imported_value(raw):
    """Record an approved decision even when it makes no actual metadata change."""
    raw.info["line_freq"] = 60
    sources = [{"source": "JSON", "line_freq": 50, "channel_types": {"Eye": "eog"}}]
    with pytest.warns(MetadataResolutionWarning) as caught:
        prepared, report = reconcile_eeg_metadata(
            raw,
            metadata_sources=sources,
            line_freq=60,
            channel_types={"Eye": "eeg"},
            correction_source="verified acquisition note",
        )
    assert len(caught) == 2
    assert report["ready"] and report["corrections"] == []
    assert len(report["decisions"]) == 2
    assert all(d["method"] == "explicit" and d["applied"] for d in report["decisions"])
    assert prepared.info["line_freq"] == 60
    assert prepared.get_channel_types() == raw.get_channel_types()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -50, True, "50"])
def test_invalid_candidates_remain_visible(raw, invalid):
    """A valid alternative does not hide an invalid source without approval."""
    sources = [
        {"source": "bad JSON", "line_freq": invalid},
        {"source": "protocol", "line_freq": 50},
    ]
    with pytest.raises(MetadataConflictError) as error:
        reconcile_eeg_metadata(raw, metadata_sources=sources)
    assert error.value.report["findings"][0]["code"] == "invalid"
    json.dumps(error.value.report, allow_nan=False)
    with pytest.warns(MetadataResolutionWarning):
        _, report = reconcile_eeg_metadata(
            raw,
            metadata_sources=sources,
            line_freq=50,
            correction_source="verified note",
        )
    assert report["findings"][0]["candidates"][1]["status"] == "invalid"


def test_sampling_mismatch_cannot_be_overridden(raw):
    """A mains-frequency choice cannot authorise a sampling-rate overwrite."""
    sources = [{"source": "JSON", "sfreq": 200, "line_freq": 50}]
    with pytest.raises(MetadataConflictError, match="sfreq") as error:
        reconcile_eeg_metadata(
            raw, metadata_sources=sources, line_freq=50, correction_source="note"
        )
    assert not error.value.report["ready"]
    assert not error.value.report["decisions"][0]["applied"]
    assert raw.info["sfreq"] == 100 and raw.info["line_freq"] is None


@pytest.mark.parametrize(
    "sources, exception",
    [
        ({"source": "JSON"}, TypeError),
        ([{"source": "raw"}], ValueError),
        ([{"source": "JSON"}, {"source": "JSON"}], ValueError),
        ([{"source": "JSON", "PowerLineFrequency": 50}], ValueError),
        ([{"source": "JSON", "channel_types": {"Absent": "eeg"}}], ValueError),
        ([{"source": "JSON", "channel_types": []}], TypeError),
    ],
)
def test_reject_ambiguous_source_schema(raw, sources, exception):
    """Do not silently ignore misspelled fields or unmatched channel identities."""
    with pytest.raises(exception):
        inspect_eeg_metadata(raw, metadata_sources=sources)


def test_invalid_role_is_reported_and_requires_source_for_choice(raw):
    """Distinguish an invalid external role from an approved final correction."""
    sources = [{"source": "channels", "channel_types": {"Eye": "not-a-type"}}]
    report = inspect_eeg_metadata(raw, metadata_sources=sources)
    assert report["findings"][2]["code"] == "invalid"
    with pytest.raises(ValueError, match="correction_source"):
        reconcile_eeg_metadata(
            raw, metadata_sources=sources, channel_types={"Eye": "eog"}
        )


@pytest.mark.parametrize("existing", [False, True])
def test_converter_blocks_before_output(raw, tmp_path, existing):
    """Reject all unresolved findings before actual export has side effects."""
    raw.info["line_freq"] = 60
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="review")
    if existing:
        destination.root.mkdir()
        (destination.root / "sentinel.txt").write_text("existing dataset")
    sources = [{"source": "JSON", "line_freq": 50, "channel_types": {"Eye": "eog"}}]
    with pytest.raises(MetadataConflictError) as error:
        convert_eeg_to_bids(
            raw,
            destination,
            metadata_sources=sources,
            format="BrainVision",
            allow_preload=True,
        )
    assert sum(f["blocking"] for f in error.value.report["findings"]) == 2
    if existing:
        assert list(destination.root.iterdir()) == [destination.root / "sentinel.txt"]
        assert (destination.root / "sentinel.txt").read_text() == "existing dataset"
    else:
        assert not destination.root.exists()


def test_reconciliation_preserves_deferred_access(raw, tmp_path, monkeypatch):
    """Do not read deferred samples while reviewing or correcting metadata."""
    path = tmp_path / "source_raw.fif"
    raw.save(path, overwrite=False, verbose="ERROR")
    deferred = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")

    def fail_read(*args, **kwargs):
        raise AssertionError("Metadata reconciliation must not read samples")

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", fail_read)
    with pytest.warns(MetadataResolutionWarning):
        prepared, report = reconcile_eeg_metadata(
            deferred, metadata_sources=[{"source": "JSON", "line_freq": 50}]
        )
    assert not prepared.preload and not deferred.preload
    assert deferred.info["line_freq"] is None and prepared.info["line_freq"] == 50
    assert report["ready"]


def test_resolved_converter_roundtrip(raw, tmp_path):
    """Write explicit approved metadata while preserving original samples/events."""
    pytest.importorskip("pybv")
    sources = [{"source": "JSON", "line_freq": 50, "channel_types": {"Eye": "eog"}}]
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="review")
    with pytest.warns(MetadataResolutionWarning):
        written = convert_eeg_to_bids(
            raw,
            destination,
            metadata_sources=sources,
            metadata_corrections={
                "channel_types": {"Eye": "eog"},
                "correction_source": "verified note",
            },
            format="BrainVision",
            allow_preload=True,
        )
    restored = read_raw_bids(written, verbose="ERROR")
    assert restored.info["line_freq"] == 50
    assert restored.get_channel_types() == ["eeg", "eog"]
    np.testing.assert_allclose(
        restored.get_data(), raw.get_data(), rtol=2e-7, atol=1e-13
    )
    assert_array_equal(restored.annotations.description, raw.annotations.description)
    assert raw.info["line_freq"] is None and raw.get_channel_types() == ["eeg", "eeg"]
