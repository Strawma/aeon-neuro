"""Test nonfinite-sample inspection without changing input recordings."""

import json
from pathlib import Path

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon_neuro.utils.quality import inspect_eeg_samples


def test_nonfinite_samples_across_chunks():
    """Include bad channels, boundary samples and absolute recording offsets."""
    data = np.zeros((2, 9))
    data[0, 0] = np.nan
    data[0, 4] = np.inf
    data[1, 8] = -np.inf
    raw = mne.io.RawArray(
        data.copy(),
        mne.create_info(["Cz", "Eye"], 100, ["eeg", "eog"]),
        first_samp=100,
    )
    raw.info["bads"] = ["Cz"]
    raw.set_annotations(mne.Annotations([0.02], [0.01], ["marker"]))
    annotations = raw.annotations.copy()

    report = inspect_eeg_samples(raw, chunk_size=np.int64(4))

    assert report == {
        "n_channels": 2,
        "n_times": 9,
        "first_samp": 100,
        "nonfinite_count": 3,
        "issues": [
            {
                "code": "nonfinite_samples",
                "channel": "Cz",
                "count": 2,
                "first_sample": 100,
            },
            {
                "code": "nonfinite_samples",
                "channel": "Eye",
                "count": 1,
                "first_sample": 108,
            },
        ],
    }
    json.dumps(report, allow_nan=False)
    assert_array_equal(raw.get_data(), data)
    assert_array_equal(raw.annotations.onset, annotations.onset)
    assert_array_equal(raw.annotations.description, annotations.description)
    assert raw.info["bads"] == ["Cz"]


def test_deferred_recording_scan(monkeypatch):
    """Read the complete bundled recording without unbounded requests or preload."""
    source = (
        Path(__file__).resolve().parents[3]
        / "example_raw_eeg/basic_classification_task/sub-01/ses-01/eeg"
        / "sub-01_ses-01_task-task_run-01_eeg.vhdr"
    )
    raw = mne.io.read_raw_brainvision(source, preload=False)
    read = raw.get_data
    windows = []

    def bounded_read(*, start, stop):
        windows.append((start, stop))
        return read(start=start, stop=stop)

    monkeypatch.setattr(raw, "get_data", bounded_read)
    report = inspect_eeg_samples(raw, chunk_size=4096)

    assert report["nonfinite_count"] == 0
    assert report["issues"] == []
    assert report["n_times"] == raw.n_times
    assert windows[0][0] == 0 and windows[-1][1] == raw.n_times
    assert all(stop - start <= 4096 for start, stop in windows)
    assert raw.preload is False


@pytest.mark.parametrize("chunk_size", [0, True, 1.5])
def test_invalid_chunk_size(chunk_size):
    """Reject windows that cannot bound a forward scan."""
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Cz"], 100, "eeg"))
    with pytest.raises(ValueError, match="positive integer"):
        inspect_eeg_samples(raw, chunk_size=chunk_size)


def test_inspection_requires_raw():
    """Require a recording with a defined channel and sample convention."""
    with pytest.raises(TypeError, match="MNE Raw"):
        inspect_eeg_samples(np.zeros((1, 10)))
