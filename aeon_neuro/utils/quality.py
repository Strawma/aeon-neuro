"""Inspect decoded EEG samples without changing the recording."""

from numbers import Integral

import numpy as np
from mne.io import BaseRaw


def inspect_eeg_samples(raw, *, chunk_size=100_000):
    """Report nonfinite decoded samples in bounded windows.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to inspect, including bad channels and annotated intervals.
        Samples, annotations and supplied metadata remain unchanged.
    chunk_size : int, default=100000
        Maximum samples per channel requested in each read. Must be positive.
        Reader-specific buffers can require additional memory.

    Returns
    -------
    report : dict
        JSON-compatible n_channels, n_times, first_samp, nonfinite_count and
        issues. Each affected channel has its count and first_sample, using
        absolute MNE sample indices including raw.first_samp. Only the first
        affected sample per channel is retained, keeping the report bounded.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw recording.
    ValueError
        If chunk_size is not a positive integer.

    Notes
    -----
    This opt-in scan reads all decoded samples and retains deferred loading.
    It does not repair values, mark bad channels or establish acquisition
    integrity, calibration or scientific signal quality. Reader interpolation,
    including mixed-rate EDF/BDF behaviour, still applies. Conversion does not
    call this scan automatically; callers decide how to handle its findings.
    Reader errors propagate rather than returning a partial report as complete.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw must be an MNE Raw recording.")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    counts = np.zeros(len(raw.ch_names), dtype=np.int64)
    first_samples = [None] * len(raw.ch_names)
    for start in range(0, raw.n_times, chunk_size):
        stop = min(start + chunk_size, raw.n_times)
        nonfinite = ~np.isfinite(raw.get_data(start=start, stop=stop))
        counts += np.count_nonzero(nonfinite, axis=1)
        for channel in np.flatnonzero(nonfinite.any(axis=1)):
            if first_samples[channel] is None:
                # Use the same absolute sample convention as MNE event arrays.
                first_samples[channel] = int(
                    raw.first_samp + start + nonfinite[channel].argmax()
                )

    return {
        "n_channels": len(raw.ch_names),
        "n_times": int(raw.n_times),
        "first_samp": int(raw.first_samp),
        "nonfinite_count": int(counts.sum()),
        "issues": [
            {
                "code": "nonfinite_samples",
                "channel": raw.ch_names[channel],
                "count": int(counts[channel]),
                "first_sample": first_samples[channel],
            }
            for channel in np.flatnonzero(counts)
        ],
    }
