"""Numerical preparation and case construction for EEG experiments."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import mne
import numpy as np
from mne.baseline import rescale
from scipy.signal import resample_poly

from aeon_neuro.datasets import (
    build_classification_dataset,
    make_event_cases,
    make_fixed_length_cases,
)
from aeon_neuro.experiments._selectors import (
    UNSUPERVISED_SELECTORS,
    make_unsupervised_selector,
)
from aeon_neuro.transformations.series import BandPowerSeriesTransformer


class ProblemError(ValueError):
    """A recipe that can't produce a dataset; the message says what to change."""


def _number(value, default=None):
    """Parse a numeric value or return the default."""
    if value in (None, "", "none"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalise(config):
    """Split a recipe into preprocessing, labels and window rules, filling defaults.

    Recipes saved by older versions of the console are still accepted.
    """
    prep = dict(config.get("preprocessing") or {})
    if "bandpass_hz" in prep and "highpass_hz" not in prep and "lowpass_hz" not in prep:
        prep["highpass_hz"], prep["lowpass_hz"] = (
            prep.get("bandpass_hz") or [None, None]
        )[:2]
    labels = dict(config.get("labels") or {})
    if labels.get("source") is None:
        labels["source"] = {"events": "events", "recordings": "field"}.get(
            labels.get("mode"), "events"
        )
        if labels["source"] == "field":
            labels.setdefault("field", "task")
    rules = dict(config.get("rules") or {})
    rules.setdefault("tmin_ms", prep.get("tmin_ms", -200))
    rules.setdefault("tmax_ms", prep.get("tmax_ms", 800))
    rules.setdefault("window_s", prep.get("epoch_seconds", 2))
    if prep.get("window_norm") is None:
        starts_before_marker = (
            labels["source"] == "events" and float(rules["tmin_ms"]) < 0
        )
        prep["window_norm"] = "baseline" if starts_before_marker else "none"
    return prep, labels, rules


def _class_labels(mapping):
    """{marker or value: class label} without the blank labels."""
    return {
        code: label.strip()
        for code, label in (mapping or {}).items()
        if label and label.strip()
    }


def shared_channels(metas, excluded):
    """Channels every recording has, and those of them that are ticked."""
    first = metas[0]["channel_names"]
    shared = [n for n in first if all(n in m["channel_names"] for m in metas[1:])]
    notes = []
    if any(len(m["channel_names"]) != len(shared) for m in metas):
        notes.append(
            f"Recordings don't all have the same channels; the {len(shared)} they share"
            " are used."
        )
    used = [n for n in shared if n not in set(excluded or [])]
    if len(used) < 2:
        raise ProblemError("Select at least two channels.")
    return shared, used, notes


DEFAULT_CONTINUOUS_ORDER = ("reference", "filter", "resample")


def continuous_order(prep):
    """Read the order for continuous steps. Older recipes used the default."""
    order = prep.get("continuous_order", DEFAULT_CONTINUOUS_ORDER)
    if (
        not isinstance(order, (list, tuple))
        or len(order) != len(DEFAULT_CONTINUOUS_ORDER)
        or any(not isinstance(step, str) for step in order)
        or set(order) != set(DEFAULT_CONTINUOUS_ORDER)
    ):
        raise ProblemError(
            "Continuous order must contain Reference, Filter and Resample exactly once"
            " each."
        )
    return tuple(order)


def transform_key(prep):
    """Return the settings that affect transformed recordings for cache keys."""
    return [
        sorted(prep.get("excluded_channels") or []),
        prep.get("reference") or "none",
        _number(prep.get("highpass_hz")),
        _number(prep.get("lowpass_hz")),
        _number(prep.get("resample_hz")),
        prep.get("anti_alias_resampling") is True,
        continuous_order(prep),
    ]


def _effective_lowpass(sfreq, high):
    """Preserve room for the MNE filter's transition band near Nyquist."""
    if high is not None and high >= sfreq / 2 * 0.95:
        return sfreq / 2 * 0.9
    return high


def _bandpass(data, sfreq, low, high):
    """Apply the active MNE filters."""
    high = _effective_lowpass(sfreq, high)
    if not (low or high):
        return data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(data, sfreq, low or None, high or None)


def _effective_reference(reference, names):
    """Resolve a named reference against available electrodes."""
    if reference == "none":
        return reference
    return reference if reference == "average" or reference in names else "none"


def _rereference(data, picked, used, reference, sfreq):
    """Apply the reference and keep only the selected analysis channels."""
    if reference == "none":
        return data
    raw = mne.io.RawArray(data, mne.create_info(picked, sfreq, "eeg"), verbose="error")
    mne.set_eeg_reference(
        raw,
        "average" if reference == "average" else [reference],
        copy=False,
        verbose="error",
    )
    return raw.get_data(picks=list(used))


def transform_recording(signal, meta, prep, used, target):
    """Apply each continuous step on the sampling grid produced by previous steps."""
    sfreq = meta["sampling_frequency"]
    names = meta["channel_names"]
    reference = _effective_reference(prep.get("reference") or "none", names)
    picked = list(used)
    # An excluded electrode can still supply the reference. Process it through
    # earlier operations so the reference stays aligned with the kept channels.
    if reference not in ("none", "average") and reference not in picked:
        picked.append(reference)
    data = np.asarray(signal[[names.index(name) for name in picked]], dtype=np.float64)
    order = continuous_order(prep)
    low, high = _number(prep.get("highpass_hz")), _number(prep.get("lowpass_hz"))
    changes_rate = target is not None and not np.isclose(target, sfreq)
    if changes_rate and order.index("resample") < order.index("filter"):
        if prep.get("anti_alias_resampling") is not True:
            raise ProblemError(
                "Enable anti-aliased resampling to lower the rate before filtering, or"
                " move Filter above Resample."
            )
        if low and (
            low >= target / 2 or (high and low >= _effective_lowpass(target, high))
        ):
            raise ProblemError(
                "Lower the high-pass cutoff for the new sampling rate, or move Filter"
                " above Resample."
            )
    for step in order:
        if step == "reference":
            data = _rereference(data, picked, used, reference, sfreq)
        elif step == "filter":
            data = _bandpass(data, sfreq, low, high)
        elif changes_rate:
            data = _resample(
                data, sfreq, target, prep.get("anti_alias_resampling") is True
            )
            sfreq = float(target)
    return data.astype(np.float32), sfreq


def _resample(data, sfreq, target, anti_alias):
    if anti_alias:
        ratio = _resampling_ratio(sfreq, target)
        return resample_poly(
            data, ratio.numerator, ratio.denominator, axis=-1, padtype="edge"
        )
    # Old recipes decimated by an integer stride. Keep that path for replay.
    downsample_target([sfreq], target)
    return data[:, :: round(sfreq / target)]


def _resampling_ratio(source, target):
    # Bound the FIR size while keeping the new sample grid aligned with
    # event onsets, even when a source has a non-integer rate or length.
    ratio = Fraction(target / source).limit_denominator(1000)
    if not ratio or abs(float(ratio) - target / source) > 1e-10:
        raise ProblemError(
            f"Cannot resample {source:g} Hz to {target:g} Hz on a stable sample grid."
        )
    return ratio


def common_rates(rates, minimum=20.0):
    """Rates every recording reaches by keeping every n-th sample."""
    lowest = min(rates)
    found = []
    for n in range(1, 101):
        candidate = lowest / n
        if candidate < minimum:
            break
        if all(abs(r / candidate - round(r / candidate)) < 1e-9 for r in rates):
            found.append(candidate)
    return found


def downsample_target(rates, requested, *, anti_alias=False):
    """Choose a common target, checking that the selected method can reach it."""
    rates = sorted(set(rates))
    if requested is None and len(rates) == 1:
        return None
    target = requested if requested is not None else rates[0]
    if not np.isfinite(target) or target <= 0:
        raise ProblemError("Choose a positive, finite resampling rate.")
    if anti_alias:
        if target > rates[0]:
            raise ProblemError(f"Choose a resampling rate at or below {rates[0]:g} Hz.")
        for source in rates:
            _resampling_ratio(source, target)
        return target
    unreachable = [
        r for r in rates if r < target or abs(r / target - round(r / target)) > 1e-9
    ]
    if not unreachable:
        return target
    possible = ", ".join(f"{r:g}" for r in common_rates(rates)[:6]) or "none"
    if requested is None:
        raise ProblemError(
            "Recordings have different sampling rates"
            f" ({', '.join(f'{r:g}' for r in rates)} Hz) and can't be matched by"
            f" keeping every n-th sample. Common rates: {possible}. Try anti-aliased"
            " resampling under Change preparation settings."
        )
    raise ProblemError(
        f"{target:g} Hz can't be reached from"
        f" {', '.join(f'{r:g}' for r in unreachable)} Hz by keeping every n-th sample."
        f" Possible: {possible} Hz."
    )


def recording_plan(metas, prep):
    """Resolve the fixed electrode selection and common output rate."""
    shared, used, notes = shared_channels(metas, prep.get("excluded_channels"))
    requested = _number(prep.get("resample_hz"))
    rates = [m["sampling_frequency"] for m in metas]
    anti_alias = prep.get("anti_alias_resampling") is True
    target = downsample_target(rates, requested, anti_alias=anti_alias)
    if anti_alias and target is not None and any(rate != target for rate in rates):
        notes.append(
            f"Recordings were aligned at {target:g} Hz. Changed rates use a polyphase"
            " anti-alias filter."
        )
    if not anti_alias and requested is None and target is not None:
        notes.append(
            "Recordings have different sampling rates; all are downsampled to"
            f" {target:g} Hz."
        )
    report = {
        "total": len(shared),
        "dropped": [n for n in shared if n not in used],
        "used": len(used),
    }
    return used, target, report, notes


def transformed_metadata(meta, used, sfreq):
    """Keep source information and update the current sample grid and electrodes."""
    kept = {k: v for k, v in meta.items() if k not in ("channel_names", "n_samples")}
    return {
        **kept,
        "channel_names": list(used),
        "sampling_frequency": float(sfreq),
        "source_sampling_frequency": meta["sampling_frequency"],
    }


def _on_grid(seconds, sfreq):
    """Round a time to the nearest sample."""
    return round(float(seconds) * sfreq) / sfreq


def _event_cases(X, metas, labels, rules):
    """Build one case per labelled marker."""
    mapping = _class_labels(labels.get("events"))
    if len(set(mapping.values())) < 2:
        raise ProblemError("Give at least two markers different class labels.")
    sfreq = metas[0]["sampling_frequency"]
    tmin = _on_grid(float(rules["tmin_ms"]) / 1000, sfreq)
    tmax = _on_grid(float(rules["tmax_ms"]) / 1000, sfreq)
    if tmax <= tmin:
        raise ProblemError("The trial end must be after its start.")
    rows = [
        {
            "recording_id": meta["recording_id"],
            "onset": _on_grid(onset, meta["sampling_frequency"]),
            "label": mapping[code],
            "group_id": f"sub-{meta['subject']}",
            "participant_id": f"sub-{meta['subject']}",
            "marker": code,
        }
        for meta in metas
        for onset, code in meta["events"]
        if code in mapping
    ]
    if not rows:
        raise ProblemError("None of the labelled markers occur in the data.")
    cases = make_event_cases(X, metas, rows, tmin=tmin, tmax=tmax)
    baseline = int(round(-tmin * sfreq)) if tmin < 0 else 0
    return cases, (tmin, tmax), baseline, mapping


def _window_settings(rules, sfreq):
    """Window length, step, start and stop in seconds, on the sample grid."""
    size = _on_grid(_number(rules.get("window_s"), 2.0), sfreq)
    step = _on_grid(_number(rules.get("step_s"), size) or size, sfreq)
    start = max(_on_grid(_number(rules.get("start_s"), 0.0), sfreq), 0.0)
    stop = _number(rules.get("stop_s"))
    stop = _on_grid(stop, sfreq) if stop else None
    if size <= 0 or step <= 0:
        raise ProblemError("Window length and step must be positive.")
    if stop is not None and stop <= start:
        raise ProblemError("The time range ends before it starts.")
    return size, step, start, stop


def _label_recordings(X, metas, labels, excluded):
    """Assign classes to recordings from the chosen field."""
    field = labels.get("field") or "task"
    values = labels.get("values") or {}
    kept_X, kept_metas, rec_labels, mapping = [], [], [], {}
    for x, meta in zip(X, metas):
        value = meta.get("fields", {}).get(field, "")
        label = (values.get(value, value) or "").strip()
        if not label:
            excluded[f"no {field} label" if value else f"{field} missing"] += 1
            continue
        mapping[value] = label
        kept_X.append(x)
        kept_metas.append(meta)
        rec_labels.append(label)
    if len(set(rec_labels)) < 2:
        raise ProblemError(
            f"Recordings need at least two different '{field}' classes "
            f"(found {', '.join(sorted(set(rec_labels))) or 'none'})."
        )
    return kept_X, kept_metas, rec_labels, mapping


def _window_state(t0, t1, timeline, roles, use_states):
    """Why a window from t0 to t1 is dropped (or None), and the state it lies in."""
    if t1 > timeline["end_at"]:
        return f"after {timeline['end_name']}", None
    skip_on, skip_names = timeline["skips"]
    if len(skip_on):
        i = np.searchsorted(skip_on, t0, side="left")
        if i < len(skip_on) and skip_on[i] < t1:
            return f"crosses {skip_names[i]}", None
    if not use_states:
        return None, None
    state_on, state_names = timeline["states"]
    last = np.searchsorted(state_on, t0, side="right") - 1
    if last < 0:
        return "before any state marker", None
    if last + 1 < len(state_on) and state_on[last + 1] < t1:
        return "crosses a state change", None
    state = state_names[last]
    return (None, state) if roles.get(state) == "state_in" else (f"in {state}", None)


def _marker_timeline(events, roles):
    """Onsets of a recording's end, no-crossing and state markers."""
    events = sorted(events)

    def onsets(kinds):
        """Onsets and names of the markers with one of these roles."""
        chosen = [(o, c) for o, c in events if roles.get(c) in kinds]
        return np.array([o for o, _ in chosen]), [c for _, c in chosen]

    ends = [(o, c) for o, c in events if roles.get(c) == "end"]
    end_at, end_name = ends[0] if ends else (np.inf, "end marker")
    return {
        "end_at": end_at,
        "end_name": end_name,
        "skips": onsets({"skip"}),
        "states": onsets({"state_in", "state_out"}),
    }


def _window_cases(X, metas, labels, rules):
    """Build fixed-length windows with field or state labels."""
    source = labels["source"]
    sfreq = metas[0]["sampling_frequency"]
    size, step, start, stop = _window_settings(rules, sfreq)
    roles = dict(rules.get("markers") or {})
    state_labels = {}
    if source == "states":
        state_labels = _class_labels(labels.get("events"))
        if len(set(state_labels.values())) < 2:
            raise ProblemError(
                "Give at least two state markers different class labels."
            )
        roles.update({code: "state_in" for code in state_labels})
    excluded_recordings = Counter()
    if source == "field":
        X, metas, rec_labels, mapping = _label_recordings(
            X, metas, labels, excluded_recordings
        )
    else:
        rec_labels, mapping = ["_"] * len(metas), state_labels
    cases = []
    for x, meta, label in zip(X, metas, rec_labels):
        if x.shape[1] / sfreq < start + size:
            excluded_recordings["shorter than one window"] += 1
            continue
        cases += make_fixed_length_cases(
            [x], [meta], labels=[label], groups=[f"sub-{meta['subject']}"],
            window_size=size, step_size=step, start=start, stop=stop,
        )  # fmt: skip
    if not cases:
        raise ProblemError("No windows fit: check the window length and time range.")
    use_states = any(role in ("state_in", "state_out") for role in roles.values())
    timelines = {m["recording_id"]: _marker_timeline(m["events"], roles) for m in metas}
    for case in cases:
        t0, t1 = case["start_sample"] / sfreq, case["stop_sample"] / sfreq
        case.update(participant_id=case["group_id"], start_s=t0, stop_s=t1)
        reason, state = _window_state(
            t0, t1, timelines[case["recording_id"]], roles, use_states
        )
        if reason:
            case.update(included=False, exclusion_reason=reason)
            if source == "states":
                case["label"] = None
        elif source == "states":
            case.update(label=state_labels[state], state=state)
    return cases, (0.0, size), 0, mapping, excluded_recordings


def _reject_by_amplitude(cases, X, metas, limit_uv):
    """Drop cases exceeding the peak-to-peak limit on any channel.

    Returns how many were dropped and how often each channel was too large.
    """
    names = metas[0]["channel_names"]
    info = mne.create_info(names, metas[0]["sampling_frequency"], "eeg")
    worst, rejected = Counter(), 0
    for x, meta in zip(X, metas):
        mine = [
            c for c in cases
            if c["recording_id"] == meta["recording_id"] and c.get("included", True)
            and c["start_sample"] >= 0 and c["stop_sample"] <= x.shape[1]
        ]  # fmt: skip
        if not mine:
            continue
        data = np.stack(
            [
                np.asarray(x[:, c["start_sample"] : c["stop_sample"]], dtype=np.float64)
                for c in mine
            ]
        )
        epochs = mne.EpochsArray(data, info, verbose="error")
        epochs.drop_bad(reject={"eeg": limit_uv * 1e-6}, verbose="error")
        for case, log in zip(mine, epochs.drop_log):
            if log:
                case.update(
                    included=False,
                    exclusion_reason=f"over {limit_uv:g} µV peak-to-peak",
                )
                worst.update(log)
                rejected += 1
    return rejected, worst


def _reject_bad_spans(cases, metas):
    """Exclude any case whose half-open interval intersects a BAD span."""
    by_recording = {}
    for meta in metas:
        spans = sorted(meta.get("bad_spans") or [])
        if spans:
            starts, stops = np.asarray(spans).T
            # A late, short annotation can sit inside an earlier, longer one.
            # The cumulative end preserves both without scanning for each case.
            by_recording[meta["recording_id"]] = (
                starts,
                np.maximum.accumulate(stops),
                meta["sampling_frequency"],
            )
    rejected = 0
    for case in cases:
        recording = case["recording_id"]
        if recording not in by_recording or not case.get("included", True):
            continue
        starts, stops, rate = by_recording[recording]
        start, stop = case["start_sample"] / rate, case["stop_sample"] / rate
        index = np.searchsorted(starts, stop, side="left") - 1
        if index >= 0 and stops[index] > start:
            case.update(included=False, exclusion_reason="overlaps annotated BAD span")
            rejected += 1
    return rejected


def build_dataset(X, metas, labels, rules, prep):
    """Apply requested case exclusions before building the classification array."""
    notes = []
    excluded_recordings = Counter()
    if labels["source"] == "events":
        cases, window, baseline, mapping = _event_cases(X, metas, labels, rules)
    else:
        cases, window, baseline, mapping, excluded_recordings = _window_cases(
            X, metas, labels, rules
        )
        used = {c["recording_id"] for c in cases}
        X = [x for x, m in zip(X, metas) if m["recording_id"] in used]
        metas = [m for m in metas if m["recording_id"] in used]
    if prep.get("exclude_bad_spans") is True:
        rejected = _reject_bad_spans(cases, metas)
        notes.append(f"Annotated BAD spans excluded {rejected} of {len(cases)} cases.")
    norm = prep.get("window_norm") or "none"
    limit = _number(prep.get("reject_uv"))
    rejected, worst = 0, Counter()
    if limit:
        rejected, worst = _reject_by_amplitude(cases, X, metas, limit)
    channels = ", ".join(n for n, _ in worst.most_common(3))
    if not any(c.get("included", True) for c in cases):
        reasons = Counter(c.get("exclusion_reason") for c in cases)
        top = ", ".join(f"{n} {r}" for r, n in reasons.most_common(3))
        raise ProblemError(
            f"Every candidate was excluded ({top})."
            + (f" Amplitude mostly on {channels}." if worst else "")
        )
    if limit and rejected > 0.3 * len(cases):
        notes.append(
            f"{rejected} of {len(cases)} cases go over {limit:g} µV peak-to-peak,"
            f" mostly on {channels}."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = build_classification_dataset(
                X, metas, cases, on_out_of_bounds="exclude"
            )
    except ValueError as error:
        if "No included cases" in str(error):
            raise ProblemError("No cases fit inside the recordings.") from error
        raise ProblemError(str(error)) from error
    signals = np.asarray(dataset.X, dtype=np.float32)
    signals *= np.float32(1e6)
    times = (
        window[0] + np.arange(signals.shape[2]) / dataset.metadata["sampling_frequency"]
    )
    if norm == "baseline" and baseline:
        rescale(signals, times, (None, 0), mode="mean", copy=False, verbose="error")
    elif norm == "centre":
        rescale(signals, times, (None, None), mode="mean", copy=False, verbose="error")
    dataset.X = signals
    dataset.metadata.update(
        window_s=window,
        mode=labels["source"],
        label_map=mapping,
        excluded_recordings=dict(excluded_recordings),
        offset_correction=(
            norm if norm == "centre" or (norm == "baseline" and baseline) else "none"
        ),
    )
    return dataset, notes


def choose_channels(dataset, prep):
    """Apply legacy all-case channel selection and return a note."""
    choice = prep.get("auto_channels") or {}
    method = choice.get("method")
    if not method:
        return None
    if method not in UNSUPERVISED_SELECTORS:
        raise ProblemError(f"Unknown automatic channel choice: {method}.")
    names = list(dataset.metadata["channel_names"])
    keep = int(_number(choice.get("keep"), 8))
    if not 2 <= keep < len(names):
        raise ProblemError(
            f"Automatic channels: keep between 2 and {len(names) - 1} of the"
            f" {len(names)} channels."
        )
    selector = make_unsupervised_selector(method, keep, len(names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(dataset.X)
    kept = sorted(int(i) for i in selector.channels_selected_)
    dataset.X = np.ascontiguousarray(dataset.X[:, kept])
    dataset.metadata["channel_names"] = [names[i] for i in kept]
    dataset.metadata["auto_channels"] = {
        "method": UNSUPERVISED_SELECTORS[method],
        "kept": [names[i] for i in kept],
    }
    return (
        f"{UNSUPERVISED_SELECTORS[method]} kept {len(kept)} of {len(names)} channels:"
        f" {', '.join(names[i] for i in kept)}."
    )


def band_power(dataset, prep, progress=None):
    """Replace each case with delta-to-gamma band-power series.

    Half-second windows every 50 ms; MiniRocket needs at least 9 points per case.
    """
    if not prep.get("band_power"):
        return
    sfreq = float(dataset.metadata["sampling_frequency"])
    if sfreq < 120:
        raise ProblemError(
            f"Band power needs at least 120 Hz; the data is at {sfreq:g} Hz. Downsample"
            " less."
        )
    window, stride = int(sfreq // 2), max(1, int(sfreq // 20))
    n_times = dataset.X.shape[2]
    points = (n_times - window) // stride + 1 if n_times >= window else 0
    if points < 9:
        raise ProblemError(
            f"Band power gives {points} point{'' if points == 1 else 's'} per case;"
            " MiniRocket needs 9. Use cases of at least"
            f" {(window + 8 * stride) / sfreq:g} s."
        )
    transformer = BandPowerSeriesTransformer(
        sfreq=sfreq, window_size=window, stride=stride
    )
    out = np.empty((len(dataset.X), 5, points), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, case in enumerate(dataset.X):
            out[i] = transformer.fit_transform(np.asarray(case, dtype=np.float64))
            if progress and i % 500 == 0:
                progress(i / len(dataset.X), f"Band power {i}/{len(dataset.X)}…")
    dataset.metadata["band_power"] = {
        "from_channels": list(dataset.metadata["channel_names"]),
        "window_s": window / sfreq,
        "step_s": stride / sfreq,
    }
    dataset.metadata["channel_names"] = list(BandPowerSeriesTransformer.FREQ_BANDS)
    dataset.X = out


def prepare_recording_cases(arrays, metas, prep, labels, rules, progress=None):
    """Build cases and apply the configured case-level transforms."""
    dataset, notes = build_dataset(
        arrays, [dict(m) for m in metas], labels, rules, prep
    )
    chosen = choose_channels(dataset, prep)
    band_power(dataset, prep, progress)
    return dataset, notes + ([chosen] if chosen else [])


def select_prepared_channels(dataset, excluded):
    """Select fixed channels from supplied cases without applying raw transforms."""
    names = dataset.metadata["channel_names"]
    excluded = set(excluded or [])
    keep = [i for i, name in enumerate(names) if name not in excluded]
    if len(keep) < 2:
        raise ProblemError("Select at least two channels.")
    if len(keep) == len(names):
        return dataset
    metadata = {**dataset.metadata, "channel_names": [names[i] for i in keep]}
    return replace(dataset, X=dataset.X[:, keep], metadata=metadata)
