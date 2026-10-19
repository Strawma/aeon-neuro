"""Inspect competing EEG metadata and require explicit choices before use."""

import warnings
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

from mne import create_info
from mne.io import BaseRaw

from aeon_neuro.utils.metadata import prepare_eeg_metadata


class MetadataConflictError(ValueError):
    """Reject unresolved metadata with a JSON-compatible ``report`` attribute."""

    def __init__(self, report):
        """Retain all findings and describe the unresolved fields for scripts."""
        self.report = report
        details = []
        for finding in report["findings"]:
            if finding["blocking"]:
                label = finding.get("channel") or finding["field"]
                values = ", ".join(
                    f"{item['value']!r} ({item['source']})"
                    for item in finding["candidates"]
                )
                details.append(f"{label}: {values}")
        super().__init__(
            "Unresolved EEG metadata; supply sourced corrections for roles/frequency "
            "or investigate sampling-rate disagreements: " + "; ".join(details)
        )


class MetadataResolutionWarning(UserWarning):
    """Identify a metadata decision that is also retained in the returned report."""


def inspect_eeg_metadata(raw, *, metadata_sources=None):
    """Compare caller-matched metadata without changing or loading samples.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Imported recording. Its metadata may contain reader defaults, not facts.
    metadata_sources : sequence of dict or None, default=None
        Each entry requires a unique nonempty ``source`` string other than
        ``"raw"`` and may provide ``line_freq`` and ``sfreq`` in Hz and
        ``channel_types`` mapping existing names to MNE types. Type names are
        case-normalised; BIDS-specific types such as TRIG require caller mapping.
        None and "n/a" mean unknown. Resolve source applicability, channel
        correspondence and BIDS inheritance before passing these entries.
        No file discovery, source authentication or precedence is inferred.

    Returns
    -------
    report : dict
        JSON-compatible findings with field/channel, candidates and origins,
        status, blocking flag and optional recommendation. ``ready`` means no
        assessed finding blocks reconciliation, not complete BIDS validity.
        Missing mains frequency permits a recommended fill only when every known
        candidate agrees and none is invalid. Unknown mains frequency may remain.

    Raises
    ------
    TypeError
        If raw, source containers or candidate types have an invalid structure.
    ValueError
        If a source identifier, key or channel name is unsupported/ambiguous.

    Notes
    -----
    Invalid values and contradictory known values are findings, not exceptions
    during inspection. All imported channel roles are listed for review.
    Sampling-rate disagreements cannot be repaired by changing descriptive
    metadata. No sample analysis establishes acquisition truth here.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw must be an MNE Raw recording.")
    sources = _validate_sources(raw, metadata_sources)
    fields = [("line_freq", None, raw.info["line_freq"])]
    fields.extend(
        ("channel_type", name, kind)
        for name, kind in zip(raw.ch_names, raw.get_channel_types())
    )
    fields.append(("sfreq", None, raw.info["sfreq"]))
    findings = []
    for field, channel, original in fields:
        candidates = [_candidate(field, original, "raw")]
        for source in sources:
            values = source.get("channel_types", {}) if channel is not None else source
            key = channel if channel is not None else field
            if key in values:
                candidates.append(_candidate(field, values[key], source["source"]))
        known = [item["value"] for item in candidates if item["status"] == "known"]
        invalid = any(item["status"] == "invalid" for item in candidates)
        different = len(set(known)) > 1
        if invalid:
            code = "invalid"
        elif different:
            code = "conflict"
        elif not known:
            code = "unknown"
        elif candidates[0]["status"] == "unknown":
            code = "fill_missing"
        else:
            code = "consistent"
        findings.append(
            {
                "field": field,
                "channel": channel,
                "code": code,
                "candidates": candidates,
                "blocking": invalid or different,
                "recommendation": known[0] if code == "fill_missing" else None,
                "resolution": None,
            }
        )
    return {
        "assessed_fields": ["line_freq", "channel_type", "sfreq"],
        "findings": findings,
        "decisions": [],
        "corrections": [],
        "ready": not any(item["blocking"] for item in findings),
    }


def reconcile_eeg_metadata(
    raw,
    *,
    metadata_sources=None,
    channel_types=None,
    line_freq=None,
    correction_source=None,
):
    """Resolve assessed metadata on a copy, raising for remaining blockers.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Imported recording; samples, annotations and input state remain intact.
    metadata_sources : sequence of dict or None, default=None
        Matched candidates as described by ``inspect_eeg_metadata``.
    channel_types : dict or None, default=None
        Explicit final channel-role choices, including choices retaining the
        imported role. Values use MNE channel types.
    line_freq : float or None, default=None
        Explicit finite positive final mains frequency, or None for no choice.
        A single agreed candidate fills unknown frequency automatically; this
        emits a warning and records its origin. None cannot discard known values.
    correction_source : str or None, default=None
        Nonempty evidence reference required for explicit choices. The caller
        establishes its truth and applicability; the library records it.

    Returns
    -------
    prepared : mne.io.BaseRaw
        Copy with resolved roles/frequency and unchanged samples/annotations.
    report : dict
        Inspection findings, final decisions and actual before/after corrections.
        Decisions are retained even when the selected value was already present.
        Persist the report separately; no automatic BIDS provenance file is made.

    Raises
    ------
    MetadataConflictError
        If assessed blockers remain. Its ``report`` contains all findings.
    TypeError
        If sources or corrections have an invalid structure.
    ValueError
        If explicit corrections are invalid or lack a source.

    Warns
    -----
    MetadataResolutionWarning
        For automatic missing-value fills and explicitly resolved conflicts.

    Notes
    -----
    No prompts, resampling, unit rescaling or source-priority guesses occur.
    Sampling-rate mismatches remain blocking until their underlying source or
    recording is corrected. Other acquisition fields and full validation remain
    outside this bounded check. Use this helper before writing, or opt in via
    ``convert_eeg_to_bids`` metadata arguments.
    """
    report = inspect_eeg_metadata(raw, metadata_sources=metadata_sources)
    # Validate explicit choices using the existing copy-preserving contract.
    prepared, changes = prepare_eeg_metadata(
        raw,
        channel_types=channel_types,
        line_freq=line_freq,
        correction_source=correction_source,
    )
    automatic_frequency = None
    for finding in report["findings"]:
        field, channel = finding["field"], finding["channel"]
        explicit = (field == "line_freq" and line_freq is not None) or (
            field == "channel_type" and channel in (channel_types or {})
        )
        if explicit:
            value = float(line_freq) if field == "line_freq" else channel_types[channel]
            decision = {
                "field": field,
                "channel": channel,
                "value": value,
                "source": correction_source.strip(),
                "method": "explicit",
                "reason": "Caller supplied a final choice and evidence reference.",
                "applied": False,
            }
        elif finding["code"] == "fill_missing" and field == "line_freq":
            automatic_frequency = finding["recommendation"]
            origins = [
                item["source"]
                for item in finding["candidates"]
                if item["status"] == "known"
            ]
            decision = {
                "field": field,
                "channel": channel,
                "value": automatic_frequency,
                "source": "; ".join(origins),
                "method": "fill_missing",
                "reason": (
                    "All known supplied candidates agree; imported value is unknown."
                ),
                "applied": False,
            }
        else:
            continue
        finding["resolution"] = decision
        finding["blocking"] = False
        report["decisions"].append(decision)
    report["ready"] = not any(item["blocking"] for item in report["findings"])
    if not report["ready"]:
        raise MetadataConflictError(report)
    if automatic_frequency is not None:
        # This only fills absent metadata; conflicting known values need a choice.
        prepared.info["line_freq"] = automatic_frequency
        decision = next(d for d in report["decisions"] if d["field"] == "line_freq")
        changes["corrections"].append(
            {
                "field": "line_freq",
                "before": None,
                "after": automatic_frequency,
                "source": decision["source"],
            }
        )
    report["corrections"] = changes["corrections"]
    for decision in report["decisions"]:
        decision["applied"] = True
    for finding in report["findings"]:
        decision = finding["resolution"]
        if decision and finding["code"] in {"fill_missing", "conflict", "invalid"}:
            label = finding["channel"] or finding["field"]
            warnings.warn(
                f"Resolved {label} to {decision['value']!r} from {decision['source']} "
                f"({decision['method']}); retain the metadata decision report.",
                MetadataResolutionWarning,
                stacklevel=2,
            )
    return prepared, report


def _validate_sources(raw, sources):
    """Reject ambiguous source records instead of silently ignoring their keys."""
    if sources is None:
        return []
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("metadata_sources must be a sequence of mappings.")
    seen = {"raw"}
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError("Each metadata source must be a mapping.")
        name = source.get("source")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Each source requires a nonempty, trimmed source name.")
        if name in seen:
            raise ValueError("Metadata source names must be unique; 'raw' is reserved.")
        seen.add(name)
        if set(source) - {"source", "line_freq", "sfreq", "channel_types"}:
            raise ValueError("Unsupported metadata source keys.")
        roles = source.get("channel_types", {})
        if not isinstance(roles, Mapping):
            raise TypeError(
                "Source channel_types must map existing channel names to types."
            )
        if set(roles) - set(raw.ch_names):
            raise ValueError("Metadata source contains channel names absent from raw.")
    return sources


def _candidate(field, value, source):
    """Normalise representational differences without inferring acquisition facts."""
    if value is None or (isinstance(value, str) and value == "n/a"):
        return {"source": source, "value": None, "status": "unknown"}
    if field == "channel_type":
        if not isinstance(value, str):
            raise TypeError("Channel type candidates must be strings or None.")
        value = value.lower()
        try:
            create_info(["candidate"], 1, value)
        except (KeyError, ValueError):
            valid = False
        else:
            valid = True
    else:
        if not isinstance(value, (Real, str)):
            raise TypeError("Frequency candidates must be numbers, strings or None.")
        valid = not isinstance(value, bool) and isinstance(value, Real)
        valid = valid and isfinite(value) and value > 0
        # Invalid NaN/Inf candidates must remain visible in strict JSON reports.
        if isinstance(value, Real) and not isinstance(value, bool):
            value = float(value) if isfinite(value) else str(value)
    return {"source": source, "value": value, "status": "known" if valid else "invalid"}
