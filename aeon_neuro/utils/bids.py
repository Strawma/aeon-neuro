"""Convert native EEG recordings and MNE Raw objects to BIDS."""

from os import PathLike
from pathlib import Path

from mne.io import BaseRaw
from mne_bids import write_raw_bids

from aeon_neuro.utils.data_io import load_brainvision_to_mne
from aeon_neuro.utils.metadata_conflicts import reconcile_eeg_metadata


def convert_eeg_to_bids(
    source,
    bids_path,
    *,
    format="auto",
    allow_preload=False,
    overwrite=False,
    metadata_sources=None,
    metadata_corrections=None,
):
    """Write a BrainVision recording or an MNE Raw object to BIDS.

    Parameters
    ----------
    source : str, os.PathLike or mne.io.BaseRaw
        BrainVision .vhdr path with its linked signal and marker files, or an
        existing MNE Raw recording. Other filename formats are not yet supported.
    bids_path : mne_bids.BIDSPath
        Destination with root, subject and task set. Supply session, run and
        datatype where needed; identifiers are not inferred from the source path.
    format : str, default="auto"
        Output encoding passed to ``mne_bids.write_raw_bids``. The default copies
        supported source files where possible. Preloaded recordings require an
        explicit encoding, such as ``"BrainVision"``, and its export backend.
    allow_preload : bool, default=False
        Permit writing preloaded Raw objects, including RawArray, when True.
        Follow MNE-BIDS requirements for the selected output encoding. BrainVision
        export requires the optional ``pybv`` package; unchanged copying does not.
    overwrite : bool, default=False
        Whether MNE-BIDS may overwrite an existing output recording.
    metadata_sources : sequence of dict or None, default=None
        Caller-matched metadata candidates for ``inspect_eeg_metadata``. Supplying
        this argument, including an empty list, enables reconciliation before
        writing. Unresolved assessed conflicts raise ``MetadataConflictError``.
        Existing sidecars and BIDS inheritance are not discovered automatically.
    metadata_corrections : dict or None, default=None
        Explicit final choices passed to ``reconcile_eeg_metadata``: only
        ``channel_types``, ``line_freq`` and ``correction_source`` are accepted.
        Supplying a mapping also enables reconciliation. Use the reconciliation
        helper directly first when the returned decision report must be saved;
        this converter continues to return only the written BIDSPath.

    Returns
    -------
    written : mne_bids.BIDSPath
        Actual output path returned by MNE-BIDS, including the recording extension.

    Raises
    ------
    TypeError
        If source is neither a filename nor an MNE Raw object.
    ValueError
        If a filename does not have the .vhdr extension, or MNE-BIDS rejects the
        recording or output configuration.
    MetadataConflictError
        If optional metadata reconciliation finds unresolved blockers. The
        exception carries a structured report and occurs before writer calls.

    Notes
    -----
    Without metadata arguments, supplied metadata is assumed correct. No filtering,
    channel selection, re-referencing or epoching is performed. Annotations are written
    using MNE-BIDS conventions; event codes may be reassigned. Existing BIDS
    sidecars are not imported when reading a native BrainVision path.

    MNE and MNE-BIDS file, metadata and export errors propagate unchanged.
    Successful writing alone does not establish full BIDS validation.
    """
    if isinstance(source, (str, PathLike)):
        if Path(source).suffix.lower() != ".vhdr":
            raise ValueError(
                "Only BrainVision .vhdr filenames are supported; "
                "pass an MNE Raw object for other input formats."
            )
        # Keep samples on disk so MNE-BIDS can copy the original recording.
        raw = load_brainvision_to_mne(source)
    elif isinstance(source, BaseRaw):
        raw = source
    else:
        raise TypeError("source must be a BrainVision filename or an MNE Raw object.")

    if metadata_sources is not None or metadata_corrections is not None:
        if metadata_corrections is None:
            metadata_corrections = {}
        if not isinstance(metadata_corrections, dict) or set(metadata_corrections) - {
            "channel_types",
            "line_freq",
            "correction_source",
        }:
            raise ValueError(
                "metadata_corrections must contain only explicit choices/source."
            )
        # Reject unresolved choices before MNE-BIDS can create dataset files.
        raw, _ = reconcile_eeg_metadata(
            raw, metadata_sources=metadata_sources, **metadata_corrections
        )

    # Delegate remaining checks and preloaded export policy to MNE-BIDS.
    return write_raw_bids(
        raw,
        bids_path,
        format=format,
        allow_preload=allow_preload,
        overwrite=overwrite,
    )
