.. _utils_ref:

Utility functions
=================

BIDS conversion
---------------

``convert_eeg_to_bids`` accepts a native BrainVision filename or an existing MNE
Raw recording and returns the written ``BIDSPath``. Supplied metadata is assumed
correct; MNE-BIDS handles writing and its existing validation. No analysis
processing or metadata repair is added. Native-path reading does not import
existing BIDS sidecars.

.. autofunction:: aeon_neuro.utils.bids.convert_eeg_to_bids

BrainVision ingestion
---------------------

Use ``load_brainvision_to_mne`` to retain channel metadata, sampling information
and annotations. ``load_brainvision_to_numpy`` retains the existing array-only
interface and optional EEG channel selection.

.. autofunction:: aeon_neuro.utils.data_io.load_brainvision_to_mne

.. autofunction:: aeon_neuro.utils.data_io.load_brainvision_to_numpy

Auxiliary information
---------------------

``load_auxiliary_info`` returns ``None`` with a warning when its JSON file cannot
be read or decoded. The directory prefix must include its trailing separator.

.. autofunction:: aeon_neuro.utils.data_io.load_auxiliary_info

Metadata preparation
---------------------

``prepare_eeg_metadata`` reports unknown/invalid mains frequency and applies
explicit channel-role/frequency corrections on a Raw copy. Corrections require
a supplied evidence reference and produce a JSON-compatible before/after report.
Samples and annotations remain unchanged; declared unit changes are recorded,
without sample rescaling or calibration repair. The report assesses mains
frequency only and does not establish complete BIDS metadata validity.

.. autofunction:: aeon_neuro.utils.metadata.prepare_eeg_metadata

Metadata conflicts
------------------

``inspect_eeg_metadata`` compares imported metadata with explicitly supplied,
caller-matched sources. It returns all assessed findings without changing or
loading samples. The bounded schema covers mains frequency, channel roles and
sampling-rate agreement. ``None`` and ``"n/a"`` remain unknown; imported roles
may be reader defaults. Source discovery, identity checks and BIDS inheritance
must be handled before supplying candidates. There is no generic source priority.

``reconcile_eeg_metadata`` returns a Raw copy and a JSON-compatible decision
report. A unique, valid mains-frequency candidate fills unknown metadata with a
``MetadataResolutionWarning``. Contradictory known values require explicit final
choices with ``correction_source``. Invalid candidates remain visible even when
an explicit choice resolves the finding. Decisions are recorded even when they
retain the imported value. Sampling-rate disagreements cannot be overridden:
investigate the source or recording, rather than editing its rate metadata.

``MetadataConflictError`` subclasses ``ValueError`` and carries ``report`` for
frontend or batch handling. Unknown mains frequency alone does not block.
Inspection never prompts for input; a caller collects choices and retries.
Warnings also accompany explicitly resolved conflicts. Reports retain candidate
origins and decisions, without authenticating their scientific correctness.

``convert_eeg_to_bids`` optionally accepts ``metadata_sources`` and
``metadata_corrections`` to reconcile before calling the writer. Its default
behaviour and BIDSPath return are unchanged. Use the helper directly to retain
the decision report; no provenance file is automatically written into BIDS.
Neither conversion nor reconciliation changes physical samples or resamples.
The remaining acquisition metadata and full BIDS validation are separate checks.

.. autofunction:: aeon_neuro.utils.metadata_conflicts.inspect_eeg_metadata

.. autofunction:: aeon_neuro.utils.metadata_conflicts.reconcile_eeg_metadata

.. autoexception:: aeon_neuro.utils.metadata_conflicts.MetadataConflictError

.. autoclass:: aeon_neuro.utils.metadata_conflicts.MetadataResolutionWarning
