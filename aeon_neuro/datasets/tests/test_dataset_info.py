"""Check in-memory metadata ownership, paired selection and the case handoff."""

import json
from copy import deepcopy

import edfio
import mne
import numpy as np
import pytest

from aeon_neuro import __version__
from aeon_neuro.datasets import (
    EEGDatasetInfo,
    RecordingInfo,
    inspect_bids_eeg,
    load_bids_dataset,
    load_classification_problem,
    make_classification_problem,
)
from aeon_neuro.datasets import _dataset_info as module


@pytest.fixture
def source(tmp_path, request):
    """Use distinguishable recordings, explicit source roles and a larger cohort."""
    mode = getattr(request, "param", "equal")
    (tmp_path / "dataset_description.json").write_text(
        '{"Name":"Generated study"}', encoding="utf-8"
    )
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-01\tcontrol\nsub-02\tcase\nsub-03\tunselected\n",
        encoding="utf-8",
    )
    (tmp_path / "participants.json").write_text(
        '{"group":{"Description":"Source group"}}', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "Generated metadata example\n", encoding="utf-8"
    )
    paths = []
    for index, subject in enumerate(("01", "02")):
        directory = tmp_path / f"sub-{subject}" / "eeg"
        directory.mkdir(parents=True)
        stem = f"sub-{subject}_task-imagery"
        rate = 64 if mode == "rate" and index else 32
        length = 128 if mode == "length" and index else 256
        names = ["C4", "C3"] if mode == "order" and index else ["C3", "C4"]
        values = np.random.default_rng(index).normal(size=(2, length))
        path = directory / f"{stem}_eeg.bdf"
        edfio.Bdf(
            [
                edfio.BdfSignal(row, rate, label=name, physical_dimension="uV")
                for row, name in zip(values, names, strict=True)
            ]
        ).write(path)
        (directory / f"{stem}_eeg.json").write_text(
            json.dumps({"SamplingFrequency": rate, "EEGReference": "Source reference"}),
            encoding="utf-8",
        )
        (directory / f"{stem}_channels.tsv").write_text(
            "name\ttype\tunits\tstatus\n"
            + "".join(
                f"{name}\tEEG\tuV\t{status}\n"
                for name, status in zip(names, ("good", "bad"), strict=True)
            ),
            encoding="utf-8",
        )
        (directory / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n1\t1\tleft\n2\t1\tright\n", encoding="utf-8"
        )
        paths.append(path)
    return tmp_path, paths


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_load_preserves_exact_samples_metadata_and_source_population(source, newline):
    """Loading preserves source units, identity and the complete participant table."""
    root, paths = source
    # Exact source text includes its line endings on every host platform.
    readme = f"Generated metadata example{newline}"
    (root / "README.md").write_bytes(readme.encode("utf-8"))
    X, info = load_bids_dataset(root)
    assert X.shape == (2, 2, 256)
    assert isinstance(info, EEGDatasetInfo) and len(info) == 2
    assert isinstance(info.recordings[0], RecordingInfo)
    assert info.source_ids == tuple(path.relative_to(root).as_posix() for path in paths)
    assert info.participant_ids == ("sub-01", "sub-02")
    assert len(info.participants["rows"]) == 3
    assert info.description["Name"] == "Generated study"
    assert info.readme == readme
    assert info.participants_json["group"]["Description"] == "Source group"
    for index, path in enumerate(paths):
        recording = info.recordings[index]
        assert recording.sfreq == 32
        assert recording.channel_names == ("C3", "C4")
        assert recording.entities["subject"] == f"0{index+1}"
        assert recording.eeg_json["EEGReference"] == "Source reference"
        assert recording.channels[0]["units"] == "uV"
        assert recording.bad_channels() == ("C4",)
        assert recording.channels_of_type("EEG") == ("C3", "C4")
        assert len(recording.events) == 2
        with mne.io.read_raw_bdf(path, preload=False, verbose=False) as raw:
            np.testing.assert_array_equal(X[index], raw.get_data())
    info.validate_layout(X)


def test_views_and_selection_do_not_share_mutable_metadata_or_signals(source):
    """Separate experiments must not mutate each other through a shared view."""
    root, _ = source
    X, info = load_bids_dataset(root)
    selected, subset = info.select(X, [1, 0])
    np.testing.assert_array_equal(selected, X[[1, 0]])
    assert subset.source_ids == info.source_ids[::-1]
    assert subset.participant_ids == info.participant_ids[::-1]
    assert [row["recording_id"] for row in subset.events] == [
        info.source_ids[1]
    ] * 2 + [info.source_ids[0]] * 2
    selected[0, 0, 0] += 1
    assert selected[0, 0, 0] != X[1, 0, 0]
    fields = subset.metadata
    fields[0]["channel_names"].reverse()
    events = subset.recordings[0].events
    events[0]["trial_type"] = "changed"
    participants = subset.participants
    participants["rows"][0]["group"] = "changed"
    assert (
        info.recordings[1].channel_names
        == subset.recordings[0].channel_names
        == ("C3", "C4")
    )
    assert subset.recordings[0].events[0]["trial_type"] == "left"
    assert info.participants == subset.participants
    assert (
        EEGDatasetInfo(json.loads(json.dumps(subset.to_dict()))).to_dict()
        == subset.to_dict()
    )


@pytest.mark.parametrize(
    "selector", [[False, True], np.array([1]), slice(None, None, -1), []]
)
def test_masks_slices_and_empty_selections_keep_recording_axis(source, selector):
    """Empty selections retain the same collection contract as nonempty subsets."""
    X, info = load_bids_dataset(source[0])
    expected = X[selector] if not isinstance(selector, list) or selector else X[:0]
    selected, subset = info.select(X, selector)
    np.testing.assert_array_equal(selected, expected)
    assert selected.ndim == 3 and len(subset) == len(selected)
    subset.validate_layout(selected)
    if not len(subset):
        assert subset.metadata == [] and subset.events == []
        assert subset.dataset["selected_participants"] == []
        assert len(subset.participants["rows"]) == 3
        empty, again = subset.select(selected, [])
        assert empty.shape == (0, 2, 256) and len(again) == 0


@pytest.mark.parametrize(
    "selector", [[0, 0], [-1], [2], [0.0], [True], [True, 0], [[0]], "01", 0]
)
def test_invalid_selectors_fail_without_modifying_inputs(source, selector):
    """Reject ambiguous or duplicate source membership before creating a subset."""
    X, info = load_bids_dataset(source[0])
    before, report = X.copy(), info.to_dict()
    with pytest.raises((ValueError, TypeError)):
        info.select(X, selector)
    np.testing.assert_array_equal(X, before)
    assert info.to_dict() == report


@pytest.mark.parametrize("source", ["length", "rate", "order"], indirect=True)
def test_incompatible_recordings_stay_as_list_and_copy_on_selection(source):
    """Different layouts or rates must not be silently harmonized by stacking."""
    X, info = load_bids_dataset(source[0])
    assert isinstance(X, list)
    selected, subset = info.select(X, [1, 0])
    assert isinstance(selected, list)
    for i in range(2):
        np.testing.assert_array_equal(selected[i], X[1 - i])
        assert not np.shares_memory(selected[i], X[1 - i])
    assert subset.source_ids == info.source_ids[::-1]
    empty, empty_info = info.select(X, [])
    assert empty == [] and len(empty_info) == 0


@pytest.mark.parametrize("fault", ["count", "channels", "samples", "dtype"])
def test_layout_mismatches_block_selection(source, fault):
    """Changed axes cannot reuse the original source-layout description."""
    X, info = load_bids_dataset(source[0])
    bad = {
        "count": X[:1],
        "channels": X[:, :1],
        "samples": X[:, :, :10],
        "dtype": X.astype(bool),
    }[fault]
    with pytest.raises(ValueError):
        info.select(bad, [0])


def test_missing_channel_status_is_not_reported_as_clean(source):
    """Unknown status does not establish that every channel is good."""
    root, paths = source
    for path in paths:
        path.with_name(path.name.replace("_eeg.bdf", "_channels.tsv")).write_text(
            "name\ttype\tunits\nC3\tEEG\tuV\nC4\tEEG\tuV\n", encoding="utf-8"
        )
    _, info = load_bids_dataset(root)
    with pytest.raises(ValueError, match="status"):
        info.recordings[0].bad_channels()


def test_loader_closes_every_owned_handle_when_a_later_read_fails(source, monkeypatch):
    """A failed recording must close both earlier and current source handles."""
    original = module.open_bids_eeg_recording
    closed = []

    def watched(root, descriptor):
        raw = original(root, descriptor)
        close = raw.close

        def close_and_record():
            closed.append(descriptor["recording_id"])
            close()

        raw.close = close_and_record
        if descriptor["subject"] == "02":

            def broken():
                raise RuntimeError("sample read failed")

            raw.get_data = broken
        return raw

    monkeypatch.setattr(module, "open_bids_eeg_recording", watched)
    with pytest.raises(RuntimeError, match="sample read failed"):
        load_bids_dataset(source[0])
    assert len(closed) == 2


def test_blocked_report_and_changed_metadata_do_not_load_partial_data(
    source, monkeypatch
):
    """Blockers and stale snapshots must prevent a usable partial load."""
    root, _ = source
    report = inspect_bids_eeg(root)
    blocked = deepcopy(report)
    blocked["ready"] = False
    with pytest.raises(ValueError, match="ready"):
        EEGDatasetInfo(blocked)
    (root / "participants.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "inspect_bids_eeg", lambda *args, **kwargs: report)
    with pytest.raises(ValueError, match="Inspected source changed"):
        load_bids_dataset(root)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "source_order",
        "event_source",
        "scope",
        "event_sample",
        "event_bounds",
        "duplicate_event",
    ],
)
def test_info_rejects_misaligned_inspection_reports(source, fault):
    """Saved facts must identify selected observations and their coordinates."""
    report = inspect_bids_eeg(source[0])
    if fault == "version":
        report["version"] = 1
    elif fault == "source_order":
        report["source_ids"].reverse()
    elif fault == "event_source":
        report["events"][0]["recording_id"] = "unknown"
    elif fault == "scope":
        report["dataset"]["selected_participants"] = []
    elif fault == "event_sample":
        report["events"][0]["bids_onset_sample"] += 1
    elif fault == "event_bounds":
        report["events"][0]["bids_onset_sample"] = 100000
    else:
        report["events"].append(deepcopy(report["events"][0]))
    with pytest.raises(ValueError):
        EEGDatasetInfo(report)


def test_selected_info_feeds_the_existing_problem_loader(source):
    """Reuse explicit task bindings while retaining a reordered source join."""
    X, info = load_bids_dataset(source[0])
    X, info = info.select(X, [1, 0])
    problem = {
        "format_version": 1,
        "id": "imagery",
        "version": "1",
        "question": "Classify declared imagery",
        "class_labels": ["left", "right"],
        "channels": ["C4", "C3"],
        "segmentation": {"mode": "event_epochs", "tmin": 0.5, "tmax": 1.5},
    }
    binding = {
        "format_version": 1,
        "problem": {"id": "imagery", "version": "1"},
        "target": {
            "source": "event",
            "column": "trial_type",
            "mapping": {"left": "left", "right": "right"},
        },
        "group": {"source": "recording", "column": "participant_id"},
        "events": {"onset": "onset", "id": "bids_event_id"},
    }
    cases = load_classification_problem(
        X, info.metadata, problem, binding, events=info.events
    )
    assert cases.X.shape == (4, 2, 32)
    assert cases.groups.tolist() == ["sub-02", "sub-02", "sub-01", "sub-01"]
    assert cases.y.tolist() == ["left", "right", "left", "right"]
    np.testing.assert_array_equal(cases.X[0], X[0, [1, 0], 48:80])


def test_constructor_retains_detached_facts_without_signal_access(source, monkeypatch):
    """Metadata construction is cheap and detached from later report changes."""
    report = inspect_bids_eeg(source[0])

    def unexpected(*args, **kwargs):
        raise AssertionError("Signal access while constructing metadata view")

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected)
    info = EEGDatasetInfo(report)
    report["recordings"][0]["eeg_metadata"]["EEGReference"] = "changed"
    report["events"][0]["trial_type"] = "changed"
    assert info.recordings[0].eeg_json["EEGReference"] == "Source reference"
    assert info.events[0]["trial_type"] == "left"


def test_single_recording_selection_and_missing_tables_remain_explicit(source):
    """Missing sidecars stay distinguishable from empty valid tables."""
    root, paths = source
    paths[0].with_name(paths[0].name.replace("_eeg.bdf", "_channels.tsv")).unlink()
    X, info = load_bids_dataset(root, subjects=["01"])
    assert X.shape == (1, 2, 256)
    assert info.recordings[0].channels is None
    assert info.recordings[0].electrodes is None
    assert info.recordings[0].coordsystem is None
    assert info.recordings[0].scan is None
    assert info.dataset["selected_participants"] == ["sub-01"]
    assert len(info.participants["rows"]) == 3
    with pytest.raises(ValueError, match="type"):
        info.recordings[0].channels_of_type("EEG")
    with pytest.raises(ValueError, match="status"):
        info.recordings[0].bad_channels()


def test_history_records_are_detached_and_preserve_original_source_facts(source):
    """Reported processing must not rewrite acquisition facts or mutate parents."""
    report = inspect_bids_eeg(source[0])
    info = EEGDatasetInfo(report)
    assert info.history == []
    assert info.to_dict() == report
    parameters = {"factor": 1e6, "channels": ["C3", "C4"], "units": "uV"}
    processed = info.with_history(
        "scale",
        parameters,
        implementation="numpy.multiply",
        version=np.__version__,
        fit_scope="none",
    )
    parameters["channels"].reverse()
    assert info.history == [] and info.to_dict() == report
    assert processed.history == [
        {
            "version": 1,
            "step": "scale",
            "parameters": {"factor": 1e6, "channels": ["C3", "C4"], "units": "uV"},
            "implementation": "numpy.multiply",
            "implementation_version": np.__version__,
            "fit_scope": "none",
            "source_ids": list(info.source_ids),
        }
    ]
    saved = processed.to_dict()
    assert saved.pop("history") == processed.history
    assert saved == report
    exposed = processed.history
    exposed[0]["parameters"]["channels"].reverse()
    assert processed.history[0]["parameters"]["channels"] == ["C3", "C4"]
    restored = EEGDatasetInfo(json.loads(json.dumps(processed.to_dict())))
    assert restored.to_dict() == processed.to_dict()
    saved = processed.to_dict()
    from_saved = EEGDatasetInfo(saved)
    saved["history"][0]["parameters"]["factor"] = 0
    assert from_saved.to_dict() == processed.to_dict()


def test_history_preserves_application_context_across_selections(source):
    """Subsetting must retain the earlier, possibly broader, fitting context."""
    X, info = load_bids_dataset(source[0])
    recorded = info.with_history(
        "external step",
        {"nested": [None, True, 2, 1.5, {"value": "text"}]},
        implementation="example.transform",
        version="1.2",
    )
    assert recorded.history[0]["fit_scope"] == "unspecified"
    selected, subset = recorded.select(X, [1])
    assert subset.history == recorded.history
    assert subset.history[0]["source_ids"] == list(info.source_ids)
    following = subset.with_history(
        "reported filter",
        {"low": 1.0, "high": None},
        implementation="example.filter",
        version="2",
        fit_scope="one recording",
    )
    assert len(recorded.history) == len(subset.history) == 1
    assert following.history[:-1] == recorded.history
    assert following.history[-1]["source_ids"] == [info.source_ids[1]]
    restored = EEGDatasetInfo(json.loads(json.dumps(following.to_dict())))
    restored.validate_layout(selected)
    empty, empty_info = following.select(selected, [])
    assert empty.shape[0] == 0 and empty_info.history == following.history
    assert EEGDatasetInfo(empty_info.to_dict()).history == following.history
    with pytest.raises(ValueError, match="empty"):
        empty_info.with_history("step", {}, implementation="example", version="1")


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        [],
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "integer key"},
        {"value": (1, 2)},
        {"value": np.int64(1)},
        {"value": np.array([1])},
        {"value": object()},
    ],
)
def test_history_requires_portable_json_parameters(source, parameters):
    """Saved parameters must not silently change types or accept nonfinite values."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0]))
    with pytest.raises(ValueError, match="parameters"):
        info.with_history("step", parameters, implementation="example", version="1")
    assert info.history == []


@pytest.mark.parametrize("field", ["step", "implementation", "version", "fit_scope"])
@pytest.mark.parametrize("value", ["", "  ", None, 1])
def test_history_requires_named_operation_and_implementation(source, field, value):
    """Operation identity and fitting scope remain explicit text."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0]))
    arguments = dict(step="step", parameters={}, implementation="example", version="1")
    arguments[field] = value
    with pytest.raises(ValueError):
        info.with_history(**arguments)
    assert info.history == []


@pytest.mark.parametrize(
    "fault",
    [
        "not_list",
        "record",
        "missing",
        "extra",
        "version",
        "boolean_version",
        "step",
        "implementation",
        "implementation_version",
        "fit_scope",
        "parameters",
        "sources",
        "duplicate_source",
        "empty_source",
    ],
)
def test_saved_history_records_are_validated(source, fault):
    """Restoring a metadata view must not silently trust malformed provenance."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0])).with_history(
        "step", {}, implementation="example", version="1"
    )
    report = info.to_dict()
    record = report["history"][0]
    if fault == "not_list":
        report["history"] = None
    elif fault == "record":
        report["history"] = ["step"]
    elif fault == "missing":
        record.pop("parameters")
    elif fault == "extra":
        record["unknown"] = "value"
    elif fault == "version":
        record["version"] = 2
    elif fault == "boolean_version":
        record["version"] = True
    elif fault in ("step", "implementation", "implementation_version", "fit_scope"):
        record[fault] = ""
    elif fault == "parameters":
        record["parameters"] = {"value": float("nan")}
    elif fault == "sources":
        record["source_ids"] = "not a list"
    elif fault == "duplicate_source":
        record["source_ids"] *= 2
    else:
        record["source_ids"] = []
    with pytest.raises(ValueError, match="history"):
        EEGDatasetInfo(report)


def test_history_records_do_not_claim_to_update_layout_or_observe_arrays(source):
    """A crop note cannot make the original metadata describe a shorter array."""
    X, info = load_bids_dataset(source[0])
    noted = info.with_history(
        "reported crop", {"stop_sample": 128}, implementation="external", version="1"
    )
    assert noted.metadata == info.metadata and noted.events == info.events
    with pytest.raises(ValueError, match="n_samples"):
        noted.validate_layout(X[:, :, :128])


def test_cyclic_history_parameters_fail_without_mutating_parent(source):
    """Recursive objects cannot be saved as portable provenance."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0]))
    parameters = {}
    parameters["cycle"] = parameters
    with pytest.raises(ValueError, match="parameters"):
        info.with_history("step", parameters, implementation="example", version="1")
    assert info.history == []


def _info_problem_documents(mode):
    problem = {
        "format_version": 1,
        "id": "info-problem",
        "version": "1",
        "question": "Classify the explicitly bound source label",
        "class_labels": ["control", "case"],
        "channels": ["C4", "C3"],
        "segmentation": (
            {"mode": mode, "tmin": 0.5, "tmax": 1.5}
            if mode == "event_epochs"
            else {"mode": mode, "window_size": 1.0}
        ),
    }
    binding = {
        "format_version": 1,
        "problem": {"id": "info-problem", "version": "1"},
        "target": (
            {
                "source": "event",
                "column": "trial_type",
                "mapping": {"left": "control", "right": "case"},
            }
            if mode == "event_epochs"
            else {
                "source": "recording",
                "column": "participant.group",
                "mapping": {"control": "control", "case": "case"},
            }
        ),
        "group": {"source": "recording", "column": "participant_id"},
        "case_fields": {
            "participant_id": {"source": "recording", "column": "participant_id"},
            "source_task": {"source": "recording", "column": "task"},
        },
    }
    if mode == "event_epochs":
        binding["events"] = {"onset": "onset", "id": "bids_event_id"}
    return problem, binding


@pytest.mark.parametrize("mode", ["event_epochs", "fixed_windows"])
def test_object_builder_matches_existing_cases_and_keeps_unified_provenance(
    source, mode
):
    """The object route must preserve case semantics, source joins and history."""
    X, info = load_bids_dataset(source[0])
    info = info.with_history(
        "metadata review",
        {"columns": ["trial_type", "participant.group"]},
        implementation="example.review",
        version="1",
        fit_scope="none",
    )
    X, info = info.select(X, [1, 0])
    problem, binding = _info_problem_documents(mode)
    expected = load_classification_problem(
        X,
        info.metadata,
        problem,
        binding,
        events=info.events if mode == "event_epochs" else None,
    )
    result = make_classification_problem(X, info, problem, binding)
    np.testing.assert_array_equal(result.X, expected.X)
    np.testing.assert_array_equal(result.y, expected.y)
    np.testing.assert_array_equal(result.groups, expected.groups)
    assert result.cases == expected.cases and result.exclusions == expected.exclusions
    assert result.cases[0]["recording_id"] == info.source_ids[0]
    assert result.cases[0]["participant_id"] == "sub-02"
    start, stop = (48, 80) if mode == "event_epochs" else (0, 32)
    np.testing.assert_array_equal(result.X[0], X[0, [1, 0], start:stop])
    assert result.y[0] == ("control" if mode == "event_epochs" else "case")
    provenance = result.metadata["provenance"]
    assert provenance["version"] == 1
    assert provenance["problem"] == expected.metadata["problem"]
    assert "problem" not in result.metadata
    source_report = info.to_dict()
    assert provenance["history"] == source_report.pop("history")
    assert provenance["source"] == source_report
    assert provenance["history"][0]["source_ids"] != list(info.source_ids)
    assert provenance["construction"] == {
        "implementation": "aeon_neuro.datasets.make_classification_problem",
        "version": __version__,
    }
    assert json.loads(json.dumps(result.metadata, allow_nan=False)) == result.metadata
    other_metadata = deepcopy(result.metadata)
    other_metadata.pop("provenance")
    expected.metadata.pop("problem")
    assert other_metadata == expected.metadata


def test_object_builder_detaches_signals_source_history_and_documents(source):
    """Changing an output must not change another problem or its original inputs."""
    X, info = load_bids_dataset(source[0])
    info = info.with_history(
        "metadata review",
        {"columns": ["trial_type"]},
        implementation="example.review",
        version="1",
    )
    problem, binding = _info_problem_documents("event_epochs")
    before = deepcopy((X, info.to_dict(), problem, binding))
    result = make_classification_problem(X, info, problem, binding)
    result.X[0] = 0
    provenance = result.metadata["provenance"]
    provenance["source"]["events"][0]["trial_type"] = "changed"
    provenance["source"]["dataset"]["participants"]["rows"][0]["group"] = "changed"
    provenance["history"][0]["parameters"]["columns"].append("changed")
    provenance["problem"]["definition"]["channels"].reverse()
    np.testing.assert_array_equal(X, before[0])
    assert (info.to_dict(), problem, binding) == before[1:]
    again = make_classification_problem(X, info, problem, binding)
    expected_source = info.to_dict()
    expected_source.pop("history")
    assert again.metadata["provenance"]["source"] == expected_source
    assert again.metadata["provenance"]["history"] == info.history


@pytest.mark.parametrize("fault", ["count", "channels", "samples"])
def test_object_builder_rejects_mismatched_source_layouts(source, fault):
    """The source view cannot silently describe changed axes or sample grids."""
    X, info = load_bids_dataset(source[0])
    changed = {"count": X[:1], "channels": X[:, :1], "samples": X[:, :, :128]}[fault]
    with pytest.raises(ValueError):
        make_classification_problem(
            changed, info, *_info_problem_documents("event_epochs")
        )


@pytest.mark.parametrize("wrong", [None, {}, [], "metadata"])
def test_object_builder_requires_the_shared_metadata_object(source, wrong):
    """Reject alternate implicit metadata/event input conventions."""
    X, _ = load_bids_dataset(source[0])
    with pytest.raises(ValueError, match="EEGDatasetInfo"):
        make_classification_problem(X, wrong, *_info_problem_documents("event_epochs"))


def test_object_builder_reads_no_bids_files_after_loading(source, monkeypatch):
    """Already-loaded sources need no second reader or event-table parser."""
    X, info = load_bids_dataset(source[0])

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "BIDS signal or metadata access during problem construction"
        )

    monkeypatch.setattr(module, "inspect_bids_eeg", unexpected)
    monkeypatch.setattr(module, "open_bids_eeg_recording", unexpected)
    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected)
    result = make_classification_problem(
        X, info, *_info_problem_documents("event_epochs")
    )
    assert result.X.shape == (4, 2, 32)
    assert result.metadata["provenance"]["history"] == []


def test_object_builder_preserves_exclusion_and_source_event_identity(source):
    """Out-of-bounds exclusions stay governed by the existing problem rules."""
    X, info = load_bids_dataset(source[0])
    problem, binding = _info_problem_documents("event_epochs")
    problem["segmentation"].update(tmin=0.0, tmax=7.0, on_out_of_bounds="exclude")
    result = make_classification_problem(X, info, problem, binding)
    expected = load_classification_problem(
        X, info.metadata, problem, binding, events=info.events
    )
    assert result.cases == expected.cases and result.exclusions == expected.exclusions
    assert len(result.cases) == len(result.exclusions) == 2
    assert result.exclusions[0]["exclusion_reason"] == "outside_recording"
    assert result.cases[0]["event_id"] == info.events[0]["bids_event_id"]


def test_object_builder_rejects_empty_views_and_missing_bound_participant_columns(
    source,
):
    """Metadata-only validity cannot supply absent labels or training observations."""
    X, info = load_bids_dataset(source[0])
    empty, empty_info = info.select(X, [])
    problem, binding = _info_problem_documents("fixed_windows")
    with pytest.raises(ValueError, match="nonempty"):
        make_classification_problem(empty, empty_info, problem, binding)
    binding["target"]["column"] = "participant.missing"
    with pytest.raises(ValueError, match="participant.missing"):
        make_classification_problem(X, info, problem, binding)
