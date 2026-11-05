"""Check current case geometry reaches the classifier without rewriting sources."""

import importlib.util
import json
from pathlib import Path

import edfio
import numpy as np
import pytest
from sklearn.base import clone

from aeon_neuro.datasets import load_bids_dataset, make_classification_problem

SPEC = importlib.util.spec_from_file_location(
    "bids_braindecode_handoff",
    Path(__file__).resolve().parents[1] / "bids_braindecode_handoff.py",
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.fixture
def prepared(tmp_path):
    """Select three ordered channels from two distinguishable 64-channel sources."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text('{"Name":"Generated study"}')
    (root / "participants.tsv").write_text("participant_id\nsub-01\nsub-02\n")
    names = ["C3", "Cz", "C4"] + [f"aux{i}" for i in range(61)]
    for index, subject in enumerate(("01", "02")):
        folder = root / f"sub-{subject}/eeg"
        folder.mkdir(parents=True)
        stem = folder / f"sub-{subject}_task-imagery"
        values = np.random.default_rng(index).normal(size=(64, 512))
        edfio.Bdf(
            [
                edfio.BdfSignal(row, 64, label=name, physical_dimension="uV")
                for row, name in zip(values, names, strict=True)
            ]
        ).write(str(stem) + "_eeg.bdf")
        Path(str(stem) + "_eeg.json").write_text(
            '{"SamplingFrequency":64,"EEGReference":"Generated reference"}'
        )
        Path(str(stem) + "_channels.tsv").write_text(
            "name\ttype\tunits\tstatus\n"
            + "".join(f"{name}\tEEG\tuV\tgood\n" for name in names)
        )
        Path(str(stem) + "_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0\t2\tleft\n2\t2\tright\n"
            "4\t2\tleft\n6\t2\tright\n"
        )
        Path(str(stem) + "_electrodes.tsv").write_text(
            "name\tx\ty\tz\nC3\t0.01\t0.02\t0.03\n"
        )
        Path(str(stem) + "_coordsystem.json").write_text(
            '{"EEGCoordinateSystem":"Other","EEGCoordinateUnits":"m",'
            '"EEGCoordinateSystemDescription":"Generated coordinates"}'
        )
    problem = {
        "format_version": 1,
        "id": "handoff",
        "version": "1",
        "question": "Classify the explicit generated event target",
        "class_labels": ["left", "right"],
        "channels": ["C4", "C3", "Cz"],
        "segmentation": {"mode": "event_epochs", "tmin": 0, "tmax": 2},
    }
    binding = {
        "format_version": 1,
        "problem": {"id": "handoff", "version": "1"},
        "target": {
            "source": "event",
            "column": "trial_type",
            "mapping": {"left": "left", "right": "right"},
        },
        "group": {"source": "recording", "column": "participant_id"},
        "events": {"onset": "onset", "id": "bids_event_id"},
    }
    X, info = load_bids_dataset(root)
    selected, subset = info.select(X, [1, 0])
    dataset = make_classification_problem(selected, subset, problem, binding)
    return root, X, info, dataset, problem, binding


def test_prepare_uses_current_case_axes_without_fitting(prepared, monkeypatch):
    """A reordered three-channel case must not inherit its 64-channel source axis."""
    _, X, info, dataset, _, _ = prepared
    before = info.to_dict()
    samples = dataset.X.copy()

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Preparing a handoff must not fit")

    monkeypatch.setattr(example.BraindecodeClassifier, "fit", unexpected_fit)
    classifier, report = example.prepare_handoff(dataset, geometry="standard_1005")
    assert classifier.sfreq == 64
    assert classifier.channel_names == ["C4", "C3", "Cz"]
    assert report["input"]["shape"] == [8, 3, 128]
    assert report["input"]["channel_names"] == classifier.channel_names
    assert report["geometry"] == "standard_1005 template; source geometry unused"
    assert report["case_ids"] == [row["case_id"] for row in dataset.cases]
    np.testing.assert_array_equal(dataset.X[0], X[1, [2, 0, 1], :128])
    np.testing.assert_array_equal(dataset.X, samples)
    assert info.to_dict() == before
    assert len(report["provenance"]["source"]["recordings"][0]["channel_names"]) == 64
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_handoff_owns_parameters_and_report(prepared):
    """Editing a classifier or exported report must not alter case/source context."""
    _, _, info, dataset, _, _ = prepared
    parameters = {"kernel_length": 4}
    classifier, report = example.prepare_handoff(
        dataset, geometry="standard_1005", model_params=parameters
    )
    assert clone(classifier).get_params()["model_params"] == parameters
    classifier.model_params["kernel_length"] = 8
    classifier.channel_names.reverse()
    report["provenance"]["source"]["recordings"][0]["channel_names"].clear()
    assert parameters == {"kernel_length": 4}
    assert dataset.metadata["channel_names"] == ["C4", "C3", "Cz"]
    assert len(info.recordings[0].channel_names) == 64
    assert report["classifier_parameters"]["model_params"] == parameters


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rate",
        "bool_rate",
        "source_names",
        "order",
        "labels",
        "ragged",
        "nonfinite",
        "provenance",
    ],
)
def test_changed_or_missing_case_metadata_blocks_handoff(prepared, fault):
    """Fail before fitting when current metadata and case arrays disagree."""
    _, _, _, dataset, _, _ = prepared
    if fault == "missing_rate":
        dataset.metadata.pop("sampling_frequency")
    elif fault == "bool_rate":
        dataset.metadata["sampling_frequency"] = True
    elif fault == "source_names":
        dataset.metadata["channel_names"] = dataset.metadata["recordings"][0][
            "channel_names"
        ]
    elif fault == "order":
        dataset.metadata["channel_names"].reverse()
    elif fault == "labels":
        dataset.y = dataset.y[:-1]
    elif fault == "ragged":
        dataset.X = list(dataset.X)
    elif fault == "nonfinite":
        dataset.X[0, 0, 0] = np.nan
    else:
        dataset.metadata.pop("provenance")
    with pytest.raises(ValueError):
        example.prepare_handoff(dataset, geometry="none")


def test_cpu_fit_is_an_interface_check_only(prepared):
    """One synthetic training epoch checks dimensions/probabilities, not accuracy."""
    _, _, info, dataset, _, _ = prepared
    before, source = dataset.X.copy(), info.to_dict()
    classifier, report = example.prepare_handoff(dataset, geometry="none")
    assert classifier.channel_names is None
    classifier.fit(dataset.X, dataset.y)
    probabilities = classifier.predict_proba(dataset.X)
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert classifier.network_.n_chans == 3
    assert classifier.network_.n_times == 128
    assert classifier.network_.sfreq == 64
    assert len(classifier.summary()["loss"]) == 1
    assert report["fit_performed"] is False
    np.testing.assert_array_equal(dataset.X, before)
    assert info.to_dict() == source


def test_cli_exports_configuration_and_rejects_existing_output(prepared, tmp_path):
    """A configuration-only local run saves portable evidence outside the source."""
    root, _, _, _, problem, binding = prepared
    paths = []
    for name, document in (("problem", problem), ("binding", binding)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        paths.append(str(path))
    output = tmp_path / "handoff.json"
    arguments = [
        str(root),
        "--subject",
        "01",
        "--problem",
        paths[0],
        "--binding",
        paths[1],
        "--geometry",
        "none",
        "--output",
        str(output),
    ]
    example.main(arguments)
    report = json.loads(output.read_text())
    assert report["input"]["shape"] == [4, 3, 128]
    assert report["fit_performed"] is False
    with pytest.raises(FileExistsError):
        example.main(arguments)


def test_geometry_choice_is_required_and_unknown_policy_fails(prepared):
    """Template electrode positions must be an explicit choice."""
    with pytest.raises(TypeError):
        example.prepare_handoff(prepared[3])
    with pytest.raises(ValueError, match="geometry"):
        example.prepare_handoff(prepared[3], geometry="source")


def test_cli_source_output_is_rejected_before_loading(prepared, monkeypatch):
    """Configuration export must not add files inside the inspected source tree."""
    root = prepared[0]

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Reject an invalid output before reading samples")

    monkeypatch.setattr(example, "load_bids_dataset", unexpected_load)
    with pytest.raises(ValueError, match="outside"):
        example.main(
            [
                str(root),
                "--subject",
                "01",
                "--problem",
                "unused.json",
                "--binding",
                "unused.json",
                "--geometry",
                "none",
                "--output",
                str(root / "handoff.json"),
            ]
        )
    assert not (root / "handoff.json").exists()
