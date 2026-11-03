"""Exercise the local walkthrough with small generated BDFs, never participant data."""

import hashlib
import importlib.util
import json
from pathlib import Path

import edfio
import mne
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cho2017_example", EXAMPLES / "cho2017_problem.py"
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
PROBLEM = EXAMPLES / "problem_definitions/cho2017.problem.json"
BINDING = EXAMPLES / "problem_definitions/cho2017.binding.json"


@pytest.fixture
def local_recording(tmp_path):
    """Use the release layout with generated samples and a matching test manifest."""
    root = tmp_path / "dataset"
    stem = root / "sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0"
    stem.parent.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "10.82901/nemar.nm000245", "Version": "1.0.2"})
    )
    Path(str(stem) + "_channels.tsv").write_text(
        "name\ttype\nC3\tEEG\nCz\tEEG\nC4\tEEG\n"
    )
    Path(str(stem) + "_eeg.json").write_text('{"SamplingFrequency": 16}')
    Path(str(stem) + "_events.tsv").write_text(
        "\ufeffonset\tduration\ttrial_type\tvalue\tsample\n"
        "1\t3\tleft_hand\t1\t16\n4\t3\tright_hand\t2\t64\n"
    )
    values = np.random.default_rng(42).normal(size=(3, 128))
    signal = Path(str(stem) + "_eeg.bdf")
    edfio.Bdf(
        [
            edfio.BdfSignal(row, 16, label=name, physical_dimension="uV")
            for row, name in zip(values, ("C3", "Cz", "C4"))
        ]
    ).write(signal)
    entries = []
    for path in root.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            # Cover both checksum encodings used by the actual NEMAR manifest.
            algorithm = "sha256" if path == signal else "git"
            digest = (
                hashlib.sha256(content).hexdigest()
                if algorithm == "sha256"
                else hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest()
            )
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(content),
                    "checksum_algorithm": algorithm,
                    "checksum": digest,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return root, manifest, signal


def test_plan_needs_no_signal_source(local_recording, monkeypatch):
    """A saved catalogue remains usable when the recording is unavailable."""
    root, manifest, signal = local_recording
    catalogue = example.inspect_recording(root, manifest, 1)
    signal.unlink()

    def unexpected_access(*args, **kwargs):
        raise AssertionError("Signal access during metadata planning")

    monkeypatch.setattr(mne.io, "read_raw_bdf", unexpected_access)
    monkeypatch.setattr(example, "_hash", unexpected_access)
    plan = example.make_plan(json.loads(json.dumps(catalogue)), PROBLEM, BINDING)
    assert plan["dense_shape"] == [2, 3, 32]
    assert [row["label"] for row in plan["cases"]] == ["left_hand", "right_hand"]
    assert plan["recordings"][0]["eligibility_policy"] == example.ELIGIBILITY
    assert [row["start_sample"] for row in plan["cases"]] == [24, 72]


def test_local_cases_match_bdf_without_source_changes(local_recording):
    """The authored intervals and selected IDs reach exact decoded samples."""
    root, manifest, signal = local_recording
    catalogue = example.inspect_recording(root, manifest, 1)
    plan = example.make_plan(catalogue, PROBLEM, BINDING)
    before = example._hash(signal)
    dataset = example.load_cases(root, plan)
    with mne.io.read_raw_bdf(signal, preload=False, verbose=False) as raw:
        for row, values in zip(dataset.cases, dataset.X):
            np.testing.assert_array_equal(
                values,
                raw.get_data(
                    picks=["C3", "Cz", "C4"],
                    start=row["start_sample"],
                    stop=row["stop_sample"],
                ),
            )
    chosen = plan["cases"][-1]["case_id"]
    subset = example.load_cases(root, plan, [chosen])
    np.testing.assert_array_equal(subset.X, dataset.X[-1:])
    assert example._hash(signal) == before


@pytest.mark.parametrize("corrupt_signal", [False, True])
def test_changed_inputs_fail_before_signal_open(
    local_recording, monkeypatch, corrupt_signal
):
    """Manifest checks reject changed sidecars or same-sized changed signals."""
    root, manifest, signal = local_recording
    catalogue = example.inspect_recording(root, manifest, 1)
    plan = example.make_plan(catalogue, PROBLEM, BINDING)

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Changed source reached the decoder")

    monkeypatch.setattr(mne.io, "read_raw_bdf", unexpected_open)
    if corrupt_signal:
        with signal.open("r+b") as stream:
            stream.write(b"X")
        with pytest.raises(ValueError, match="Signal differs"):
            example.load_cases(root, plan)
    else:
        signal.with_suffix(".json").write_text("{}")
        with pytest.raises(ValueError, match="Metadata differs"):
            example.inspect_recording(root, manifest, 1)


def test_cli_saves_portable_plan_and_raw_outputs(
    local_recording, tmp_path, monkeypatch
):
    """Exercise all documented commands and refuse reuse of an output directory."""
    root, manifest, _ = local_recording
    catalogue, plan, output = (
        tmp_path / "catalogue.json",
        tmp_path / "plan.json",
        tmp_path / "cases",
    )
    commands = [
        ["inspect", "--dataset", root, "--manifest", manifest, "--output", catalogue],
        ["plan", "--catalogue", catalogue, "--output", plan],
        ["materialize", "--dataset", root, "--plan", plan, "--output", output],
    ]
    for args in commands:
        monkeypatch.setattr("sys.argv", ["cho2017_problem.py", *map(str, args)])
        example.main()
    assert np.load(output / "X.npy", allow_pickle=False).shape == (2, 3, 32)
    assert np.load(output / "y.npy", allow_pickle=False).tolist() == [
        "left_hand",
        "right_hand",
    ]
    assert len(json.loads((output / "cases.json").read_text())) == 2
    with pytest.raises(FileExistsError, match="fresh output"):
        example.main()
