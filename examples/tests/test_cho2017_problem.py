"""Exercise the local walkthrough with small generated BDFs, never participant data."""

import hashlib
import importlib.util
import json
import os
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
        "1\t3\tleft_hand\t1\t16\n4\t3\tright_hand\t2\t64\n",
        encoding="utf-8",
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
    assert catalogue["ready"]
    assert catalogue["recordings"][0]["participant_id"] == "sub-1"
    assert {issue["code"] for issue in catalogue["issues"]} >= {
        "participants_missing",
        "quality_unreviewed",
        "content_unverified",
    }
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


def test_local_cases_match_bdf_without_source_changes(local_recording, monkeypatch):
    """The authored intervals and selected IDs reach exact decoded samples."""
    root, manifest, signal = local_recording
    catalogue = example.inspect_recording(root, manifest, 1)
    plan = example.make_plan(catalogue, PROBLEM, BINDING)
    before = example._hash(signal)
    opened = []
    original = example.open_bids_eeg_recording

    def watched(root, descriptor):
        opened.append(descriptor["recording_id"])
        return original(root, descriptor)

    monkeypatch.setattr(example, "open_bids_eeg_recording", watched)
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
    assert opened == [catalogue["source_ids"][0]] * 2
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
        stat = signal.stat()
        with signal.open("r+b") as stream:
            stream.write(b"X")
        # Unchanged size/time must still fail the trusted manifest checksum.
        os.utime(signal, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError, match="Signal checksum differs"):
            example.load_cases(root, plan)
    else:
        signal.with_suffix(".json").write_text("{}")
        with pytest.raises(ValueError, match="Metadata differs"):
            example.inspect_recording(root, manifest, 1)


def test_changed_metadata_after_planning_blocks_materialization(
    local_recording, monkeypatch
):
    """The shared opener verifies the saved sidecars before opening signals."""
    root, manifest, signal = local_recording
    catalogue = example.inspect_recording(root, manifest, 1)
    plan = example.make_plan(catalogue, PROBLEM, BINDING)
    signal.with_suffix(".json").write_text("{}")

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Changed metadata reached the decoder")

    monkeypatch.setattr(mne.io, "read_raw_bdf", unexpected_open)
    with pytest.raises(ValueError, match="Inspected source changed"):
        example.load_cases(root, plan)


@pytest.mark.parametrize("in_manifest", [False, True])
def test_additional_inspected_metadata_needs_release_verification(
    local_recording, in_manifest
):
    """Optional source facts cannot silently bypass the declared release pin."""
    root, manifest, _ = local_recording
    participants = root / "participants.tsv"
    participants.write_text("participant_id\tcohort\nsub-1\tcontrol\n")
    if in_manifest:
        entries = json.loads(manifest.read_text())
        entries.append(
            {
                "path": "participants.tsv",
                "size": participants.stat().st_size,
                "checksum_algorithm": "sha256",
                "checksum": hashlib.sha256(participants.read_bytes()).hexdigest(),
            }
        )
        manifest.write_text(json.dumps(entries))
        catalogue = example.inspect_recording(root, manifest, 1)
        assert catalogue["recordings"][0]["participant.cohort"] == "control"
    else:
        with pytest.raises(ValueError, match="absent from the release manifest"):
            example.inspect_recording(root, manifest, 1)


def test_adapter_blockers_keep_their_diagnostic_message(local_recording):
    """A release-consistent sidecar can still contradict the recording header."""
    root, manifest, signal = local_recording
    sidecar = signal.with_suffix(".json")
    sidecar.write_text('{"SamplingFrequency": 32}')
    entries = json.loads(manifest.read_text())
    entry = next(row for row in entries if row["path"].endswith("_eeg.json"))
    entry.update(
        size=sidecar.stat().st_size,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    )
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="inspection_failed:.*SamplingFrequency"):
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


@pytest.fixture
def quality_rows():
    """Use the verified class-stacked layout with two original trials per class."""
    markers = np.zeros(20)
    markers[[3, 13]] = 1
    eeg = {
        "srate": 512,
        "imagery_event": markers,
        "n_imagery_trials": 2,
        "imagery_left": np.zeros((68, 20)),
        "imagery_right": np.zeros((68, 20)),
        "bad_trial_indices": {
            "bad_trial_idx_voltage": [np.array([]), np.array([])],
            "bad_trial_idx_mi": [np.array([]), np.array([])],
        },
    }
    events = [
        {
            "trial_type": label,
            "value": str(code),
            "sample": str(offset + cue),
            "onset": str((offset + cue) / 512),
        }
        for label, code, offset in [("left_hand", 1, 0), ("right_hand", 2, 520)]
        for cue in (3, 13)
    ]
    return eeg, events


@pytest.mark.parametrize("origin", [0, 1])
def test_quality_flags_remain_class_local_and_explicit(quality_rows, origin):
    """Flag kinds form a union without guessing an origin or changing source rows."""
    eeg, events = quality_rows
    before = json.loads(json.dumps(events))
    eeg["bad_trial_indices"]["bad_trial_idx_voltage"] = [
        np.array([origin]),
        np.array([]),
    ]
    eeg["bad_trial_indices"]["bad_trial_idx_mi"] = [
        np.array([]),
        np.array([origin + 1]),
    ]
    joined = example._join_source_quality(
        eeg, events, index_origin=origin, buffer_samples=500
    )
    assert [row["included"] for row in joined] == [False, True, True, False]
    assert [row["original_trial_id"] for row in joined] == [
        f"left_hand:{origin}",
        f"left_hand:{origin + 1}",
        f"right_hand:{origin}",
        f"right_hand:{origin + 1}",
    ]
    assert joined[0]["bad_voltage"] and not joined[0]["bad_emg"]
    assert joined[-1]["bad_emg"] and not joined[-1]["bad_voltage"]
    assert joined[0]["exclusion_reason"] == "source_quality"
    assert [row["source_trial_start_sample"] for row in joined] == [0, 10, 520, 530]
    assert events == before


@pytest.mark.parametrize(
    "fault", ["sample", "onset", "label", "code", "count", "markers", "shape", "flags"]
)
def test_quality_join_rejects_mismatched_source_trials(quality_rows, fault):
    """A plausible count alone cannot attach quality decisions to shifted events."""
    eeg, events = quality_rows
    if fault in ("sample", "onset", "label", "code"):
        column, value = {
            "sample": ("sample", "4"),
            "onset": ("onset", "0"),
            "label": ("trial_type", "right_hand"),
            "code": ("value", "2"),
        }[fault]
        events[0][column] = value
    elif fault == "count":
        events.pop()
    elif fault == "markers":
        eeg["imagery_event"][13] = 0
        eeg["imagery_event"][14] = 1
    elif fault == "shape":
        eeg["imagery_left"] = np.zeros((64, 20))
    else:
        del eeg["bad_trial_indices"]
    with pytest.raises(ValueError):
        example._join_source_quality(eeg, events, index_origin=1, buffer_samples=500)


@pytest.mark.parametrize("values", [None, [0], [3], [1.5], [np.nan], [1, 1]])
def test_invalid_quality_indices_cannot_become_clean_trials(values):
    """Missing, fractional, repeated and out-of-range flags remain blockers."""
    with pytest.raises(ValueError):
        example._flagged_trials(values, count=2, origin=1)


@pytest.fixture
def quality_source(local_recording):
    """Add a tiny source MAT with a flagged right-hand trial to the release."""
    from scipy.io import savemat

    root, manifest, signal = local_recording
    event_file = signal.with_name(signal.name.removesuffix("_eeg.bdf") + "_events.tsv")
    event_file.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "1\t3\tleft_hand\t1\t16\n5\t3\tright_hand\t2\t80\n",
        encoding="utf-8",
    )
    markers = np.zeros(64)
    markers[16] = 1
    voltage, emg = np.empty(2, dtype=object), np.empty(2, dtype=object)
    voltage[0], voltage[1] = np.array([]), np.array([])
    emg[0], emg[1] = np.array([]), np.array([1])
    mat = root / "sourcedata/s01.mat"
    mat.parent.mkdir()
    savemat(
        mat,
        {
            "eeg": {
                "srate": 16,
                "imagery_event": markers,
                "n_imagery_trials": 1,
                "imagery_left": np.zeros((68, 64)),
                "imagery_right": np.zeros((68, 64)),
                "bad_trial_indices": {
                    "bad_trial_idx_voltage": voltage,
                    "bad_trial_idx_mi": emg,
                },
            }
        },
    )
    entries = json.loads(manifest.read_text())
    event_entry = next(row for row in entries if row["path"].endswith("_events.tsv"))
    event_entry.update(
        size=event_file.stat().st_size,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(event_file.read_bytes()).hexdigest(),
    )
    entries.append(
        {
            "path": "sourcedata/s01.mat",
            "size": mat.stat().st_size,
            "checksum_algorithm": "sha256",
            "checksum": hashlib.sha256(mat.read_bytes()).hexdigest(),
        }
    )
    manifest.write_text(json.dumps(entries))
    return root, manifest, signal, mat


def test_quality_catalogue_preserves_evidence_and_raw_values(quality_source):
    """The optional policy changes eligibility while the default route stays raw."""
    root, manifest, signal, mat = quality_source
    before = example._hash(mat), example._hash(signal)
    unfiltered = example.inspect_recording(root, manifest, 1)
    assert len(example.make_plan(unfiltered, PROBLEM, BINDING)["cases"]) == 2
    quality = example.inspect_quality_recording(
        root, manifest, 1, index_origin=1, buffer_samples=0, cohort="all"
    )
    binding = EXAMPLES / "problem_definitions/cho2017-quality.binding.json"
    plan = example.make_plan(quality, PROBLEM, binding)
    assert len(plan["cases"]) == len(plan["exclusions"]) == 1
    assert plan["exclusions"][0]["event_id"] == "right_hand:1"
    assert plan["exclusions"][0]["exclusion_reason"] == "source_quality"
    assert quality["recordings"][0]["source_quality"]["excluded_subjects"] == []
    dataset = example.load_cases(root, plan)
    assert dataset.cases[0]["trial_id"] == "left_hand:1"
    with mne.io.read_raw_bdf(signal, preload=False, verbose=False) as raw:
        np.testing.assert_array_equal(
            dataset.X[0], raw.get_data(picks=["C3", "Cz", "C4"], start=24, stop=56)
        )
    assert before == (example._hash(mat), example._hash(signal))


def test_quality_plan_needs_no_source_mat_or_signal(quality_source, monkeypatch):
    """MAT decoding occurs at the explicit quality stage, never saved planning."""
    root, manifest, signal, mat = quality_source
    quality = example.inspect_quality_recording(
        root, manifest, 1, index_origin=1, buffer_samples=0, cohort="paper"
    )
    signal.unlink()
    mat.unlink()

    def unexpected(*args, **kwargs):
        raise AssertionError("Source access during saved metadata planning")

    monkeypatch.setattr(example, "_hash", unexpected)
    monkeypatch.setattr("scipy.io.loadmat", unexpected)
    plan = example.make_plan(
        json.loads(json.dumps(quality)),
        PROBLEM,
        EXAMPLES / "problem_definitions/cho2017-quality.binding.json",
    )
    assert plan["dense_shape"] == [1, 3, 32]


@pytest.mark.parametrize(
    "subject, cohort, origin, buffer",
    [
        (29, "paper", 1, 0),
        (34, "paper", 1, 0),
        (1, "paper", True, 0),
        (1, "all", 1, -1),
        (1, "unknown", 1, 0),
    ],
)
def test_quality_policy_choices_fail_before_source_access(
    tmp_path, subject, cohort, origin, buffer
):
    """Published cohort exclusions and invalid choices need no source decoding."""
    with pytest.raises(ValueError):
        example.inspect_quality_recording(
            tmp_path,
            tmp_path / "absent",
            subject,
            index_origin=origin,
            buffer_samples=buffer,
            cohort=cohort,
        )


def test_quality_mat_checksum_is_verified_before_decoding(quality_source, monkeypatch):
    """A changed original file cannot supply trusted trial flags."""
    root, manifest, _, mat = quality_source
    with mat.open("r+b") as stream:
        stream.write(b"X")

    def unexpected(*args, **kwargs):
        raise AssertionError("Unverified MAT source reached the decoder")

    monkeypatch.setattr("scipy.io.loadmat", unexpected)
    with pytest.raises(ValueError, match="MATLAB source differs"):
        example.inspect_quality_recording(
            root, manifest, 1, index_origin=1, buffer_samples=0, cohort="paper"
        )


def test_changed_quality_evidence_blocks_materialization(quality_source, monkeypatch):
    """The shared opener revalidates saved source-quality evidence as well."""
    root, manifest, _, mat = quality_source
    quality = example.inspect_quality_recording(
        root, manifest, 1, index_origin=1, buffer_samples=0, cohort="paper"
    )
    plan = example.make_plan(
        quality, PROBLEM, EXAMPLES / "problem_definitions/cho2017-quality.binding.json"
    )
    stat = mat.stat()
    with mat.open("r+b") as stream:
        stream.write(b"X")
    os.utime(mat, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def unexpected(*args, **kwargs):
        raise AssertionError("Changed quality evidence reached the BDF decoder")

    monkeypatch.setattr(mne.io, "read_raw_bdf", unexpected)
    with pytest.raises(ValueError, match="Inspected source changed"):
        example.load_cases(root, plan)


def test_quality_cli_writes_policy_plan_and_exclusions(
    quality_source, tmp_path, monkeypatch
):
    """Exercise the opt-in command and preserve excluded trials in output artifacts."""
    root, manifest, _, _ = quality_source
    catalogue, plan, output = (
        tmp_path / "quality.json",
        tmp_path / "quality-plan.json",
        tmp_path / "quality-cases",
    )
    binding = EXAMPLES / "problem_definitions/cho2017-quality.binding.json"
    commands = [
        [
            "quality",
            "--dataset",
            root,
            "--manifest",
            manifest,
            "--subject",
            "1",
            "--index-origin",
            "1",
            "--buffer-samples",
            "0",
            "--cohort",
            "paper",
            "--output",
            catalogue,
        ],
        ["plan", "--catalogue", catalogue, "--binding", binding, "--output", plan],
        ["materialize", "--dataset", root, "--plan", plan, "--output", output],
    ]
    for args in commands:
        monkeypatch.setattr("sys.argv", ["cho2017_problem.py", *map(str, args)])
        example.main()
    assert np.load(output / "X.npy").shape == (1, 3, 32)
    excluded = json.loads((output / "exclusions.json").read_text())
    assert excluded[0]["event_id"] == "right_hand:1"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["recordings"][0]["source_quality"]["cohort"] == "paper"
    assert (
        "Unflagged does not certify" in metadata["recordings"][0]["eligibility_policy"]
    )
