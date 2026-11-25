"""Test experiment portability and execution without Console imports."""

import json

import numpy as np
import pytest

from aeon_neuro.experiments import (
    ExperimentRecipeError,
    load_experiment_recipe,
    make_experiment_recipe,
    run_experiment,
)


def test_load_legacy_and_embedded_console_recipes(tmp_path):
    """Check round trips and reject conflicting compatibility settings."""
    config = {"classifiers": ["riemann_mdm"], "authoring": {"question": "example"}}
    recipe = make_experiment_recipe(config)
    legacy = {
        "format": "eeg-console-setup",
        "format_version": 2,
        "source": {"kind": "raw"},
        "configuration": config,
    }
    assert load_experiment_recipe(legacy) == recipe
    assert recipe["configuration"] == {"classifiers": ["riemann_mdm"]}
    envelope = {**legacy, "experiment": recipe}
    file = tmp_path / "setup.json"
    file.write_text(json.dumps(envelope))
    assert load_experiment_recipe(file) == recipe
    config["classifiers"].append("csp_lda")
    assert recipe["configuration"]["classifiers"] == ["riemann_mdm"]
    with pytest.raises(ExperimentRecipeError, match="shared recipe differs"):
        load_experiment_recipe(envelope)


def test_refuse_wrong_version_and_invalid_operation_order():
    """Reject unsupported or malformed documents before execution."""
    recipe = make_experiment_recipe({})
    with pytest.raises(ExperimentRecipeError, match="format or version"):
        load_experiment_recipe({**recipe, "format_version": 99})
    with pytest.raises(ExperimentRecipeError, match="exactly once"):
        make_experiment_recipe({"preprocessing": {"continuous_order": ["resample"]}})
    with pytest.raises(ExperimentRecipeError, match="portable JSON"):
        make_experiment_recipe({"preprocessing": {"highpass_hz": float("nan")}})


def test_run_raw_recipe_and_prepared_replay():
    """Run a raw recipe and evaluate the same cases as supplied input."""
    rng = np.random.default_rng(5)
    signals = [rng.normal(0, 1e-6, (3, 1536)) for _ in range(2)]
    metadata = [
        {
            "recording_id": f"rec-{i}",
            "subject": str(i),
            "channel_names": ["C3", "C4", "Cz"],
            "sampling_frequency": 128,
            "events": [(n, "a" if n % 2 else "b") for n in range(1, 9)],
        }
        for i in range(2)
    ]
    config = {
        "labels": {"source": "events", "events": {"a": "A", "b": "B"}},
        "rules": {"tmin_ms": 0, "tmax_ms": 500},
        "preprocessing": {
            "resample_hz": 64,
            "anti_alias_resampling": True,
            "continuous_order": ["resample", "reference", "filter"],
            "lowpass_hz": 40,
            "excluded_channels": ["Cz"],
        },
        "evaluation": {"mode": "between", "scheme": "lopo"},
        "classifiers": ["riemann_mdm"],
    }
    dataset, result = run_experiment(
        make_experiment_recipe(config), signals, metadata, include_evidence=True
    )
    assert dataset.X.shape == (16, 2, 32)
    assert result["n_folds"] == 2
    prepared, replay = run_experiment(
        make_experiment_recipe(config, input_kind="prepared"),
        dataset,
        include_evidence=True,
    )
    assert prepared is dataset
    assert replay == result
