"""Validate recipe author inputs independently of native BIDS readers."""

import json
from copy import deepcopy

import pytest

from aeon_neuro.datasets import resolve_classification_recipe

pytest.importorskip("jsonschema")


def _fixed_recipe():
    return {
        "spec_version": "0.3.0-draft",
        "target": {"source": "participants.tsv", "column": "group"},
        "segmentation": {"mode": "fixed_windows", "window_seconds": 1},
        "evaluation": {"split": {"stratify_by": "target"}},
    }


def test_defaults_without_input_mutation():
    """Resolve omissions without inventing signal facts or aliasing the input."""
    original = _fixed_recipe()
    before = deepcopy(original)
    resolved, origins = resolve_classification_recipe(original)
    assert original == before
    assert resolved["segmentation"]["stride_seconds"] == 1
    assert "sampling_frequency_hz" not in resolved["segmentation"]
    assert resolved["evaluation"]["split"]["fractions"] == {
        "train": 0.8,
        "validation": 0,
        "test": 0.2,
    }
    assert resolved["evaluation"]["split"]["random_seed"] == 42
    assert origins["evaluation.split.random_seed"] == "default"
    assert resolved["evaluation"]["group_by"] == ["bids_subject"]
    assert (
        resolve_classification_recipe(original, random_state=7)[0]["evaluation"][
            "split"
        ]["random_seed"]
        == 7
    )


@pytest.mark.parametrize(
    "change",
    [
        {"spec_version": "next"},
        {"extra": True},
        {"segmentation": {"mode": "fixed_windows"}},
        {"segmentation": {"mode": "fixed_windows", "window_seconds": 0}},
        {"segmentation": {"mode": "event_epochs", "stop_seconds": 0}},
        {"segmentation": {"mode": "fixed_windows", "window_seconds": float("nan")}},
        {"evaluation": {"split": {"random_seed": -1}}},
        {
            "evaluation": {
                "split": {"fractions": {"train": 0.5, "validation": 0.2, "test": 0.2}}
            }
        },
    ],
)
def test_invalid_author_choices(change):
    """Reject malformed choices instead of treating invalid values as omissions."""
    recipe = _fixed_recipe()
    recipe.update(change)
    with pytest.raises(ValueError):
        resolve_classification_recipe(recipe)


def test_duplicate_json_key_is_an_error(tmp_path):
    """Reject conflicting repeated settings instead of accepting the last value."""
    path = tmp_path / "ambiguous.json"
    path.write_text(
        json.dumps(_fixed_recipe()).replace(
            '"window_seconds": 1', '"window_seconds": 2, "window_seconds": 1'
        )
    )
    with pytest.raises(ValueError, match="Duplicate recipe JSON key"):
        resolve_classification_recipe(path)
