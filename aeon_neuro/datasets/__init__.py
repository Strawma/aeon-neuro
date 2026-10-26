"""Utilities for loading datasets."""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "load_kdd_example",
    "load_kdd_full_example",
    "load_eeg_classification",
    "resolve_classification_recipe",
]

from aeon_neuro.datasets._data_loaders import load_eeg_classification
from aeon_neuro.datasets._recipe import resolve_classification_recipe
from aeon_neuro.datasets._single_problem_loaders import (
    load_kdd_example,
    load_kdd_full_example,
)
