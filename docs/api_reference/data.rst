.. _data_ref:

Data
====

Classification case construction
--------------------------------

.. autoclass:: aeon_neuro.datasets.ClassificationDataset

.. autofunction:: aeon_neuro.datasets.make_recording_cases

.. autofunction:: aeon_neuro.datasets.make_fixed_length_cases

.. autofunction:: aeon_neuro.datasets.make_event_cases

.. autofunction:: aeon_neuro.datasets.build_classification_dataset

.. autofunction:: aeon_neuro.datasets.validate_group_split

Classification recipe format (experimental)
------------------------------------------

.. toctree::
   :maxdepth: 1

   ../recipe_format

.. autofunction:: aeon_neuro.datasets.resolve_classification_recipe

Classification recipe loading (experimental)
-------------------------------------------

.. toctree::
   :maxdepth: 1

   ../classification_recipes

.. autoclass:: aeon_neuro.datasets.ClassificationRecipeResult

.. autofunction:: aeon_neuro.datasets.load_bids_classification
