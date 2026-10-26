Classification recipe format (experimental)
===========================================

A recipe records choices that BIDS does not supply: which labels define a
classification task, which intervals become cases, and which groups stay
together during evaluation. Version ``0.3.0-draft`` is a GDP proposal, not an
accepted BIDS extension. Review the scientific choices for each dataset.

Install ``aeon-neuro[recipes]`` to enable JSON Schema validation. The public
``resolve_classification_recipe`` function accepts a dictionary or UTF-8 JSON
filename and returns a separate resolved dictionary and setting origins.
It validates author choices and supplies defaults; it performs no data reads,
splitting, signal processing or model fitting.

Minimum author input
--------------------

.. code-block:: python

   from aeon_neuro.datasets import resolve_classification_recipe

   recipe, origins = resolve_classification_recipe({
       "spec_version": "0.3.0-draft",
       "target": {"source": "participants.tsv", "column": "group"},
       "segmentation": {"mode": "fixed_windows", "window_seconds": 2},
   })

The BIDS root is supplied separately to a data loader. For event-labelled
epochs, specify ``target.source: events.tsv`` and
``segmentation: {mode: event_epochs, stop_seconds: 2}``. Source label strings
are used directly unless a mapping is declared. Neither target meanings nor
appropriate epoch duration can be inferred safely from BIDS alone.

Omitted settings
----------------

* Selection includes available subjects, sessions and runs. An eventual loader
  must resolve an omitted task only when one task is available.
* Channels use BIDS-declared EEG roles. No rate or channel identity is invented.
* Fixed-window stride equals its duration. The interval covers the recording;
  incomplete windows are dropped. Event epochs begin at the source onset.
* Evaluation groups whole BIDS subjects and shuffles groups with seed 42 into
  80% training and 20% testing, without stratification. A split manifest is
  optional for replay. The data loader must still check that a split is possible.
* Missing labels are errors. There is no implicit filtering, normalisation,
  resampling, re-referencing or artifact rejection.

The packaged ``classification_recipe.schema.json`` documents supported fields
and rejects unknown keys. JSON Schema defaults are annotations; the resolver
explicitly fills supported omissions and records their origins. Repeated JSON
keys, non-finite numbers and inconsistent semantic choices are errors. Input
dictionaries are never mutated. Data-dependent validation belongs in the loader.

The format also permits explicit source-label mappings/exclusions, channel
names or prefixes, state-bounded windows, custom event-number blocks,
chronological ordering and manifest replay. These are deliberate protocol
choices rather than requirements for every recipe.
