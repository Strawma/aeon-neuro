"""Read classification tables and validate reproducible group partitions."""

import csv

import numpy as np
from sklearn.model_selection import train_test_split


def _read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not fields:
            raise ValueError(f"Missing or duplicate TSV columns: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed TSV rows: {path}")
    return rows, fields


def _validate_partitions(cases, groups, splits):
    assignments = {}
    for name, indices in splits.items():
        for index in indices:
            if int(index) in assignments:
                raise ValueError("A case occurs in more than one partition.")
            assignments[int(index)] = name
    if len(assignments) != len(cases):
        raise ValueError("Split assignments must cover all included cases.")
    group_parts = {}
    for index, group in enumerate(groups):
        group_parts.setdefault(group, set()).add(assignments[index])
    if any(len(value) != 1 for value in group_parts.values()):
        raise ValueError("A group crosses split partitions.")
    import heapq

    records = {}
    for index, row in enumerate(cases):
        records.setdefault(row["source_identity"], []).append(
            (row["start_sample"], row["stop_sample"], assignments[index])
        )
    for intervals in records.values():
        active, counts = [], {}
        for start, stop, part in sorted(intervals):
            while active and active[0][0] <= start:
                _, expired = heapq.heappop(active)
                counts[expired] -= 1
            if any(count and name != part for name, count in counts.items()):
                raise ValueError("Source sample intervals overlap across partitions.")
            heapq.heappush(active, (stop, part))
            counts[part] = counts.get(part, 0) + 1
    classes = {row["label"] for row in cases}
    if len(classes) < 2:
        raise ValueError("Classification requires at least two retained classes.")
    for name, indices in splits.items():
        if not len(indices) or {cases[i]["label"] for i in indices} != classes:
            raise ValueError(
                f"Split {name!r} does not retain all classes. "
                "Change the declared protocol; no seed retries are performed."
            )


def _split_cases(cases, recipe, recipe_dir):
    evaluation = recipe["evaluation"]
    split = evaluation["split"]
    groups = [row["group_id"] for row in cases]
    unique = sorted(set(groups), key=lambda value: value.encode("utf-8"))
    if split["method"] == "supplied_manifest":
        if recipe_dir is None:
            raise ValueError(
                "Manifest replay requires a recipe filename to bind relative paths."
            )
        path = (recipe_dir / split["manifest"]).resolve()
        if not path.is_relative_to(recipe_dir.resolve()):
            raise ValueError("Split manifest escapes the recipe directory.")
        rows, fields = _read_tsv(path)
        keys = split["join_keys"]
        if any(set(keys) - row.keys() for row in cases):
            raise ValueError(
                "Split manifest join_keys must name available case fields."
            )
        if set(keys + [split["split_column"]]) - set(fields):
            raise ValueError("Split manifest is missing join or split columns.")
        lookup = {}
        for row in rows:
            key = tuple(row[k] for k in keys)
            if key in lookup:
                raise ValueError("Duplicate split-manifest join key.")
            lookup[key] = row[split["split_column"]]
        parts, used = [], set()
        for row in cases:
            key = tuple(str(row[k]) for k in keys)
            if key not in lookup:
                raise ValueError("Split manifest does not cover an included case.")
            parts.append(lookup[key])
            used.add(key)
        if set(lookup) != used or set(parts) - {"train", "validation", "test"}:
            raise ValueError(
                "Split manifest has unused identities or unknown partitions."
            )
        if not {"train", "test"}.issubset(parts):
            raise ValueError("Split manifest needs train and test partitions.")
        splits = {
            name: np.flatnonzero(np.asarray(parts) == name)
            for name in ("train", "validation", "test")
            if name in parts
        }
    else:
        fractions = split["fractions"]
        names = [
            name for name in ("train", "validation", "test") if fractions[name] > 0
        ]
        if len(unique) < len(names):
            raise ValueError(
                "Insufficient independent groups for the requested split; "
                "explicitly choose an appropriate grouping, "
                "never a per-window fallback."
            )
        if split["strategy"] == "chronological":
            if any(set(split["order_by"]) - row.keys() for row in cases):
                raise ValueError(
                    "Chronological order_by must name available case fields."
                )
            order = {}
            for row in cases:
                value = tuple(row[k] for k in split["order_by"])
                if row["group_id"] in order and order[row["group_id"]] != value:
                    raise ValueError(
                        "Chronological ordering must be constant within a group."
                    )
                order[row["group_id"]] = value
            unique.sort(key=lambda key: (order[key], key))
            n_test = int(np.ceil(len(unique) * fractions["test"]))
            n_val = int(np.ceil(len(unique) * fractions["validation"]))
            n_train = len(unique) - n_test - n_val
            if n_train < 1:
                raise ValueError(
                    "Insufficient groups for chronological split fractions."
                )
            by_part = {"train": unique[:n_train], "test": unique[n_train + n_val :]}
            if n_val:
                by_part["validation"] = unique[n_train : n_train + n_val]
        else:
            group_labels = None
            if split["stratify_by"] == "target":
                labels = {group: set() for group in unique}
                for row in cases:
                    labels[row["group_id"]].add(row["label"])
                if any(len(values) != 1 for values in labels.values()):
                    raise ValueError(
                        "Target stratification requires a single label per group."
                    )
                group_labels = {
                    group: next(iter(values)) for group, values in labels.items()
                }

            def shuffle(values, fraction):
                stratify = (
                    None if group_labels is None else [group_labels[g] for g in values]
                )
                return train_test_split(
                    values,
                    test_size=fraction,
                    random_state=split["random_seed"],
                    stratify=stratify,
                )

            train, test = shuffle(unique, fractions["test"])
            by_part = {"train": train, "test": test}
            if fractions["validation"]:
                train, validation = shuffle(
                    train, fractions["validation"] / (1 - fractions["test"])
                )
                by_part.update(train=train, validation=validation)
        membership = {
            group: name for name, values in by_part.items() for group in values
        }
        splits = {
            name: np.asarray(
                [i for i, group in enumerate(groups) if membership[group] == name],
                dtype=int,
            )
            for name in names
        }
    _validate_partitions(cases, groups, splits)
    for name, indices in splits.items():
        for index in indices:
            cases[index]["split"] = name
    return splits
