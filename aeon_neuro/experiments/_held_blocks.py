"""Read and check declared two-fold within-person block assignments."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

ASSIGNMENTS_FILE = "block_assignments.tsv"
MODE = "held_blocks"
AGGREGATION = "mean_participant_balanced_accuracy"


@dataclass(frozen=True)
class Case:
    case_id: str
    participant_id: str
    recording_id: str
    block_id: str
    label: str
    test_fold: str
    start_sample: int
    stop_sample: int


def read_assignments(path):
    """Load the supplied coordinates and IDs, preserving identifiers as text."""
    text_fields = (
        "case_id",
        "participant_id",
        "recording_id",
        "block_id",
        "label",
        "test_fold",
    )
    required = {*text_fields, "start_sample", "stop_sample"}
    cases = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = reader.fieldnames or []
        if required - set(columns) or len(columns) != len(set(columns)):
            raise ValueError(
                f"Expected unique columns including {', '.join(sorted(required))}."
            )
        for line, row in enumerate(reader, start=2):
            values = [(row[key] or "").strip() for key in text_fields]
            if None in row or any(
                value.lower() in {"", "n/a", "na", "unknown"} for value in values
            ):
                raise ValueError(
                    f"Line {line} has missing case, source, block, fold or label"
                    " information."
                )
            try:
                start, stop = int(row["start_sample"]), int(row["stop_sample"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Line {line} needs integer source samples."
                ) from error
            if start < 0 or stop <= start:
                raise ValueError(f"Line {line} needs a nonempty source interval.")
            case = Case(*values, start, stop)
            if case.case_id in cases:
                raise ValueError(f"Duplicate case ID: {case.case_id}.")
            cases[case.case_id] = case
    return cases


def check_assignments(cases):
    """Return exact fold memberships after checking block and sample separation."""
    people = defaultdict(list)
    for case in cases.values():
        people[case.participant_id].append(case)
    if not people:
        raise ValueError("No cases were supplied.")
    labels = {case.label for case in cases.values()}
    if len(labels) < 2:
        raise ValueError("Supply at least two outcome labels.")
    expected = {}
    for person, rows in people.items():
        by_block = defaultdict(set)
        by_fold = defaultdict(list)
        for case in rows:
            by_block[case.recording_id, case.block_id].add(case.test_fold)
            by_fold[case.test_fold].append(case)
        if any(len(folds) != 1 for folds in by_block.values()):
            raise ValueError(
                f"{person}: a block is assigned to more than one test fold."
            )
        if len(by_fold) != 2:
            raise ValueError(f"{person}: expected exactly two test folds.")
        first, second = (by_fold[key] for key in sorted(by_fold))
        if any(
            a.recording_id == b.recording_id
            and a.start_sample < b.stop_sample
            and b.start_sample < a.stop_sample
            for a, b in product(first, second)
        ):
            raise ValueError(f"{person}: source intervals overlap across folds.")
        for held_out, train in ((first, second), (second, first)):
            if {case.label for case in held_out} != labels:
                raise ValueError(
                    f"{person}, fold {held_out[0].test_fold}: missing outcome labels on"
                    " one side."
                )
            expected[frozenset(case.case_id for case in held_out)] = {
                case.case_id for case in train
            }
    return expected
