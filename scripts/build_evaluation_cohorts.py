#!/usr/bin/env python3
"""Freeze deterministic C3 evaluation cohorts before any model score is inspected."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ALGORITHM_VERSION = "c3_evaluation_cohorts_v0.1"
EVALUATION_PARTITIONS = ("validation", "test")
COHORT_FULL = "full_evidence_pool"
COHORT_BALANCED = "balanced_explicit_1to1"
COHORT_CURATED = "balanced_curated_1to1"
COHORT_SCREEN = "balanced_screen_1to1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader)


def require_fields(rows: list[dict[str, str]], required: set[str], table_name: str) -> None:
    if not rows:
        raise ValueError(f"{table_name} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{table_name} is missing columns: {', '.join(sorted(missing))}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_selection_key(seed: int, cohort: str, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{cohort}:{pair_id}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "cohort",
        "partition",
        "pair_id",
        "label",
        "evidence_class",
        "bootstrap_group",
        "selection_reason",
        "positive_selection_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cohort_rows(
    cohort: str,
    partition: str,
    negatives: list[dict[str, str]],
    positives: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    if not negatives:
        raise ValueError(f"{partition}/{cohort} has no explicit negatives")
    if len(positives) < len(negatives):
        raise ValueError(
            f"{partition}/{cohort} cannot form a 1:1 cohort: "
            f"{len(positives)} positives for {len(negatives)} negatives"
        )
    selected_positives = sorted(
        positives,
        key=lambda row: (stable_selection_key(seed, cohort, row["pair_id"]), row["pair_id"]),
    )[: len(negatives)]
    ranks = {row["pair_id"]: str(index) for index, row in enumerate(selected_positives, start=1)}
    rows: list[dict[str, str]] = []
    for pair in negatives + selected_positives:
        is_positive = pair["label"] == "1"
        rows.append(
            {
                "cohort": cohort,
                "partition": partition,
                "pair_id": pair["pair_id"],
                "label": pair["label"],
                "evidence_class": pair["evidence_class"],
                "bootstrap_group": pair["complex_group_id"] or pair["pair_id"],
                "selection_reason": "deterministic_positive_sample" if is_positive else "all_explicit_negatives",
                "positive_selection_rank": ranks.get(pair["pair_id"], ""),
            }
        )
    return rows


def build_cohorts(pairs_path: Path, splits_path: Path, output_dir: Path, seed: int = 20260915) -> dict:
    pair_rows = read_csv(pairs_path)
    split_rows = read_csv(splits_path)
    require_fields(
        pair_rows,
        {"pair_id", "label", "evidence_class", "complex_group_id", "pair_status"},
        "pairs.csv",
    )
    require_fields(
        split_rows,
        {"split_scheme", "fold", "pair_id", "partition"},
        "splits.csv",
    )

    pairs_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(pair_rows, start=2):
        pair_id = row["pair_id"].strip()
        if not pair_id or pair_id in pairs_by_id:
            raise ValueError(f"pairs.csv:{line}: missing or duplicate pair_id {pair_id!r}")
        pairs_by_id[pair_id] = {key: value.strip() for key, value in row.items()}

    assignments: dict[tuple[str, str], list[dict[str, str]]] = {
        (partition, "all"): [] for partition in EVALUATION_PARTITIONS
    }
    seen_assignments: set[tuple[str, str]] = set()
    for line, split in enumerate(split_rows, start=2):
        if split["split_scheme"].strip() != "c3_primary" or split["fold"].strip() != "0":
            continue
        partition = split["partition"].strip()
        if partition not in EVALUATION_PARTITIONS:
            continue
        pair_id = split["pair_id"].strip()
        key = (partition, pair_id)
        if not pair_id or key in seen_assignments:
            raise ValueError(f"splits.csv:{line}: missing or duplicate C3 assignment {key!r}")
        seen_assignments.add(key)
        if pair_id not in pairs_by_id:
            raise ValueError(f"splits.csv:{line}: unknown pair_id {pair_id}")
        pair = pairs_by_id[pair_id]
        if pair["pair_status"] != "eligible" or pair["label"] not in {"0", "1"}:
            raise ValueError(f"splits.csv:{line}: {pair_id} is not an eligible labeled pair")
        assignments[(partition, "all")].append(pair)

    output_rows: list[dict[str, str]] = []
    count_summary: dict[str, dict[str, dict[str, int | float]]] = {}
    for partition in EVALUATION_PARTITIONS:
        assigned = assignments[(partition, "all")]
        if not assigned:
            raise ValueError(f"No c3_primary fold-0 rows for partition {partition}")
        positives = [row for row in assigned if row["label"] == "1"]
        negatives = [row for row in assigned if row["label"] == "0"]
        curated = [row for row in negatives if row["evidence_class"] == "curated_negative"]
        screen = [row for row in negatives if row["evidence_class"] == "screen_negative"]
        unexpected = sorted(
            {row["evidence_class"] for row in negatives}
            - {"curated_negative", "screen_negative", "compartment_negative"}
        )
        if unexpected:
            raise ValueError(f"{partition} has unsupported negative evidence classes: {unexpected}")

        full_rows = [
            {
                "cohort": COHORT_FULL,
                "partition": partition,
                "pair_id": pair["pair_id"],
                "label": pair["label"],
                "evidence_class": pair["evidence_class"],
                "bootstrap_group": pair["complex_group_id"] or pair["pair_id"],
                "selection_reason": "all_c3_explicit_labeled_pairs",
                "positive_selection_rank": "",
            }
            for pair in assigned
        ]
        cohorts = {
            COHORT_FULL: full_rows,
            COHORT_BALANCED: _cohort_rows(COHORT_BALANCED, partition, negatives, positives, seed),
            COHORT_CURATED: _cohort_rows(COHORT_CURATED, partition, curated, positives, seed),
            COHORT_SCREEN: _cohort_rows(COHORT_SCREEN, partition, screen, positives, seed),
        }
        count_summary[partition] = {}
        for cohort, rows in cohorts.items():
            labels = Counter(row["label"] for row in rows)
            count_summary[partition][cohort] = {
                "rows": len(rows),
                "positives": labels["1"],
                "negatives": labels["0"],
                "prevalence": labels["1"] / len(rows),
            }
            output_rows.extend(rows)

    output_rows.sort(key=lambda row: (row["cohort"], row["partition"], row["pair_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "cohort_membership.csv"
    write_csv(membership_path, output_rows)
    metadata = {
        "algorithm_version": ALGORITHM_VERSION,
        "status": "frozen_before_model_scoring",
        "seed": seed,
        "inputs": {
            "pairs_sha256": sha256_file(pairs_path),
            "splits_sha256": sha256_file(splits_path),
        },
        "cohorts": {
            COHORT_FULL: "All eligible labeled C3 pairs; prevalence is observed evidence-pool prevalence.",
            COHORT_BALANCED: "All explicit negatives plus an equal deterministic sample of direct positives.",
            COHORT_CURATED: "All curated negatives plus an equal deterministic sample of direct positives.",
            COHORT_SCREEN: "All screen negatives plus an equal deterministic sample of direct positives.",
        },
        "counts": count_summary,
        "outputs": {
            "cohort_membership.csv": {
                "rows": len(output_rows),
                "sha256": sha256_file(membership_path),
            }
        },
        "interpretation_limits": [
            "Balanced cohorts are fixed case-control discrimination tests, not estimates of natural PPI prevalence.",
            "Precision@K and EF values apply only to the named cohort unless an operational candidate pool is separately defined.",
            "No unobserved pair is introduced as a negative.",
        ],
    }
    (output_dir / "cohort_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    metadata = build_cohorts(args.pairs, args.splits, args.output_dir, args.seed)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
