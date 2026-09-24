#!/usr/bin/env python3
"""Create a frozen-test constant-score null reference for B1 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from protenix_ppi.scripts.embedding_bundle import sha256_file
from protenix_ppi.scripts.evaluate_comparison_suite import validate_freeze_manifest


METHOD = "C0_constant_score_reference"
CONSTANT_SCORE = 0.5
REGISTERED_SEEDS = (42, 123, 999)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader)


def generate_reference(
    pairs_path: Path,
    splits_path: Path,
    cohort_membership_path: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed must be one of {list(REGISTERED_SEEDS)}")
    if output_dir.exists():
        raise ValueError("constant-reference output already exists; use a new directory")
    frozen = validate_freeze_manifest(freeze_manifest_path, cohort_membership_path)
    if sha256_file(pairs_path) != frozen["pairs"]:
        raise ValueError("pairs.csv differs from benchmark freeze manifest")
    if sha256_file(splits_path) != frozen["splits"]:
        raise ValueError("splits.csv differs from benchmark freeze manifest")

    pairs = read_csv(pairs_path)
    pair_by_id: dict[str, dict[str, str]] = {}
    required_pair_fields = {
        "pair_id",
        "label",
        "evidence_class",
        "complex_group_id",
        "pair_status",
    }
    if not pairs or required_pair_fields - set(pairs[0]):
        raise ValueError("pairs.csv lacks fields required for the constant reference")
    for line, row in enumerate(pairs, start=2):
        pair_id = row["pair_id"].strip()
        if not pair_id or pair_id in pair_by_id:
            raise ValueError(f"pairs.csv:{line}: missing or duplicate pair_id")
        pair_by_id[pair_id] = row

    test_ids: set[str] = set()
    for line, row in enumerate(read_csv(splits_path), start=2):
        if row.get("split_scheme", "").strip() != "c3_primary" or row.get("fold", "").strip() != "0":
            continue
        if row.get("partition", "").strip() != "test":
            continue
        pair_id = row.get("pair_id", "").strip()
        if not pair_id or pair_id in test_ids:
            raise ValueError(f"splits.csv:{line}: missing or duplicate C3 test pair")
        if pair_id not in pair_by_id:
            raise ValueError(f"splits.csv:{line}: unknown pair_id {pair_id}")
        pair = pair_by_id[pair_id]
        if pair["pair_status"].strip() != "eligible" or pair["label"].strip() not in {"0", "1"}:
            raise ValueError(f"splits.csv:{line}: pair is not eligible and labeled")
        test_ids.add(pair_id)
    if not test_ids:
        raise ValueError("splits.csv has no c3_primary fold-0 test rows")

    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.csv"
    fields = ["pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"]
    with predictions_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair_id in sorted(test_ids):
            pair = pair_by_id[pair_id]
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "label": pair["label"].strip(),
                    "score": format(CONSTANT_SCORE, ".17g"),
                    "partition": "test",
                    "evidence_class": pair["evidence_class"].strip(),
                    "bootstrap_group": pair["complex_group_id"].strip() or pair_id,
                }
            )
    metadata = {
        "schema_version": "1.0",
        "method": METHOD,
        "seed": seed,
        "seed_affects_scores": False,
        "split_scheme": "c3_primary",
        "fold": "0",
        "selection_cohort": None,
        "score_policy": "constant_tied_score",
        "constant_score": CONSTANT_SCORE,
        "test_pair_count": len(test_ids),
        "expected_average_precision": "evaluated_cohort_prevalence",
        "input_sha256": {
            "pairs": sha256_file(pairs_path),
            "splits": sha256_file(splits_path),
            "cohort_membership": sha256_file(cohort_membership_path),
        },
        "benchmark_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "output_sha256": {"predictions": sha256_file(predictions_path)},
        "interpretation": (
            "All candidates are tied. Tie-aware AP equals cohort prevalence and no arbitrary "
            "random row order is presented as a scientific baseline."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--cohort-membership", type=Path, required=True)
    parser.add_argument("--benchmark-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    metadata = generate_reference(
        args.pairs,
        args.splits,
        args.cohort_membership,
        args.benchmark_freeze_manifest,
        args.output_dir,
        args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
