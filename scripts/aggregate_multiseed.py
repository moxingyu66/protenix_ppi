#!/usr/bin/env python3
"""Aggregate registered B0--B5 seeds on one frozen evaluation cohort."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from protenix_ppi.scripts.evaluate_predictions import (
    align_predictions,
    evaluate,
    read_cohort_membership,
    read_predictions,
    select_cohort,
)
from protenix_ppi.scripts.pair_feature_bundle import sha256_file


REGISTERED_SEEDS = (42, 123, 999)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260915
AGGREGATE_METRICS = (
    "average_precision",
    "pr_auc_trapezoid",
    "auroc",
    "precision_at_50",
    "precision_at_100",
    "recall_at_50",
    "recall_at_100",
    "ef_at_1_percent",
    "bedroc_alpha_20",
    "brier_score",
    "ece_10_equal_width",
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    required = {
        "seed",
        "candidate_predictions",
        "candidate_metadata",
        "baseline_predictions",
        "baseline_metadata",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"seed manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("seed manifest is empty")
    return rows


def resolve_path(value: str, manifest_path: Path, label: str) -> Path:
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    return candidate


def load_seed_metadata(path: Path, seed: int, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} metadata: {exc}") from exc
    if not isinstance(value, dict) or value.get("seed") != seed:
        raise ValueError(f"{label} metadata seed does not match manifest seed {seed}")
    return value


def assert_same_observations(reference, candidate, label: str) -> None:
    left, right = align_predictions(reference, candidate)
    for first, second in zip(left, right):
        if first.evidence_class != second.evidence_class:
            raise ValueError(f"{label} evidence_class mismatch for {first.pair_id}")


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n_seeds": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_metric_block(results: list[dict], block: str) -> dict:
    output = {}
    for metric in AGGREGATE_METRICS:
        values = [item[block].get(metric) for item in results]
        if any(value is None for value in values):
            output[metric] = None
        else:
            output[metric] = summarize([float(value) for value in values])
    return output


def aggregate(
    manifest_path: Path,
    cohort_membership_path: Path,
    cohort: str,
    partition: str = "test",
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    rows = read_manifest(manifest_path)
    parsed: dict[int, dict] = {}
    members = read_cohort_membership(cohort_membership_path, cohort, partition)
    reference_candidate = None
    reference_baseline = None
    for line, row in enumerate(rows, start=2):
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise ValueError(f"seed manifest:{line}: invalid seed") from exc
        if seed in parsed:
            raise ValueError(f"seed manifest:{line}: duplicate seed {seed}")
        candidate_path = resolve_path(row["candidate_predictions"], manifest_path, "candidate predictions")
        candidate_metadata_path = resolve_path(row["candidate_metadata"], manifest_path, "candidate metadata")
        baseline_path = resolve_path(row["baseline_predictions"], manifest_path, "baseline predictions")
        baseline_metadata_path = resolve_path(row["baseline_metadata"], manifest_path, "baseline metadata")
        candidate_metadata = load_seed_metadata(candidate_metadata_path, seed, "candidate")
        baseline_metadata = load_seed_metadata(baseline_metadata_path, seed, "baseline")
        for label, metadata, prediction_path in (
            ("candidate", candidate_metadata, candidate_path),
            ("baseline", baseline_metadata, baseline_path),
        ):
            expected_prediction_hash = str(
                metadata.get("output_sha256", {}).get("predictions", "")
            ).strip().lower()
            if expected_prediction_hash and expected_prediction_hash != sha256_file(prediction_path):
                raise ValueError(
                    f"{label} metadata prediction SHA-256 does not match manifest seed {seed}"
                )
        candidate = select_cohort(read_predictions(candidate_path, partition), members)
        baseline = select_cohort(read_predictions(baseline_path, partition), members)
        candidate, baseline = align_predictions(candidate, baseline)
        if reference_candidate is None:
            reference_candidate = candidate
            reference_baseline = baseline
        else:
            assert_same_observations(reference_candidate, candidate, f"candidate seed {seed}")
            assert_same_observations(reference_baseline, baseline, f"baseline seed {seed}")
        result = evaluate(candidate, baseline, bootstrap_replicates, bootstrap_seed)
        parsed[seed] = {
            "evaluation": result,
            "files": {
                "candidate_predictions": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
                "candidate_metadata": {"path": str(candidate_metadata_path), "sha256": sha256_file(candidate_metadata_path)},
                "baseline_predictions": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
                "baseline_metadata": {"path": str(baseline_metadata_path), "sha256": sha256_file(baseline_metadata_path)},
            },
            "methods": {
                "candidate": candidate_metadata.get("method"),
                "baseline": baseline_metadata.get("method"),
            },
        }
    if set(parsed) != set(REGISTERED_SEEDS):
        raise ValueError(
            f"seed manifest must contain exactly {list(REGISTERED_SEEDS)}; observed {sorted(parsed)}"
        )
    candidate_methods = {parsed[seed]["methods"]["candidate"] for seed in REGISTERED_SEEDS}
    baseline_methods = {parsed[seed]["methods"]["baseline"] for seed in REGISTERED_SEEDS}
    if None in candidate_methods or len(candidate_methods) != 1:
        raise ValueError("candidate method identity differs across registered seeds")
    if None in baseline_methods or len(baseline_methods) != 1:
        raise ValueError("baseline method identity differs across registered seeds")
    ordered_results = [parsed[seed]["evaluation"] for seed in REGISTERED_SEEDS]
    deltas = [
        float(item["comparison"]["average_precision_absolute_delta"])
        for item in ordered_results
    ]
    relative_deltas = [
        float(item["comparison"]["average_precision_relative_delta"])
        for item in ordered_results
    ]
    return {
        "schema_version": "1.0",
        "registered_seeds": list(REGISTERED_SEEDS),
        "partition": partition,
        "cohort": cohort,
        "cohort_membership_sha256": sha256_file(cohort_membership_path),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "per_seed": {str(seed): parsed[seed] for seed in REGISTERED_SEEDS},
        "aggregate": {
            "candidate": aggregate_metric_block(ordered_results, "candidate"),
            "baseline": aggregate_metric_block(ordered_results, "baseline"),
            "comparison": {
                "average_precision_absolute_delta": summarize(deltas),
                "average_precision_relative_delta": summarize(relative_deltas),
                "seeds_with_positive_absolute_delta": sum(value > 0 for value in deltas),
                "all_registered_seeds_positive": all(value > 0 for value in deltas),
            },
        },
        "interpretation": "Across-seed mean/SD and per-seed paired bootstrap are both required; this file does not select a best seed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort-membership", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.manifest, args.cohort_membership, args.cohort, args.partition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
