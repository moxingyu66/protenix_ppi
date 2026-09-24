#!/usr/bin/env python3
"""Run a registered three-seed comparison over every mandatory C3 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.aggregate_multiseed import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    REGISTERED_SEEDS,
    aggregate,
    load_seed_metadata,
    read_manifest,
    resolve_path,
)
from protenix_ppi.scripts.embedding_bundle import sha256_file
from protenix_ppi.scripts.evaluate_predictions import read_predictions


PRIMARY_COHORT = "balanced_explicit_1to1"
REQUIRED_COHORTS = (
    PRIMARY_COHORT,
    "full_evidence_pool",
    "balanced_curated_1to1",
    "balanced_screen_1to1",
)
REGISTERED_COMPARISONS = {
    "b1_vs_constant": {
        "candidate_method": "B1_Protenix_zero_shot",
        "baseline_method": "C0_constant_score_reference",
        "claim": "native Protenix zero-shot signal beyond a prevalence-level tied ranking",
    },
    "b0b_vs_b0a": {
        "candidate_method": "B0b_frozen_PLM_symmetric_logistic_regression",
        "baseline_method": "B0a_symmetric_AAC_logistic_regression",
        "claim": "modern frozen sequence representation beyond amino-acid composition",
    },
    "b2_vs_b0b": {
        "candidate_method": "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
        "baseline_method": "B0b_frozen_PLM_symmetric_logistic_regression",
        "claim": "frozen Protenix structural representation beyond the strong sequence baseline",
    },
    "b2_vs_b1": {
        "candidate_method": "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
        "baseline_method": "B1_Protenix_zero_shot",
        "claim": "frozen Protenix linear probe beyond native zero-shot ipTM",
    },
    "b3_vs_b2": {
        "candidate_method": "B3_frozen_Protenix_nonlinear_PPI_task_head",
        "baseline_method": "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
        "claim": "task-specific nonlinear head beyond the frozen linear probe",
    },
    "b3_vs_b1": {
        "candidate_method": "B3_frozen_Protenix_nonlinear_PPI_task_head",
        "baseline_method": "B1_Protenix_zero_shot",
        "claim": "task-specific nonlinear head beyond native zero-shot ipTM",
    },
}

TRAINED_METHODS = {
    "B0a_symmetric_AAC_logistic_regression",
    "B0b_frozen_PLM_symmetric_logistic_regression",
    "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
    "B3_frozen_Protenix_nonlinear_PPI_task_head",
}
ORIGINAL_PROTEIN_METHODS = {"B1_Protenix_zero_shot"}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _record_hash(mapping: dict[str, Any], key: str, source: str) -> str:
    value = str(mapping.get(key, {}).get("sha256", "")).strip().lower()
    if len(value) != 64:
        raise ValueError(f"{source} lacks a valid SHA-256 for {key}")
    return value


def validate_freeze_manifest(
    freeze_manifest_path: Path, cohort_membership_path: Path
) -> dict[str, str]:
    freeze = load_json(freeze_manifest_path)
    if freeze.get("status") != "frozen_before_model_scoring":
        raise ValueError("benchmark freeze manifest has invalid status")
    outputs = freeze.get("outputs", {})
    inputs = freeze.get("inputs", {})
    expected_cohort_hash = _record_hash(
        outputs, "evaluation_cohorts/cohort_membership.csv", "freeze manifest outputs"
    )
    if expected_cohort_hash != sha256_file(cohort_membership_path):
        raise ValueError("cohort membership SHA-256 differs from benchmark freeze manifest")
    validation = freeze.get("validation", {})
    if validation.get("passed") is not True or validation.get("errors"):
        raise ValueError("benchmark freeze manifest does not prove a passing C3 validation")
    return {
        "proteins_original": _record_hash(inputs, "proteins.csv", "freeze manifest inputs"),
        "proteins_clustered": _record_hash(
            inputs, "proteins.clustered.csv", "freeze manifest inputs"
        ),
        "pairs": _record_hash(inputs, "pairs.csv", "freeze manifest inputs"),
        "evidence": _record_hash(inputs, "evidence.csv", "freeze manifest inputs"),
        "splits": _record_hash(outputs, "splits/c3_primary/splits.csv", "freeze manifest outputs"),
        "cohort_membership": expected_cohort_hash,
    }


def validate_comparison_manifest(
    comparison_manifest_path: Path,
    comparison: str,
    frozen_input_hashes: dict[str, str],
) -> dict[str, Any]:
    rule = REGISTERED_COMPARISONS[comparison]
    rows = read_manifest(comparison_manifest_path)
    seen: set[int] = set()
    feature_bundle_locks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for line, row in enumerate(rows, start=2):
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise ValueError(f"comparison manifest:{line}: invalid seed") from exc
        if seed in seen:
            raise ValueError(f"comparison manifest:{line}: duplicate seed {seed}")
        seen.add(seed)
        candidate_path = resolve_path(
            row["candidate_predictions"], comparison_manifest_path, "candidate predictions"
        )
        baseline_path = resolve_path(
            row["baseline_predictions"], comparison_manifest_path, "baseline predictions"
        )
        candidate_metadata = load_seed_metadata(
            resolve_path(row["candidate_metadata"], comparison_manifest_path, "candidate metadata"),
            seed,
            "candidate",
        )
        baseline_metadata = load_seed_metadata(
            resolve_path(row["baseline_metadata"], comparison_manifest_path, "baseline metadata"),
            seed,
            "baseline",
        )
        for role, metadata, predictions, expected_method in (
            ("candidate", candidate_metadata, candidate_path, rule["candidate_method"]),
            ("baseline", baseline_metadata, baseline_path, rule["baseline_method"]),
        ):
            if metadata.get("method") != expected_method:
                raise ValueError(
                    f"{role} seed {seed} method is not registered for {comparison}"
                )
            if metadata.get("split_scheme") != "c3_primary" or str(metadata.get("fold")) != "0":
                raise ValueError(f"{role} seed {seed} does not use c3_primary fold 0")
            if expected_method in TRAINED_METHODS:
                if metadata.get("selection_cohort") != PRIMARY_COHORT:
                    raise ValueError(
                        f"{role} seed {seed} did not select hyperparameters on {PRIMARY_COHORT}"
                    )
            elif metadata.get("selection_cohort") not in (None, ""):
                raise ValueError(f"{role} seed {seed} null/zero-shot method selected a cohort")
            inputs = metadata.get("input_sha256")
            if not isinstance(inputs, dict):
                raise ValueError(f"{role} seed {seed} lacks input_sha256 metadata")
            expected_inputs = {
                "pairs": frozen_input_hashes["pairs"],
                "splits": frozen_input_hashes["splits"],
            }
            if expected_method in TRAINED_METHODS:
                expected_inputs.update(
                    {
                        "proteins": frozen_input_hashes["proteins_clustered"],
                        "evidence": frozen_input_hashes["evidence"],
                        "cohort_membership": frozen_input_hashes["cohort_membership"],
                    }
                )
            elif expected_method in ORIGINAL_PROTEIN_METHODS:
                expected_inputs.update(
                    {
                        "proteins": frozen_input_hashes["proteins_original"],
                        "evidence": frozen_input_hashes["evidence"],
                    }
                )
            else:
                expected_inputs["cohort_membership"] = frozen_input_hashes[
                    "cohort_membership"
                ]
            for key, expected_hash in expected_inputs.items():
                if str(inputs.get(key, "")).strip().lower() != expected_hash:
                    raise ValueError(
                        f"{role} seed {seed} {key} hash differs from the frozen benchmark"
                    )
            expected_prediction_hash = str(
                metadata.get("output_sha256", {}).get("predictions", "")
            ).strip().lower()
            if expected_prediction_hash != sha256_file(predictions):
                raise ValueError(f"{role} seed {seed} prediction hash differs from metadata")
            if expected_method == "B1_Protenix_zero_shot":
                if (
                    metadata.get("primary_score") != "iptm"
                    or metadata.get("sample_selection") != "native_rank_0"
                    or metadata.get("label_fitted_combination") is not False
                ):
                    raise ValueError(
                        f"{role} seed {seed} violates the frozen B1 zero-shot score policy"
                    )
            if expected_method == "C0_constant_score_reference":
                if (
                    metadata.get("score_policy") != "constant_tied_score"
                    or float(metadata.get("constant_score", -1)) != 0.5
                    or metadata.get("seed_affects_scores") is not False
                ):
                    raise ValueError(f"{role} seed {seed} has invalid constant-reference policy")
                scores = {item.score for item in read_predictions(predictions, "test")}
                if scores != {0.5}:
                    raise ValueError(
                        f"{role} seed {seed} constant-reference predictions are not all 0.5"
                    )
        feature_bundle_locks.append(
            (
                candidate_metadata.get("pair_feature_bundle_files_sha256", {}),
                baseline_metadata.get("pair_feature_bundle_files_sha256", {}),
            )
        )

    if seen != set(REGISTERED_SEEDS):
        raise ValueError(
            f"comparison manifest must contain exactly {list(REGISTERED_SEEDS)}; observed {sorted(seen)}"
        )
    if comparison == "b3_vs_b2":
        for candidate_lock, baseline_lock in feature_bundle_locks:
            if not candidate_lock or candidate_lock != baseline_lock:
                raise ValueError("B3 and B2 must use the identical frozen Protenix feature bundle")
    return rule


def _support_summary(primary: dict[str, Any]) -> dict[str, Any]:
    comparison = primary["aggregate"]["comparison"]
    mean_delta = comparison["average_precision_absolute_delta"]["mean"]
    lower_bounds = {
        str(seed): primary["per_seed"][str(seed)]["evaluation"]["comparison"]
        ["paired_bootstrap_absolute_delta"]["lower_95"]
        for seed in REGISTERED_SEEDS
    }
    mean_positive = mean_delta > 0
    all_seed_directions_positive = comparison["all_registered_seeds_positive"] is True
    all_interval_lower_bounds_positive = all(
        value is not None and value > 0 for value in lower_bounds.values()
    )
    return {
        "rule_version": "conservative_three_seed_support_v0.1",
        "requirements": {
            "across_seed_mean_absolute_AP_delta_gt_0": mean_positive,
            "all_three_seed_AP_deltas_gt_0": all_seed_directions_positive,
            "all_three_paired_bootstrap_95pct_lower_bounds_gt_0": (
                all_interval_lower_bounds_positive
            ),
        },
        "mean_absolute_AP_delta": mean_delta,
        "per_seed_paired_bootstrap_lower_95": lower_bounds,
        "claim_supported": (
            mean_positive and all_seed_directions_positive and all_interval_lower_bounds_positive
        ),
        "interpretation": (
            "Failure means the registered comparison is not established by this conservative rule; "
            "it is not evidence that the methods are equivalent."
        ),
    }


def run_suite(
    comparison_manifest_path: Path,
    cohort_membership_path: Path,
    freeze_manifest_path: Path,
    comparison: str,
    output_dir: Path,
    *,
    partition: str = "test",
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if comparison not in REGISTERED_COMPARISONS:
        raise ValueError(f"unregistered comparison: {comparison}")
    if partition != "test":
        raise ValueError("registered comparison suites are restricted to the frozen test partition")
    if output_dir.exists():
        raise ValueError("comparison suite output already exists; use a new directory")
    frozen_input_hashes = validate_freeze_manifest(
        freeze_manifest_path, cohort_membership_path
    )
    rule = validate_comparison_manifest(
        comparison_manifest_path, comparison, frozen_input_hashes
    )

    cohort_results = {
        cohort: aggregate(
            comparison_manifest_path,
            cohort_membership_path,
            cohort,
            partition=partition,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for cohort in REQUIRED_COHORTS
    }
    suite = {
        "schema_version": "1.0",
        "comparison": comparison,
        **rule,
        "partition": partition,
        "registered_seeds": list(REGISTERED_SEEDS),
        "required_cohorts": list(REQUIRED_COHORTS),
        "benchmark_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "cohort_membership_sha256": sha256_file(cohort_membership_path),
        "comparison_manifest_sha256": sha256_file(comparison_manifest_path),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "primary_support": _support_summary(cohort_results[PRIMARY_COHORT]),
        "mandatory_secondary_results_complete": set(cohort_results) == set(REQUIRED_COHORTS),
        "cohort_results": cohort_results,
        "interpretation_limits": [
            "The primary balanced cohort is a fixed case-control discrimination test.",
            "Balanced-cohort precision and enrichment are not proteome-wide positive predictive values.",
            "No best seed is selected.",
        ],
    }
    output_dir.mkdir(parents=True)
    for cohort, result in cohort_results.items():
        (output_dir / f"{cohort}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (output_dir / "comparison_suite.json").write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort-membership", type=Path, required=True)
    parser.add_argument("--benchmark-freeze-manifest", type=Path, required=True)
    parser.add_argument("--comparison", choices=sorted(REGISTERED_COMPARISONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_suite(
        args.manifest,
        args.cohort_membership,
        args.benchmark_freeze_manifest,
        args.comparison,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
