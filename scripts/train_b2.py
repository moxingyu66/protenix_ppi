#!/usr/bin/env python3
"""Train B2 from a validated, frozen Protenix pair-feature bundle."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn

from protenix_ppi.scripts.evaluate_predictions import Prediction, average_precision
from protenix_ppi.scripts.pair_feature_bundle import load_pair_feature_bundle, sha256_file
from protenix_ppi.scripts.train_b0a import DEFAULT_C_GRID, make_pipeline, read_csv
from protenix_ppi.scripts.training_cohort import PRIMARY_SELECTION_COHORT, select_examples_by_cohort
from protenix_ppi.scripts.validate_dataset import validate_tables


def build_examples(pairs_path: Path, splits_path: Path, bundle, split_scheme: str, fold: str):
    pairs = {row["pair_id"].strip(): row for row in read_csv(pairs_path)}
    examples = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    for row in read_csv(splits_path):
        if row["split_scheme"].strip() != split_scheme or row["fold"].strip() != fold:
            continue
        pair_id = row["pair_id"].strip()
        partition = row["partition"].strip()
        if partition not in examples:
            continue
        if pair_id in seen:
            raise ValueError(f"duplicate split assignment for {pair_id}")
        seen.add(pair_id)
        examples[partition].append((pairs[pair_id], bundle.vector(pair_id)))
    for partition, values in examples.items():
        labels = {int(pair["label"]) for pair, _ in values}
        if labels != {0, 1}:
            raise ValueError(f"{partition} partition must contain both labels; observed {sorted(labels)}")
    return examples


def arrays(examples):
    return np.vstack([features for _, features in examples]), np.asarray(
        [int(pair["label"]) for pair, _ in examples], dtype=np.int64
    )


def select_c(train_examples, validation_examples, c_grid, seed):
    train_x, train_y = arrays(train_examples)
    validation_x, validation_y = arrays(validation_examples)
    results = []
    for c_value in c_grid:
        model = make_pipeline(c_value, seed)
        model.fit(train_x, train_y)
        scores = model.predict_proba(validation_x)[:, 1]
        rows = [
            Prediction(str(index), int(label), float(score), "validation", "", str(index))
            for index, (label, score) in enumerate(zip(validation_y, scores))
        ]
        ap = average_precision(rows)
        if ap is None:
            raise ValueError("validation average precision is undefined")
        results.append({"C": c_value, "validation_average_precision": ap})
    selected = min(results, key=lambda item: (-item["validation_average_precision"], item["C"]))
    return selected["C"], results


def train(
    proteins_path: Path,
    pairs_path: Path,
    evidence_path: Path,
    splits_path: Path,
    feature_bundle: Path,
    backbone_lock: Path,
    cohort_membership_path: Path,
    output_dir: Path,
    split_scheme: str,
    fold: str,
    seed: int,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    selection_cohort: str = PRIMARY_SELECTION_COHORT,
) -> dict:
    validation = validate_tables(proteins_path, pairs_path, evidence_path, splits_path)
    if validation.errors:
        raise ValueError("dataset validation failed:\n- " + "\n- ".join(validation.errors))
    bundle = load_pair_feature_bundle(
        feature_bundle, pairs_path, splits_path, backbone_lock, split_scheme, fold
    )
    examples = build_examples(pairs_path, splits_path, bundle, split_scheme, fold)
    selection_examples = select_examples_by_cohort(
        examples["validation"], cohort_membership_path, selection_cohort
    )
    selected_c, validation_results = select_c(examples["train"], selection_examples, c_grid, seed)

    final_x, final_y = arrays(examples["train"] + examples["validation"])
    model = make_pipeline(selected_c, seed)
    model.fit(final_x, final_y)
    test_x, _ = arrays(examples["test"])
    test_scores = model.predict_proba(test_x)[:, 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (pair, _), score in zip(examples["test"], test_scores):
            pair_id = pair["pair_id"].strip()
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "label": pair["label"].strip(),
                    "score": format(float(score), ".17g"),
                    "partition": "test",
                    "evidence_class": pair["evidence_class"].strip(),
                    "bootstrap_group": pair["complex_group_id"].strip() or pair_id,
                }
            )
    joblib.dump(model, output_dir / "model.joblib")
    bundle_files = (
        "features.npy",
        "feature_index.csv",
        "feature_metadata.json",
        "feature_extraction_lock.json",
        "b2_hook_validation.json",
    )
    metadata = {
        "method": "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
        "seed": seed,
        "split_scheme": split_scheme,
        "fold": fold,
        "selected_C": selected_c,
        "c_grid": list(c_grid),
        "validation_results": validation_results,
        "selection_cohort": selection_cohort,
        "selection_cohort_rows": len(selection_examples),
        "pair_feature_bundle": asdict(bundle.summary),
        "pair_feature_bundle_files_sha256": {
            filename: sha256_file(feature_bundle / filename) for filename in bundle_files
        },
        "partition_counts": {partition: len(values) for partition, values in examples.items()},
        "input_sha256": {
            "proteins": sha256_file(proteins_path),
            "pairs": sha256_file(pairs_path),
            "evidence": sha256_file(evidence_path),
            "splits": sha256_file(splits_path),
            "backbone_lock": sha256_file(backbone_lock),
            "cohort_membership": sha256_file(cohort_membership_path),
        },
        "output_sha256": {"predictions": sha256_file(prediction_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--backbone-lock", type=Path, required=True)
    parser.add_argument("--cohort-membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-scheme", default="c3_primary")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = train(
        args.proteins,
        args.pairs,
        args.evidence,
        args.splits,
        args.feature_bundle,
        args.backbone_lock,
        args.cohort_membership,
        args.output_dir,
        args.split_scheme,
        args.fold,
        args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
