#!/usr/bin/env python3
"""Train the B3 nonlinear PPI task head on frozen Protenix pair features."""

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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from protenix_ppi.scripts.evaluate_predictions import Prediction, average_precision
from protenix_ppi.scripts.pair_feature_bundle import load_pair_feature_bundle, sha256_file
from protenix_ppi.scripts.train_b0a import read_csv
from protenix_ppi.scripts.train_b2 import arrays, build_examples
from protenix_ppi.scripts.training_cohort import PRIMARY_SELECTION_COHORT, select_examples_by_cohort
from protenix_ppi.scripts.validate_dataset import validate_tables


HIDDEN_DIM = 64
ALPHA = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("balanced weights require both labels")
    return np.asarray([len(labels) / (2 * counts[label]) for label in labels], dtype=np.float64)


def make_head(seed: int, batch_size: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_DIM,),
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size=batch_size,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        tol=0.0,
        warm_start=True,
    )


def select_epoch(train_examples, validation_examples, seed: int, max_epochs: int, patience: int):
    train_x, train_y = arrays(train_examples)
    validation_x, validation_y = arrays(validation_examples)
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    validation_x = scaler.transform(validation_x)
    batch_size = min(BATCH_SIZE, len(train_y))
    head = make_head(seed, batch_size)
    weights = balanced_sample_weights(train_y)
    curve = []
    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        head.partial_fit(train_x, train_y, classes=np.asarray([0, 1]), sample_weight=weights)
        scores = head.predict_proba(validation_x)[:, 1]
        rows = [
            Prediction(str(index), int(label), float(score), "validation", "", str(index))
            for index, (label, score) in enumerate(zip(validation_y, scores))
        ]
        ap = average_precision(rows)
        if ap is None:
            raise ValueError("validation average precision is undefined")
        curve.append({"epoch": epoch, "validation_average_precision": ap})
        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_epoch <= 0:
        raise ValueError("no valid B3 epoch was selected")
    return best_epoch, curve


def fit_fixed_epochs(examples, seed: int, epochs: int):
    features, labels = arrays(examples)
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    batch_size = min(BATCH_SIZE, len(labels))
    head = make_head(seed, batch_size)
    weights = balanced_sample_weights(labels)
    for _ in range(epochs):
        head.partial_fit(transformed, labels, classes=np.asarray([0, 1]), sample_weight=weights)
    return scaler, head, batch_size


def parameter_count(feature_dim: int) -> int:
    return feature_dim * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1


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
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    selection_cohort: str = PRIMARY_SELECTION_COHORT,
) -> dict:
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
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
    selected_epoch, validation_curve = select_epoch(
        examples["train"], selection_examples, seed, max_epochs, patience
    )
    scaler, head, effective_batch_size = fit_fixed_epochs(
        examples["train"] + examples["validation"], seed, selected_epoch
    )
    test_x, _ = arrays(examples["test"])
    test_scores = head.predict_proba(scaler.transform(test_x))[:, 1]

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
    model_path = output_dir / "task_head.joblib"
    joblib.dump({"scaler": scaler, "head": head}, model_path)
    bundle_files = (
        "features.npy",
        "feature_index.csv",
        "feature_metadata.json",
        "feature_extraction_lock.json",
        "b2_hook_validation.json",
    )
    metadata = {
        "method": "B3_frozen_Protenix_nonlinear_PPI_task_head",
        "backbone_updated": False,
        "optimizer_contains_backbone_parameters": False,
        "seed": seed,
        "split_scheme": split_scheme,
        "fold": fold,
        "selection_cohort": selection_cohort,
        "selection_cohort_rows": len(selection_examples),
        "architecture": {
            "input_dim": bundle.summary.feature_dim,
            "hidden_layers": [HIDDEN_DIM],
            "activation": "relu",
            "output": "binary_logit",
            "trainable_parameter_count": parameter_count(bundle.summary.feature_dim),
        },
        "training": {
            "optimizer": "adam",
            "alpha_l2": ALPHA,
            "learning_rate": LEARNING_RATE,
            "nominal_batch_size": BATCH_SIZE,
            "effective_batch_size_final_fit": effective_batch_size,
            "class_weighting": "balanced_sample_weight",
            "maximum_epochs": max_epochs,
            "patience": patience,
            "selected_epoch": selected_epoch,
            "selection_metric": "validation_average_precision",
            "validation_curve": validation_curve,
            "final_refit": "train_plus_validation_for_selected_epoch_count",
        },
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
        "output_sha256": {
            "predictions": sha256_file(prediction_path),
            "task_head": sha256_file(model_path),
        },
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
