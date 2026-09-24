#!/usr/bin/env python3
"""Train the leakage-controlled B0a symmetric sequence-composition baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protenix_ppi.scripts.evaluate_predictions import Prediction, average_precision
from protenix_ppi.scripts.training_cohort import PRIMARY_SELECTION_COHORT, select_examples_by_cohort
from protenix_ppi.scripts.validate_dataset import validate_tables


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amino_acid_composition(sequence: str) -> np.ndarray:
    sequence = sequence.strip().upper()
    counts = Counter(sequence)
    denominator = sum(counts[residue] for residue in STANDARD_AA)
    if denominator == 0:
        raise ValueError("sequence contains no standard amino-acid residues")
    return np.asarray([counts[residue] / denominator for residue in STANDARD_AA], dtype=np.float64)


def symmetric_pair_features(sequence_a: str, sequence_b: str) -> np.ndarray:
    composition_a = amino_acid_composition(sequence_a)
    composition_b = amino_acid_composition(sequence_b)
    log_length_a = math.log1p(len(sequence_a))
    log_length_b = math.log1p(len(sequence_b))
    length_ratio = min(len(sequence_a), len(sequence_b)) / max(len(sequence_a), len(sequence_b))
    return np.concatenate(
        [
            (composition_a + composition_b) / 2,
            np.abs(composition_a - composition_b),
            composition_a * composition_b,
            np.asarray(
                [
                    (log_length_a + log_length_b) / 2,
                    abs(log_length_a - log_length_b),
                    length_ratio,
                ],
                dtype=np.float64,
            ),
        ]
    )


def feature_names() -> list[str]:
    return (
        [f"aac_mean_{residue}" for residue in STANDARD_AA]
        + [f"aac_absdiff_{residue}" for residue in STANDARD_AA]
        + [f"aac_product_{residue}" for residue in STANDARD_AA]
        + ["log_length_mean", "log_length_absdiff", "length_min_max_ratio"]
    )


def make_pipeline(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_examples(
    proteins_path: Path,
    pairs_path: Path,
    splits_path: Path,
    split_scheme: str,
    fold: str,
) -> dict[str, list[tuple[dict[str, str], np.ndarray]]]:
    proteins = {row["uniprot_accession"].strip(): row for row in read_csv(proteins_path)}
    pairs = {row["pair_id"].strip(): row for row in read_csv(pairs_path)}
    examples: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {
        "train": [], "validation": [], "test": []
    }
    seen_assignments: set[str] = set()
    for row in read_csv(splits_path):
        if row["split_scheme"].strip() != split_scheme or row["fold"].strip() != fold:
            continue
        pair_id = row["pair_id"].strip()
        partition = row["partition"].strip()
        if partition not in examples:
            continue
        if pair_id in seen_assignments:
            raise ValueError(f"duplicate split assignment for {pair_id}")
        seen_assignments.add(pair_id)
        pair = pairs[pair_id]
        if pair["pair_status"].strip() != "eligible" or pair["label"].strip() not in {"0", "1"}:
            raise ValueError(f"split contains ineligible or unlabeled pair {pair_id}")
        sequence_a = proteins[pair["uniprot_a"].strip()]["sequence"]
        sequence_b = proteins[pair["uniprot_b"].strip()]["sequence"]
        examples[partition].append((pair, symmetric_pair_features(sequence_a, sequence_b)))
    for partition, values in examples.items():
        labels = {int(pair["label"]) for pair, _ in values}
        if labels != {0, 1}:
            raise ValueError(f"{partition} partition must contain both labels; observed {sorted(labels)}")
    return examples


def arrays(examples: list[tuple[dict[str, str], np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([features for _, features in examples]),
        np.asarray([int(pair["label"]) for pair, _ in examples], dtype=np.int64),
    )


def select_c(
    train_examples: list[tuple[dict[str, str], np.ndarray]],
    validation_examples: list[tuple[dict[str, str], np.ndarray]],
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    train_x, train_y = arrays(train_examples)
    validation_x, validation_y = arrays(validation_examples)
    results: list[dict[str, float]] = []
    for c_value in c_grid:
        model = make_pipeline(c_value, seed)
        model.fit(train_x, train_y)
        scores = model.predict_proba(validation_x)[:, 1]
        prediction_rows = [
            Prediction(
                pair_id=str(index),
                label=int(label),
                score=float(score),
                partition="validation",
                evidence_class="",
                bootstrap_group=str(index),
            )
            for index, (label, score) in enumerate(zip(validation_y, scores))
        ]
        ap = average_precision(prediction_rows)
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
    examples = build_examples(proteins_path, pairs_path, splits_path, split_scheme, fold)
    selection_examples = select_examples_by_cohort(
        examples["validation"], cohort_membership_path, selection_cohort
    )
    selected_c, validation_results = select_c(examples["train"], selection_examples, c_grid, seed)
    final_examples = examples["train"] + examples["validation"]
    final_x, final_y = arrays(final_examples)
    model = make_pipeline(selected_c, seed)
    model.fit(final_x, final_y)

    test_x, _ = arrays(examples["test"])
    test_scores = model.predict_proba(test_x)[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    metadata = {
        "method": "B0a_symmetric_AAC_logistic_regression",
        "seed": seed,
        "split_scheme": split_scheme,
        "fold": fold,
        "selected_C": selected_c,
        "c_grid": list(c_grid),
        "validation_results": validation_results,
        "selection_cohort": selection_cohort,
        "selection_cohort_rows": len(selection_examples),
        "feature_names": feature_names(),
        "feature_count": len(feature_names()),
        "partition_counts": {partition: len(values) for partition, values in examples.items()},
        "partition_positive_counts": {
            partition: sum(int(pair["label"]) for pair, _ in values)
            for partition, values in examples.items()
        },
        "input_sha256": {
            "proteins": sha256_file(proteins_path),
            "pairs": sha256_file(pairs_path),
            "evidence": sha256_file(evidence_path),
            "splits": sha256_file(splits_path),
            "cohort_membership": sha256_file(cohort_membership_path),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "interpretation_warning": "B0a is a composition sanity baseline; B0b frozen protein-language-model embeddings remain required for H2.",
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
