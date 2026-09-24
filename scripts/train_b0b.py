#!/usr/bin/env python3
"""Train B0b from a validated frozen per-protein embedding bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn

from protenix_ppi.scripts.embedding_bundle import load_bundle, sha256_file
from protenix_ppi.scripts.evaluate_predictions import Prediction, average_precision
from protenix_ppi.scripts.train_b0a import DEFAULT_C_GRID, make_pipeline, read_csv
from protenix_ppi.scripts.training_cohort import PRIMARY_SELECTION_COHORT, select_examples_by_cohort
from protenix_ppi.scripts.validate_dataset import validate_tables


def symmetric_embedding_features(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    length_a: int,
    length_b: int,
) -> np.ndarray:
    if embedding_a.shape != embedding_b.shape or embedding_a.ndim != 1:
        raise ValueError("pair embeddings must be one-dimensional and have matching shape")
    if length_a <= 0 or length_b <= 0:
        raise ValueError("protein lengths must be positive")
    log_a = math.log1p(length_a)
    log_b = math.log1p(length_b)
    return np.concatenate(
        [
            (embedding_a + embedding_b) / 2,
            np.abs(embedding_a - embedding_b),
            embedding_a * embedding_b,
            np.asarray(
                [(log_a + log_b) / 2, abs(log_a - log_b), min(length_a, length_b) / max(length_a, length_b)],
                dtype=np.float64,
            ),
        ]
    )


def build_examples(
    proteins_path: Path,
    pairs_path: Path,
    splits_path: Path,
    embeddings: dict[str, np.ndarray],
    split_scheme: str,
    fold: str,
) -> dict[str, list[tuple[dict[str, str], np.ndarray]]]:
    proteins = {row["uniprot_accession"].strip(): row for row in read_csv(proteins_path)}
    pairs = {row["pair_id"].strip(): row for row in read_csv(pairs_path)}
    examples: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {
        "train": [], "validation": [], "test": []
    }
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
        pair = pairs[pair_id]
        a = pair["uniprot_a"].strip()
        b = pair["uniprot_b"].strip()
        features = symmetric_embedding_features(
            embeddings[a],
            embeddings[b],
            int(proteins[a]["sequence_length"]),
            int(proteins[b]["sequence_length"]),
        )
        examples[partition].append((pair, features))
    for partition, values in examples.items():
        labels = {int(pair["label"]) for pair, _ in values}
        if labels != {0, 1}:
            raise ValueError(f"{partition} partition must contain both labels; observed {sorted(labels)}")
    return examples


def arrays(examples: list[tuple[dict[str, str], np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([value for _, value in examples]), np.asarray(
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
            Prediction(str(i), int(label), float(score), "validation", "", str(i))
            for i, (label, score) in enumerate(zip(validation_y, scores))
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
    embedding_bundle: Path,
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
    embeddings, bundle_summary = load_bundle(embedding_bundle, proteins_path)
    examples = build_examples(proteins_path, pairs_path, splits_path, embeddings, split_scheme, fold)
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
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
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
    metadata = {
        "method": "B0b_frozen_PLM_symmetric_logistic_regression",
        "seed": seed,
        "split_scheme": split_scheme,
        "fold": fold,
        "selected_C": selected_c,
        "c_grid": list(c_grid),
        "validation_results": validation_results,
        "selection_cohort": selection_cohort,
        "selection_cohort_rows": len(selection_examples),
        "embedding_bundle": asdict(bundle_summary),
        "embedding_bundle_files_sha256": {
            filename: sha256_file(embedding_bundle / filename)
            for filename in ("embeddings.npy", "embedding_index.csv", "embedding_metadata.json")
        },
        "pair_feature": "embedding_mean_absdiff_product_plus_symmetric_lengths",
        "feature_count": bundle_summary.embedding_dim * 3 + 3,
        "partition_counts": {partition: len(values) for partition, values in examples.items()},
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
    parser.add_argument("--embedding-bundle", type=Path, required=True)
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
        args.embedding_bundle,
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
