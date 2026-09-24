import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from protenix_ppi.scripts.evaluate_predictions import read_predictions
from protenix_ppi.scripts.train_b0a import (
    amino_acid_composition,
    feature_names,
    symmetric_pair_features,
    train,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_validation_cohort(root: Path, pairs_path: Path, splits_path: Path) -> Path:
    pairs = {row["pair_id"]: row for row in read_csv(pairs_path)}
    rows = []
    for split in read_csv(splits_path):
        if split["partition"] != "validation":
            continue
        pair = pairs[split["pair_id"]]
        rows.append(
            {
                "cohort": "balanced_explicit_1to1",
                "partition": "validation",
                "pair_id": pair["pair_id"],
                "label": pair["label"],
                "evidence_class": pair["evidence_class"],
                "bootstrap_group": pair["complex_group_id"] or pair["pair_id"],
            }
        )
    path = root / "cohort_membership.csv"
    write_csv(path, list(rows[0]), rows)
    return path


def build_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    proteins = []
    sequences = [
        "AAAAAACCCC", "AAAACCCCCC", "DDDDDDEEEE", "DDDDEEEEEE",
        "FFFFFGGGGG", "FFFFGGGGGG", "HHHHHIIIII", "HHHHIIIIII",
        "KKKKKLLLLL", "KKKKLLLLLL", "MMMMMNNNNN", "MMMMNNNNNN",
    ]
    for index, sequence in enumerate(sequences, start=1):
        accession = f"P{index:05d}"
        proteins.append(
            {
                "uniprot_accession": accession,
                "gene_symbol": accession,
                "taxid": "9606",
                "sequence": sequence,
                "sequence_length": str(len(sequence)),
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "uniprot_release": "test",
                "sequence_version": "1",
                "is_reviewed": "true",
                "homology_cluster_30": f"C{index}",
                "retrieved_at": "2026-09-15",
            }
        )
    definitions = [
        ("P00001", "P00002", "1", "train"),
        ("P00003", "P00004", "0", "train"),
        ("P00005", "P00006", "1", "validation"),
        ("P00007", "P00008", "0", "validation"),
        ("P00009", "P00010", "1", "test"),
        ("P00011", "P00012", "0", "test"),
    ]
    pairs = []
    evidence = []
    splits = []
    for index, (a, b, label, partition) in enumerate(definitions, start=1):
        pair_id = f"HUMAN_{a}_{b}"
        positive = label == "1"
        pairs.append(
            {
                "pair_id": pair_id,
                "uniprot_a": a,
                "uniprot_b": b,
                "label": label,
                "evidence_class": "direct_positive" if positive else "curated_negative",
                "complex_group_id": "",
                "pair_status": "eligible",
                "exclusion_reason": "",
            }
        )
        evidence.append(
            {
                "evidence_id": f"E{index}",
                "pair_id": pair_id,
                "source_database": "fixture",
                "source_version": "1",
                "source_record_id": f"R{index}",
                "publication_id": "",
                "interaction_type_mi": "MI:0407" if positive else "",
                "detection_method_mi": "MI:0018",
                "evidence_polarity": "positive" if positive else "negative",
                "negative_definition": "" if positive else "curated negative",
                "license": "test",
                "retrieved_at": "2026-09-15",
            }
        )
        splits.append(
            {
                "split_scheme": "c3_primary",
                "fold": "0",
                "pair_id": pair_id,
                "partition": partition,
                "pm_class": "" if partition == "train" else "C3",
            }
        )
    proteins_path = root / "proteins.csv"
    pairs_path = root / "pairs.csv"
    evidence_path = root / "evidence.csv"
    splits_path = root / "splits.csv"
    write_csv(proteins_path, list(proteins[0]), proteins)
    write_csv(pairs_path, list(pairs[0]), pairs)
    write_csv(evidence_path, list(evidence[0]), evidence)
    write_csv(splits_path, list(splits[0]), splits)
    return proteins_path, pairs_path, evidence_path, splits_path


class TrainB0aTests(unittest.TestCase):
    def test_composition_is_normalized(self):
        vector = amino_acid_composition("AACD")
        self.assertAlmostEqual(float(vector.sum()), 1.0)
        self.assertAlmostEqual(float(vector[0]), 0.5)

    def test_features_are_swap_invariant(self):
        left = symmetric_pair_features("AAAACCCC", "DDDDEEEE")
        right = symmetric_pair_features("DDDDEEEE", "AAAACCCC")
        np.testing.assert_allclose(left, right)
        self.assertEqual(len(left), len(feature_names()))

    def test_end_to_end_training_writes_test_only_predictions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence, splits = build_fixture(root)
            cohorts = write_validation_cohort(root, pairs, splits)
            output = root / "output"
            metadata = train(proteins, pairs, evidence, splits, cohorts, output, "c3_primary", "0", 42)
            self.assertIn(metadata["selected_C"], metadata["c_grid"])
            self.assertEqual(metadata["feature_count"], 63)
            self.assertTrue((output / "model.joblib").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            prediction_rows = read_predictions(output / "predictions.csv", "test")
            self.assertEqual(len(prediction_rows), 2)
            self.assertTrue(all(row.partition == "test" for row in prediction_rows))
            saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["method"], "B0a_symmetric_AAC_logistic_regression")

    def test_training_is_deterministic_for_fixed_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence, splits = build_fixture(root)
            cohorts = write_validation_cohort(root, pairs, splits)
            first = root / "first"
            second = root / "second"
            train(proteins, pairs, evidence, splits, cohorts, first, "c3_primary", "0", 42)
            train(proteins, pairs, evidence, splits, cohorts, second, "c3_primary", "0", 42)
            self.assertEqual((first / "predictions.csv").read_bytes(), (second / "predictions.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
