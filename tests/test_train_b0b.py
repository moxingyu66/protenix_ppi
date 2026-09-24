import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from protenix_ppi.scripts.embedding_bundle import load_bundle, sha256_file
from protenix_ppi.scripts.evaluate_predictions import read_predictions
from protenix_ppi.scripts.train_b0b import symmetric_embedding_features, train
from protenix_ppi.tests.test_train_b0a import build_fixture, write_validation_cohort


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_bundle(root: Path, proteins_path: Path, dimension: int = 8) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    proteins = read_csv(proteins_path)
    embeddings = np.arange(len(proteins) * dimension, dtype=np.float32).reshape(len(proteins), dimension) / 100
    np.save(bundle / "embeddings.npy", embeddings, allow_pickle=False)
    with (bundle / "embedding_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uniprot_accession", "sequence_sha256", "row_index"])
        writer.writeheader()
        for index, row in enumerate(proteins):
            writer.writerow(
                {
                    "uniprot_accession": row["uniprot_accession"],
                    "sequence_sha256": row["sequence_sha256"],
                    "row_index": index,
                }
            )
    metadata = {
        "schema_version": "1.0",
        "model_id": "fixture/esm2",
        "model_revision": "a" * 40,
        "checkpoint_files_sha256": {"model.safetensors": "b" * 64},
        "layer": 1,
        "pooling": "mean_residue_tokens_after_overlap_average",
        "max_residues_per_window": 1022,
        "window_stride": 511,
        "window_tail_policy": "fixed_stride_until_previous_window_covers_sequence_end",
        "embedding_dim": dimension,
        "dtype": "float32",
        "protein_table_sha256": sha256_file(proteins_path),
        "sequence_count": len(proteins),
        "tokenizer_version": "fixture",
        "python_version": "fixture",
        "torch_version": "fixture",
        "transformers_version": "fixture",
        "device": "cpu",
        "compute_dtype": "float32",
        "created_at": "2026-09-15T00:00:00+08:00",
    }
    (bundle / "embedding_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return bundle


class TrainB0bTests(unittest.TestCase):
    def test_embedding_pair_features_are_swap_invariant(self):
        a = np.asarray([1.0, 2.0])
        b = np.asarray([3.0, 4.0])
        np.testing.assert_allclose(
            symmetric_embedding_features(a, b, 100, 200),
            symmetric_embedding_features(b, a, 200, 100),
        )

    def test_bundle_rejects_sequence_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, _, _, _ = build_fixture(root)
            bundle = write_bundle(root, proteins)
            rows = read_csv(bundle / "embedding_index.csv")
            rows[0]["sequence_sha256"] = "0" * 64
            with (bundle / "embedding_index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "sequence hash mismatch"):
                load_bundle(bundle, proteins)

    def test_bundle_rejects_mutable_model_revision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, _, _, _ = build_fixture(root)
            bundle = write_bundle(root, proteins)
            metadata_path = bundle / "embedding_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["model_revision"] = "main"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable hexadecimal revision"):
                load_bundle(bundle, proteins)

    def test_end_to_end_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence, splits = build_fixture(root)
            cohorts = write_validation_cohort(root, pairs, splits)
            bundle = write_bundle(root, proteins)
            output = root / "output"
            metadata = train(
                proteins, pairs, evidence, splits, bundle, cohorts, output, "c3_primary", "0", 42
            )
            self.assertEqual(metadata["feature_count"], 27)
            self.assertEqual(metadata["embedding_bundle"]["embedding_dim"], 8)
            self.assertTrue((output / "model.joblib").is_file())
            rows = read_predictions(output / "predictions.csv", "test")
            self.assertEqual(len(rows), 2)

    def test_fixed_seed_is_deterministic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence, splits = build_fixture(root)
            cohorts = write_validation_cohort(root, pairs, splits)
            bundle = write_bundle(root, proteins)
            first = root / "first"
            second = root / "second"
            train(proteins, pairs, evidence, splits, bundle, cohorts, first, "c3_primary", "0", 42)
            train(proteins, pairs, evidence, splits, bundle, cohorts, second, "c3_primary", "0", 42)
            self.assertEqual((first / "predictions.csv").read_bytes(), (second / "predictions.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
