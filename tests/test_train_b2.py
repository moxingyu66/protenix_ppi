import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from protenix_ppi.scripts.pair_feature_bundle import (
    load_pair_feature_bundle,
    sha256_file,
    sha256_vector,
)
from protenix_ppi.scripts.evaluate_predictions import read_predictions
from protenix_ppi.scripts.train_b2 import train
from protenix_ppi.tests.test_train_b0a import build_fixture, write_validation_cohort


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_backbone_lock(path: Path) -> None:
    value = {
        "schema_version": "1.0",
        "repository_url": "https://github.com/bytedance/Protenix.git",
        "source_commit": "a" * 40,
        "package_version": "fixture",
        "model_name": "protenix_base_default_v1.0.0",
        "declared_training_cutoff": "2021-09-30",
        "checkpoint_path": "/fixture/checkpoint.pt",
        "checkpoint_sha256": "b" * 64,
        "python_version": "3.11",
        "torch_version": "fixture",
        "cuda_runtime": "fixture",
        "nvidia_driver": "fixture",
        "triangle_attention_kernel": "fixture",
        "triangle_multiplicative_kernel": "fixture",
        "g0_route": "B",
        "created_at": "2026-09-15T00:00:00+08:00",
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def write_pair_feature_bundle(
    root: Path,
    pairs_path: Path,
    splits_path: Path,
    backbone_lock_path: Path,
    dimension: int = 6,
) -> Path:
    bundle = root / "pair_bundle"
    bundle.mkdir(parents=True)
    pair_rows = {row["pair_id"]: row for row in read_csv(pairs_path)}
    assignments = [
        row
        for row in read_csv(splits_path)
        if row["split_scheme"] == "c3_primary" and row["fold"] == "0"
    ]
    assignments.sort(key=lambda row: row["pair_id"])
    features = np.arange(len(assignments) * dimension, dtype=np.float32).reshape(len(assignments), dimension) / 100
    np.save(bundle / "features.npy", features, allow_pickle=False)
    extraction_lock = {
        "schema_version": "1.0",
        "model_name": "protenix_base_default_v1.0.0",
        "declared_training_cutoff": "2021-09-30",
        "source_commit": "a" * 40,
        "backbone_lock_sha256": sha256_file(backbone_lock_path),
        "g1_manifest_sha256": "c" * 64,
        "hook_evidence_sha256": "d" * 64,
        "inference_condition_lock_sha256": "e" * 64,
        "seed_policy": "registered_fixture_seed",
        "sample_policy": "rank_zero_fixture",
        "label_blind": True,
        "swap_invariant": True,
        "swap_invariance_audit": {
            "pair_count": 3,
            "tolerance": 1e-5,
            "maximum_absolute_delta": 2e-6,
        },
        "feature_components": [
            {
                "name": "protenix_pair_z_cross_chain_mean_std",
                "source_tensor": "Protenix.get_pairformer_output:return[2]:z",
                "pooling": "cross_chain_direction_symmetrized_mean_and_std",
            }
        ],
    }
    (bundle / "feature_extraction_lock.json").write_text(json.dumps(extraction_lock), encoding="utf-8")
    hook_validation = {
        "schema_version": "1.0",
        "status": "passed",
        "lock": {
            "lock_sha256": sha256_file(bundle / "feature_extraction_lock.json"),
            "source_commit": "a" * 40,
            "hook_evidence_sha256": "d" * 64,
            "g1_manifest_sha256": "c" * 64,
            "backbone_lock_sha256": sha256_file(backbone_lock_path),
            "feature_component": "protenix_pair_z_cross_chain_mean_std",
            "tensor_shape_before_pooling": [1, 32, 32, 128],
            "c_z": 128,
        },
        "hook_evidence": {
            "source_commit": "a" * 40,
            "tensor_shape_before_pooling": [1, 32, 32, 128],
            "c_z": 128,
            "swap_pair_count": 3,
            "swap_tolerance": 1e-5,
            "swap_maximum_absolute_delta": 2e-6,
            "source_files": {
                "protenix/model/protenix.py": "f" * 64,
                "protenix/model/modules/pairformer.py": "0" * 64,
            },
        },
        "backbone_lock_sha256": sha256_file(backbone_lock_path),
        "g1_manifest_sha256": "c" * 64,
    }
    (bundle / "b2_hook_validation.json").write_text(
        json.dumps(hook_validation), encoding="utf-8"
    )
    with (bundle / "feature_index.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["row_index", "pair_id", "partition", "input_sha256", "feature_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, assignment in enumerate(assignments):
            pair_id = assignment["pair_id"]
            self_check = pair_rows[pair_id]
            writer.writerow(
                {
                    "row_index": index,
                    "pair_id": pair_id,
                    "partition": assignment["partition"],
                    "input_sha256": hashlib.sha256(pair_id.encode()).hexdigest(),
                    "feature_sha256": sha256_vector(features[index]),
                }
            )
            assert self_check["pair_status"] == "eligible"
    metadata = {
        "schema_version": "1.0",
        "model_name": "protenix_base_default_v1.0.0",
        "declared_training_cutoff": "2021-09-30",
        "checkpoint_sha256": "b" * 64,
        "feature_definition_version": "fixture_v1",
        "pair_count": len(assignments),
        "feature_dim": dimension,
        "feature_names": [f"f{index}" for index in range(dimension)],
        "dtype": "float32",
        "pairs_sha256": sha256_file(pairs_path),
        "splits_sha256": sha256_file(splits_path),
        "backbone_lock_sha256": sha256_file(backbone_lock_path),
        "feature_extraction_lock_sha256": sha256_file(bundle / "feature_extraction_lock.json"),
        "b2_hook_validation_sha256": sha256_file(bundle / "b2_hook_validation.json"),
    }
    (bundle / "feature_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return bundle


class TrainB2Tests(unittest.TestCase):
    def test_bundle_requires_passed_hook_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, pairs, _, splits = build_fixture(root)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            (bundle / "b2_hook_validation.json").unlink()
            with self.assertRaisesRegex(ValueError, "b2_hook_validation.json"):
                load_pair_feature_bundle(bundle, pairs, splits, backbone)

    def test_bundle_rejects_unpassed_hook_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, pairs, _, splits = build_fixture(root)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            report_path = bundle / "b2_hook_validation.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "failed"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            metadata_path = bundle / "feature_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["b2_hook_validation_sha256"] = sha256_file(report_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status=passed"):
                load_pair_feature_bundle(bundle, pairs, splits, backbone)

    def test_bundle_rejects_hook_validation_lock_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, pairs, _, splits = build_fixture(root)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            lock_path = bundle / "feature_extraction_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["seed_policy"] = "changed_after_validation"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            metadata_path = bundle / "feature_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["feature_extraction_lock_sha256"] = sha256_file(lock_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match feature_extraction_lock"):
                load_pair_feature_bundle(bundle, pairs, splits, backbone)

    def test_bundle_rejects_feature_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, pairs, _, splits = build_fixture(root)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            rows = read_csv(bundle / "feature_index.csv")
            rows[0]["feature_sha256"] = "0" * 64
            with (bundle / "feature_index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "feature_sha256 mismatch"):
                load_pair_feature_bundle(bundle, pairs, splits, backbone)

    def test_bundle_rejects_non_label_blind_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, pairs, _, splits = build_fixture(root)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            lock_path = bundle / "feature_extraction_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["label_blind"] = False
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label_blind=true"):
                load_pair_feature_bundle(bundle, pairs, splits, backbone)

    def test_end_to_end_training_is_deterministic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence, splits = build_fixture(root)
            cohorts = write_validation_cohort(root, pairs, splits)
            backbone = root / "backbone_lock.json"
            write_backbone_lock(backbone)
            bundle = write_pair_feature_bundle(root, pairs, splits, backbone)
            first = root / "first"
            second = root / "second"
            metadata = train(
                proteins, pairs, evidence, splits, bundle, backbone, cohorts, first, "c3_primary", "0", 42
            )
            train(proteins, pairs, evidence, splits, bundle, backbone, cohorts, second, "c3_primary", "0", 42)
            self.assertEqual(metadata["pair_feature_bundle"]["feature_dim"], 6)
            self.assertTrue((first / "model.joblib").is_file())
            self.assertEqual((first / "predictions.csv").read_bytes(), (second / "predictions.csv").read_bytes())
            self.assertEqual(len(read_predictions(first / "predictions.csv", "test")), 2)


if __name__ == "__main__":
    unittest.main()
