import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.aggregate_multiseed import aggregate


FIELDS = ["pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"]


def write_predictions(path: Path, positive_score: float, negative_score: float) -> None:
    rows = [
        {"pair_id": "P1", "label": 1, "score": positive_score, "partition": "test", "evidence_class": "direct_positive", "bootstrap_group": "P1"},
        {"pair_id": "P2", "label": 0, "score": negative_score, "partition": "test", "evidence_class": "curated_negative", "bootstrap_group": "P2"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path, seeds=(42, 123, 999)) -> tuple[Path, Path]:
    cohort = root / "cohorts.csv"
    with cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cohort", "partition", "pair_id", "label", "evidence_class", "bootstrap_group"])
        writer.writeheader()
        writer.writerow({"cohort": "primary", "partition": "test", "pair_id": "P1", "label": 1, "evidence_class": "direct_positive", "bootstrap_group": "P1"})
        writer.writerow({"cohort": "primary", "partition": "test", "pair_id": "P2", "label": 0, "evidence_class": "curated_negative", "bootstrap_group": "P2"})
    manifest_rows = []
    for offset, seed in enumerate(seeds):
        candidate = root / f"candidate_{seed}.csv"
        baseline = root / f"baseline_{seed}.csv"
        candidate_metadata = root / f"candidate_{seed}.json"
        baseline_metadata = root / f"baseline_{seed}.json"
        write_predictions(candidate, 0.9 - offset * 0.01, 0.1 + offset * 0.01)
        write_predictions(baseline, 0.4, 0.6)
        candidate_metadata.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "method": "candidate",
                    "output_sha256": {
                        "predictions": hashlib.sha256(candidate.read_bytes()).hexdigest()
                    },
                }
            ),
            encoding="utf-8",
        )
        baseline_metadata.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "method": "baseline",
                    "output_sha256": {
                        "predictions": hashlib.sha256(baseline.read_bytes()).hexdigest()
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "seed": seed,
                "candidate_predictions": candidate.name,
                "candidate_metadata": candidate_metadata.name,
                "baseline_predictions": baseline.name,
                "baseline_metadata": baseline_metadata.name,
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest, cohort


class AggregateMultiseedTests(unittest.TestCase):
    def test_aggregates_exact_registered_seeds(self):
        with TemporaryDirectory() as directory:
            manifest, cohort = build_fixture(Path(directory))
            result = aggregate(manifest, cohort, "primary", bootstrap_replicates=20, bootstrap_seed=7)
            self.assertEqual(result["registered_seeds"], [42, 123, 999])
            self.assertTrue(result["aggregate"]["comparison"]["all_registered_seeds_positive"])
            self.assertEqual(result["aggregate"]["candidate"]["average_precision"]["mean"], 1.0)

    def test_rejects_missing_registered_seed(self):
        with TemporaryDirectory() as directory:
            manifest, cohort = build_fixture(Path(directory), seeds=(42, 123))
            with self.assertRaisesRegex(ValueError, "exactly"):
                aggregate(manifest, cohort, "primary", bootstrap_replicates=5)

    def test_rejects_metadata_seed_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort = build_fixture(root)
            path = root / "candidate_42.json"
            path.write_text(json.dumps({"seed": 999, "method": "candidate"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata seed"):
                aggregate(manifest, cohort, "primary", bootstrap_replicates=5)

    def test_rejects_method_drift_across_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort = build_fixture(root)
            path = root / "candidate_999.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["method"] = "different_candidate"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "method identity differs"):
                aggregate(manifest, cohort, "primary", bootstrap_replicates=5)

    def test_rejects_prediction_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort = build_fixture(root)
            path = root / "candidate_42.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction SHA-256"):
                aggregate(manifest, cohort, "primary", bootstrap_replicates=5)


if __name__ == "__main__":
    unittest.main()
