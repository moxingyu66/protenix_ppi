import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_comparison_suite import run_suite


PREDICTION_FIELDS = [
    "pair_id",
    "label",
    "score",
    "partition",
    "evidence_class",
    "bootstrap_group",
]
COHORT_FIELDS = [
    "cohort",
    "partition",
    "pair_id",
    "label",
    "evidence_class",
    "bootstrap_group",
]
CORE_HASHES = {
    "proteins_original": "0" * 64,
    "proteins_clustered": "1" * 64,
    "pairs": "2" * 64,
    "evidence": "3" * 64,
    "splits": "4" * 64,
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    observations = [
        ("P1", 1, "direct_positive"),
        ("P2", 1, "direct_positive"),
        ("N1", 0, "curated_negative"),
        ("N2", 0, "screen_negative"),
    ]
    cohorts = root / "cohort_membership.csv"
    cohort_rows = []
    for cohort, pair_ids in (
        ("balanced_explicit_1to1", {"P1", "P2", "N1", "N2"}),
        ("full_evidence_pool", {"P1", "P2", "N1", "N2"}),
        ("balanced_curated_1to1", {"P1", "N1"}),
        ("balanced_screen_1to1", {"P2", "N2"}),
    ):
        for pair_id, label, evidence_class in observations:
            if pair_id in pair_ids:
                cohort_rows.append(
                    {
                        "cohort": cohort,
                        "partition": "test",
                        "pair_id": pair_id,
                        "label": label,
                        "evidence_class": evidence_class,
                        "bootstrap_group": pair_id,
                    }
                )
    write_csv(cohorts, COHORT_FIELDS, cohort_rows)
    cohort_hash = file_hash(cohorts)
    frozen_hashes = {
        "proteins": CORE_HASHES["proteins_clustered"],
        "pairs": CORE_HASHES["pairs"],
        "evidence": CORE_HASHES["evidence"],
        "splits": CORE_HASHES["splits"],
        "cohort_membership": cohort_hash,
    }

    freeze_manifest = root / "benchmark_freeze_manifest.json"
    freeze_manifest.write_text(
        json.dumps(
            {
                "status": "frozen_before_model_scoring",
                "inputs": {
                    "proteins.csv": {"sha256": CORE_HASHES["proteins_original"]},
                    "proteins.clustered.csv": {"sha256": CORE_HASHES["proteins_clustered"]},
                    "pairs.csv": {"sha256": CORE_HASHES["pairs"]},
                    "evidence.csv": {"sha256": CORE_HASHES["evidence"]},
                },
                "outputs": {
                    "splits/c3_primary/splits.csv": {"sha256": CORE_HASHES["splits"]},
                    "evaluation_cohorts/cohort_membership.csv": {"sha256": cohort_hash},
                },
                "validation": {"passed": True, "errors": []},
            }
        ),
        encoding="utf-8",
    )

    manifest_rows = []
    feature_lock = {"features.npy": "a" * 64}
    for seed in (42, 123, 999):
        candidate = root / f"candidate_{seed}.csv"
        baseline = root / f"baseline_{seed}.csv"
        candidate_rows = []
        baseline_rows = []
        for pair_id, label, evidence_class in observations:
            candidate_rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "score": 0.9 if label else 0.1,
                    "partition": "test",
                    "evidence_class": evidence_class,
                    "bootstrap_group": pair_id,
                }
            )
            baseline_rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "score": 0.4 if label else 0.6,
                    "partition": "test",
                    "evidence_class": evidence_class,
                    "bootstrap_group": pair_id,
                }
            )
        write_csv(candidate, PREDICTION_FIELDS, candidate_rows)
        write_csv(baseline, PREDICTION_FIELDS, baseline_rows)
        candidate_metadata = root / f"candidate_{seed}.json"
        baseline_metadata = root / f"baseline_{seed}.json"
        common = {
            "seed": seed,
            "split_scheme": "c3_primary",
            "fold": "0",
            "selection_cohort": "balanced_explicit_1to1",
            "input_sha256": frozen_hashes,
            "pair_feature_bundle_files_sha256": feature_lock,
        }
        candidate_metadata.write_text(
            json.dumps(
                {
                    **common,
                    "method": "B3_frozen_Protenix_nonlinear_PPI_task_head",
                    "output_sha256": {"predictions": file_hash(candidate)},
                }
            ),
            encoding="utf-8",
        )
        baseline_metadata.write_text(
            json.dumps(
                {
                    **common,
                    "method": "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
                    "output_sha256": {"predictions": file_hash(baseline)},
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
    manifest = root / "comparison_manifest.csv"
    write_csv(manifest, list(manifest_rows[0]), manifest_rows)
    return manifest, cohorts, freeze_manifest


class EvaluateComparisonSuiteTests(unittest.TestCase):
    def test_runs_all_mandatory_cohorts_and_support_rule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohorts, freeze = build_fixture(root)
            output = root / "suite"
            result = run_suite(
                manifest,
                cohorts,
                freeze,
                "b3_vs_b2",
                output,
                bootstrap_replicates=100,
                bootstrap_seed=7,
            )
            self.assertTrue(result["mandatory_secondary_results_complete"])
            self.assertTrue(result["primary_support"]["claim_supported"])
            self.assertEqual(len(result["cohort_results"]), 4)
            self.assertIn(
                "bedroc_alpha_20",
                result["cohort_results"]["balanced_explicit_1to1"]["aggregate"][
                    "candidate"
                ],
            )
            self.assertTrue((output / "comparison_suite.json").is_file())
            with self.assertRaisesRegex(ValueError, "output already exists"):
                run_suite(manifest, cohorts, freeze, "b3_vs_b2", output)

    def test_rejects_feature_bundle_drift_between_b3_and_b2(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohorts, freeze = build_fixture(root)
            path = root / "candidate_42.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["pair_feature_bundle_files_sha256"] = {"features.npy": "b" * 64}
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical frozen Protenix feature bundle"):
                run_suite(manifest, cohorts, freeze, "b3_vs_b2", root / "suite")

    def test_b1_vs_constant_accepts_zero_shot_and_null_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohorts, freeze = build_fixture(root)
            cohort_hash = file_hash(cohorts)
            for seed in (42, 123, 999):
                candidate_path = root / f"candidate_{seed}.json"
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
                candidate["method"] = "B1_Protenix_zero_shot"
                candidate["selection_cohort"] = None
                candidate["primary_score"] = "iptm"
                candidate["sample_selection"] = "native_rank_0"
                candidate["label_fitted_combination"] = False
                candidate["input_sha256"] = {
                    "proteins": CORE_HASHES["proteins_original"],
                    "pairs": CORE_HASHES["pairs"],
                    "evidence": CORE_HASHES["evidence"],
                    "splits": CORE_HASHES["splits"],
                }
                candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

                baseline_path = root / f"baseline_{seed}.json"
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                baseline["method"] = "C0_constant_score_reference"
                baseline["selection_cohort"] = None
                baseline["score_policy"] = "constant_tied_score"
                baseline["constant_score"] = 0.5
                baseline["seed_affects_scores"] = False
                baseline["input_sha256"] = {
                    "pairs": CORE_HASHES["pairs"],
                    "splits": CORE_HASHES["splits"],
                    "cohort_membership": cohort_hash,
                }
                prediction_path = root / f"baseline_{seed}.csv"
                prediction_rows = []
                with prediction_path.open("r", encoding="utf-8", newline="") as handle:
                    prediction_rows = list(csv.DictReader(handle))
                for row in prediction_rows:
                    row["score"] = "0.5"
                write_csv(prediction_path, PREDICTION_FIELDS, prediction_rows)
                baseline["output_sha256"] = {"predictions": file_hash(prediction_path)}
                baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            result = run_suite(
                manifest,
                cohorts,
                freeze,
                "b1_vs_constant",
                root / "suite",
                bootstrap_replicates=20,
                bootstrap_seed=7,
            )
            self.assertEqual(result["comparison"], "b1_vs_constant")

    def test_b3_vs_b1_accepts_original_and_clustered_protein_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohorts, freeze = build_fixture(root)
            for seed in (42, 123, 999):
                path = root / f"baseline_{seed}.json"
                metadata = json.loads(path.read_text(encoding="utf-8"))
                metadata["method"] = "B1_Protenix_zero_shot"
                metadata["selection_cohort"] = None
                metadata["primary_score"] = "iptm"
                metadata["sample_selection"] = "native_rank_0"
                metadata["label_fitted_combination"] = False
                metadata["input_sha256"] = {
                    "proteins": CORE_HASHES["proteins_original"],
                    "pairs": CORE_HASHES["pairs"],
                    "evidence": CORE_HASHES["evidence"],
                    "splits": CORE_HASHES["splits"],
                }
                path.write_text(json.dumps(metadata), encoding="utf-8")
            result = run_suite(
                manifest,
                cohorts,
                freeze,
                "b3_vs_b1",
                root / "suite",
                bootstrap_replicates=20,
                bootstrap_seed=7,
            )
            self.assertEqual(result["comparison"], "b3_vs_b1")

    def test_rejects_cohort_membership_not_bound_to_freeze(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohorts, freeze = build_fixture(root)
            cohorts.write_text(cohorts.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from benchmark freeze"):
                run_suite(manifest, cohorts, freeze, "b3_vs_b2", root / "suite")


if __name__ == "__main__":
    unittest.main()
