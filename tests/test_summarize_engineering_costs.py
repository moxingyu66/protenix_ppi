import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.summarize_engineering_costs import summarize


HEX64 = "a" * 64


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path, missing_pair: bool = False):
    splits = root / "splits.csv"
    write_csv(
        splits,
        [
            {"split_scheme": "c3_primary", "fold": "0", "pair_id": "P1", "partition": "test"},
            {"split_scheme": "c3_primary", "fold": "0", "pair_id": "P2", "partition": "test"},
        ],
    )
    pair_rows = []
    for seed in (42, 123, 999):
        pair_ids = ("P1",) if missing_pair and seed == 42 else ("P1", "P2")
        for pair_id in pair_ids:
            pair_rows.append(
                {
                    "method": "B1",
                    "seed": seed,
                    "stage": "inference",
                    "pair_id": pair_id,
                    "partition": "test",
                    "status": "success",
                    "gpu_count": 1,
                    "gpu_model": "fixture GPU",
                    "peak_gpu_memory_mib": 1000,
                    "wall_time_seconds": 10,
                    "preprocessing_seconds": 2,
                    "output_bytes": 100,
                    "input_residues": 200,
                    "input_sha256": HEX64,
                    "command_sha256": HEX64,
                    "stdout_sha256": HEX64,
                }
            )
    pair_costs = root / "pair_costs.csv"
    write_csv(pair_costs, pair_rows)
    run_costs = root / "run_costs.csv"
    write_csv(
        run_costs,
        [
            {
                "method": "B3",
                "seed": seed,
                "stage": "head_training",
                "status": "success",
                "gpu_count": 0,
                "gpu_model": "",
                "peak_gpu_memory_mib": "",
                "wall_time_seconds": 5,
                "output_bytes": 50,
                "command_sha256": HEX64,
                "stdout_sha256": HEX64,
            }
            for seed in (42, 123, 999)
        ],
    )
    return pair_costs, run_costs, splits


class SummarizeEngineeringCostsTests(unittest.TestCase):
    def test_complete_fixture_reports_throughput_and_gpu_hours(self):
        with TemporaryDirectory() as directory:
            pair_costs, run_costs, splits = build_fixture(Path(directory))
            result = summarize(pair_costs, run_costs, splits)
            self.assertTrue(result["experiment_execution_complete"])
            item = result["pair_costs"]["B1|seed=42|stage=inference"]
            self.assertEqual(item["successful_pairs"], 2)
            self.assertAlmostEqual(item["total_gpu_hours"], 20 / 3600)
            self.assertGreater(item["successful_pairs_per_wall_hour"], 0)

    def test_missing_pair_is_visible_and_fails_completion(self):
        with TemporaryDirectory() as directory:
            pair_costs, run_costs, splits = build_fixture(Path(directory), missing_pair=True)
            result = summarize(pair_costs, run_costs, splits)
            self.assertFalse(result["experiment_execution_complete"])
            self.assertEqual(
                result["pair_costs"]["B1|seed=42|stage=inference"]["missing_attempt_pair_ids"],
                ["P2"],
            )

    def test_invalid_digest_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pair_costs, run_costs, splits = build_fixture(root)
            with pair_costs.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["stdout_sha256"] = "bad"
            write_csv(pair_costs, rows)
            with self.assertRaisesRegex(ValueError, "invalid SHA-256"):
                summarize(pair_costs, run_costs, splits)

    def test_missing_training_seed_fails_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pair_costs, run_costs, splits = build_fixture(root)
            with run_costs.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            write_csv(run_costs, rows[:-1])
            result = summarize(pair_costs, run_costs, splits)
            self.assertFalse(result["registered_seed_coverage_complete"])
            self.assertFalse(result["experiment_execution_complete"])


if __name__ == "__main__":
    unittest.main()
