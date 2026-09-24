import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_predictions import (
    average_precision,
    read_cohort_membership,
    read_predictions,
    select_cohort,
)
from protenix_ppi.scripts.make_constant_reference import generate_reference
from protenix_ppi.tests.test_evaluate_comparison_suite import build_fixture, file_hash, write_csv


def prepare_inputs(root: Path):
    _, cohorts, freeze = build_fixture(root)
    pairs = root / "pairs.csv"
    splits = root / "splits.csv"
    observations = [
        ("P1", "1", "direct_positive"),
        ("P2", "1", "direct_positive"),
        ("N1", "0", "curated_negative"),
        ("N2", "0", "screen_negative"),
    ]
    pair_rows = [
        {
            "pair_id": pair_id,
            "label": label,
            "evidence_class": evidence_class,
            "complex_group_id": "",
            "pair_status": "eligible",
        }
        for pair_id, label, evidence_class in observations
    ]
    split_rows = [
        {
            "split_scheme": "c3_primary",
            "fold": "0",
            "pair_id": pair_id,
            "partition": "test",
            "pm_class": "C3",
        }
        for pair_id, _, _ in observations
    ]
    write_csv(pairs, list(pair_rows[0]), pair_rows)
    write_csv(splits, list(split_rows[0]), split_rows)
    manifest = json.loads(freeze.read_text(encoding="utf-8"))
    manifest["inputs"]["pairs.csv"]["sha256"] = file_hash(pairs)
    manifest["outputs"]["splits/c3_primary/splits.csv"]["sha256"] = file_hash(splits)
    freeze.write_text(json.dumps(manifest), encoding="utf-8")
    return pairs, splits, cohorts, freeze


class MakeConstantReferenceTests(unittest.TestCase):
    def test_constant_reference_has_prevalence_average_precision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, splits, cohorts, freeze = prepare_inputs(root)
            output = root / "constant_42"
            metadata = generate_reference(pairs, splits, cohorts, freeze, output, 42)
            predictions = read_predictions(output / "predictions.csv", "test")
            members = read_cohort_membership(
                cohorts, "balanced_explicit_1to1", "test"
            )
            selected = select_cohort(predictions, members)
            self.assertAlmostEqual(average_precision(selected), 0.5)
            self.assertEqual(metadata["method"], "C0_constant_score_reference")
            self.assertFalse(metadata["seed_affects_scores"])
            with self.assertRaisesRegex(ValueError, "output already exists"):
                generate_reference(pairs, splits, cohorts, freeze, output, 42)

    def test_rejects_unregistered_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, splits, cohorts, freeze = prepare_inputs(root)
            with self.assertRaisesRegex(ValueError, "seed must be"):
                generate_reference(pairs, splits, cohorts, freeze, root / "out", 7)

    def test_rejects_split_hash_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, splits, cohorts, freeze = prepare_inputs(root)
            splits.write_text(splits.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "splits.csv differs"):
                generate_reference(pairs, splits, cohorts, freeze, root / "out", 42)


if __name__ == "__main__":
    unittest.main()
