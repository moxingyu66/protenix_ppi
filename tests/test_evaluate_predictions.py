from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import math
import unittest

from protenix_ppi.scripts.evaluate_predictions import (
    Prediction,
    align_predictions,
    auroc,
    average_precision,
    bedroc,
    brier_score,
    enrichment_factor,
    evaluate,
    expected_calibration_error,
    precision_at_k,
    pr_auc_trapezoid,
    read_cohort_membership,
    read_predictions,
    select_cohort,
)


def predictions(labels, scores):
    return [
        Prediction(
            pair_id=f"P{index}",
            label=label,
            score=score,
            partition="test",
            evidence_class="direct_positive" if label else "curated_negative",
            bootstrap_group=f"G{index}",
        )
        for index, (label, score) in enumerate(zip(labels, scores))
    ]


def reference_bedroc_without_ties(labels, alpha=20.0):
    total = len(labels)
    positives = sum(labels)
    random_mean = (1 - math.exp(-alpha)) / (
        total * (math.exp(alpha / total) - 1)
    )
    rie = sum(
        math.exp(-alpha * rank / total)
        for rank, label in enumerate(labels, start=1)
        if label
    ) / (positives * random_mean)
    ratio = positives / total
    rie_max = (1 - math.exp(-alpha * ratio)) / (ratio * (1 - math.exp(-alpha)))
    rie_min = (1 - math.exp(alpha * ratio)) / (ratio * (1 - math.exp(alpha)))
    return (rie - rie_min) / (rie_max - rie_min)


class EvaluatePredictionsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        values = predictions([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(average_precision(values), 1.0)
        self.assertAlmostEqual(auroc(values), 1.0)
        self.assertAlmostEqual(precision_at_k(values, 2), 1.0)
        self.assertAlmostEqual(bedroc(values), 1.0)

    def test_reversed_ranking(self):
        values = predictions([1, 0], [0.1, 0.9])
        self.assertAlmostEqual(average_precision(values), 0.5)
        self.assertAlmostEqual(auroc(values), 0.0)
        self.assertAlmostEqual(bedroc(values), 0.0)

    def test_ties_are_order_independent(self):
        left = predictions([1, 0], [0.5, 0.5])
        right = list(reversed(left))
        self.assertAlmostEqual(average_precision(left), 0.5)
        self.assertAlmostEqual(average_precision(right), 0.5)
        self.assertAlmostEqual(auroc(left), 0.5)
        self.assertAlmostEqual(precision_at_k(left, 1), 0.5)
        self.assertAlmostEqual(bedroc(left), bedroc(right))

    def test_bedroc_matches_reference_formula_without_ties(self):
        labels = [1, 0, 1, 0, 0]
        values = predictions(labels, [0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertAlmostEqual(bedroc(values, 20.0), reference_bedroc_without_ties(labels))

    def test_bedroc_is_undefined_for_single_class(self):
        self.assertIsNone(bedroc(predictions([1, 1], [0.9, 0.8])))
        self.assertIsNone(bedroc(predictions([0, 0], [0.2, 0.1])))

    def test_trapezoid_is_reported_separately(self):
        values = predictions([1, 0], [0.1, 0.9])
        self.assertNotEqual(average_precision(values), pr_auc_trapezoid(values))

    def test_calibration_metrics(self):
        values = predictions([1, 0], [0.8, 0.2])
        self.assertAlmostEqual(brier_score(values), 0.04)
        self.assertAlmostEqual(expected_calibration_error(values, 10), 0.2)
        logits = predictions([1, 0], [2.0, -1.0])
        self.assertIsNone(brier_score(logits))
        self.assertIsNone(expected_calibration_error(logits))

    def test_enrichment_uses_prevalence(self):
        values = predictions([1] + [0] * 99, [1.0] + [0.0] * 99)
        self.assertAlmostEqual(enrichment_factor(values, 0.01), 100.0)

    def test_paired_comparison_and_bootstrap(self):
        candidate = predictions([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        baseline = predictions([1, 1, 0, 0], [0.6, 0.4, 0.7, 0.3])
        result = evaluate(candidate, baseline, bootstrap_replicates=100, bootstrap_seed=7)
        self.assertGreater(result["comparison"]["average_precision_absolute_delta"], 0)
        self.assertGreater(result["comparison"]["paired_bootstrap_absolute_delta"]["replicates_valid"], 0)

    def test_alignment_rejects_different_pair_sets(self):
        candidate = predictions([1, 0], [0.9, 0.1])
        baseline = predictions([1, 0], [0.8, 0.2])
        baseline[1] = Prediction("OTHER", 0, 0.2, "test", "curated_negative", "G1")
        with self.assertRaisesRegex(ValueError, "pair sets differ"):
            align_predictions(candidate, baseline)

    def test_reader_rejects_duplicate_pairs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"],
                )
                writer.writeheader()
                writer.writerow({"pair_id": "P1", "label": 1, "score": 0.9, "partition": "test", "evidence_class": "direct_positive", "bootstrap_group": "G1"})
                writer.writerow({"pair_id": "P1", "label": 1, "score": 0.8, "partition": "test", "evidence_class": "direct_positive", "bootstrap_group": "G1"})
            with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
                read_predictions(path, "test")

    def test_cohort_selection_validates_provenance_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cohorts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cohort", "partition", "pair_id", "label", "evidence_class", "bootstrap_group"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cohort": "balanced_explicit_1to1",
                        "partition": "test",
                        "pair_id": "P0",
                        "label": "1",
                        "evidence_class": "direct_positive",
                        "bootstrap_group": "G0",
                    }
                )
                writer.writerow(
                    {
                        "cohort": "balanced_explicit_1to1",
                        "partition": "test",
                        "pair_id": "P1",
                        "label": "0",
                        "evidence_class": "curated_negative",
                        "bootstrap_group": "G1",
                    }
                )
            members = read_cohort_membership(path, "balanced_explicit_1to1", "test")
            selected = select_cohort(predictions([1, 0, 1], [0.9, 0.1, 0.8]), members)
            self.assertEqual([item.pair_id for item in selected], ["P0", "P1"])

            bad = dict(members)
            bad["P1"] = type(members["P1"])("P1", 0, "screen_negative", "G1")
            with self.assertRaisesRegex(ValueError, "evidence_class mismatch"):
                select_cohort(predictions([1, 0, 1], [0.9, 0.1, 0.8]), bad)

    def test_cohort_selection_rejects_missing_prediction(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cohorts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cohort", "partition", "pair_id", "label", "evidence_class", "bootstrap_group"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cohort": "c",
                        "partition": "test",
                        "pair_id": "MISSING",
                        "label": "1",
                        "evidence_class": "direct_positive",
                        "bootstrap_group": "MISSING",
                    }
                )
            members = read_cohort_membership(path, "c", "test")
            with self.assertRaisesRegex(ValueError, "missing cohort pairs"):
                select_cohort(predictions([1], [0.9]), members)


if __name__ == "__main__":
    unittest.main()
