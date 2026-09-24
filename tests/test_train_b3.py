from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from protenix_ppi.scripts.evaluate_predictions import read_predictions
from protenix_ppi.scripts.train_b3 import balanced_sample_weights, parameter_count, train
from protenix_ppi.tests.test_train_b0a import build_fixture, write_validation_cohort
from protenix_ppi.tests.test_train_b2 import write_backbone_lock, write_pair_feature_bundle


class TrainB3Tests(unittest.TestCase):
    def test_balanced_sample_weights_equalize_class_mass(self):
        labels = np.asarray([1, 1, 1, 0])
        weights = balanced_sample_weights(labels)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), float(weights[labels == 0].sum()))

    def test_parameter_count(self):
        self.assertEqual(parameter_count(6), 6 * 64 + 64 + 64 + 1)

    def test_end_to_end_is_deterministic_and_backbone_free(self):
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
                proteins,
                pairs,
                evidence,
                splits,
                bundle,
                backbone,
                cohorts,
                first,
                "c3_primary",
                "0",
                42,
                max_epochs=8,
                patience=3,
            )
            train(
                proteins,
                pairs,
                evidence,
                splits,
                bundle,
                backbone,
                cohorts,
                second,
                "c3_primary",
                "0",
                42,
                max_epochs=8,
                patience=3,
            )
            self.assertFalse(metadata["backbone_updated"])
            self.assertFalse(metadata["optimizer_contains_backbone_parameters"])
            self.assertGreater(metadata["training"]["selected_epoch"], 0)
            self.assertEqual((first / "predictions.csv").read_bytes(), (second / "predictions.csv").read_bytes())
            self.assertEqual(len(read_predictions(first / "predictions.csv", "test")), 2)


if __name__ == "__main__":
    unittest.main()
