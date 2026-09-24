import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.build_evaluation_cohorts import build_cohorts, stable_selection_key


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fixture(root: Path) -> tuple[Path, Path]:
    pairs = []
    splits = []
    for partition in ("train", "validation", "test"):
        for index in range(8):
            label = "1" if index < 4 else "0"
            evidence_class = "direct_positive"
            if index in (4, 5):
                evidence_class = "curated_negative"
            elif index in (6, 7):
                evidence_class = "screen_negative"
            pair_id = f"{partition}_{index}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "evidence_class": evidence_class,
                    "complex_group_id": "G_SHARED" if index < 2 else "",
                    "pair_status": "eligible",
                }
            )
            splits.append(
                {
                    "split_scheme": "c3_primary",
                    "fold": "0",
                    "pair_id": pair_id,
                    "partition": partition,
                }
            )
    pairs_path = root / "pairs.csv"
    splits_path = root / "splits.csv"
    write_csv(pairs_path, list(pairs[0]), pairs)
    write_csv(splits_path, list(splits[0]), splits)
    return pairs_path, splits_path


class BuildEvaluationCohortsTests(unittest.TestCase):
    def test_stable_selection_key_is_repeatable(self):
        self.assertEqual(stable_selection_key(7, "c", "p"), stable_selection_key(7, "c", "p"))
        self.assertNotEqual(stable_selection_key(7, "c", "p"), stable_selection_key(8, "c", "p"))

    def test_builds_deterministic_balanced_cohorts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, splits = fixture(root)
            first = root / "first"
            second = root / "second"
            metadata = build_cohorts(pairs, splits, first, seed=17)
            build_cohorts(pairs, splits, second, seed=17)
            self.assertEqual(
                (first / "cohort_membership.csv").read_bytes(),
                (second / "cohort_membership.csv").read_bytes(),
            )
            rows = read_csv(first / "cohort_membership.csv")
            self.assertFalse(any(row["partition"] == "train" for row in rows))
            for partition in ("validation", "test"):
                explicit = [
                    row
                    for row in rows
                    if row["partition"] == partition and row["cohort"] == "balanced_explicit_1to1"
                ]
                self.assertEqual(sum(row["label"] == "1" for row in explicit), 4)
                self.assertEqual(sum(row["label"] == "0" for row in explicit), 4)
                curated = [
                    row
                    for row in rows
                    if row["partition"] == partition and row["cohort"] == "balanced_curated_1to1"
                ]
                self.assertEqual(len(curated), 4)
                self.assertEqual({row["evidence_class"] for row in curated if row["label"] == "0"}, {"curated_negative"})
            self.assertEqual(metadata["status"], "frozen_before_model_scoring")
            saved = json.loads((first / "cohort_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["outputs"]["cohort_membership.csv"]["rows"], len(rows))

    def test_rejects_missing_negative_stratum(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, splits = fixture(root)
            rows = read_csv(pairs)
            rows = [row for row in rows if row["evidence_class"] != "screen_negative"]
            write_csv(pairs, list(rows[0]), rows)
            split_rows = [row for row in read_csv(splits) if row["pair_id"] in {item["pair_id"] for item in rows}]
            write_csv(splits, list(split_rows[0]), split_rows)
            with self.assertRaisesRegex(ValueError, "no explicit negatives"):
                build_cohorts(pairs, splits, root / "out")


if __name__ == "__main__":
    unittest.main()
