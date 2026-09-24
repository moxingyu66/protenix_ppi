import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.make_c3_split import (
    generate,
    pair_retention_decisions,
    stable_unit_interval,
)
from protenix_ppi.scripts.validate_dataset import validate_tables


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_dense_fixture(root: Path) -> tuple[Path, Path, Path]:
    proteins = []
    for index in range(1, 61):
        accession = f"P{index:05d}"
        sequence = "ACDEFGHIKL" + chr(ord("A") + index % 20)
        sequence = sequence.replace("B", "N").replace("J", "L").replace("O", "K").replace("U", "C")
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
                "homology_cluster_30": f"CL{index:03d}",
                "retrieved_at": "2026-09-15",
            }
        )

    pairs = []
    evidence = []
    evidence_index = 0
    for left in range(1, 60, 2):
        for offset in (1, 3, 5, 7):
            right = ((left + offset - 1) % 60) + 1
            if left == right:
                continue
            a, b = sorted((f"P{left:05d}", f"P{right:05d}"))
            pair_id = f"HUMAN_{a}_{b}"
            if any(item["pair_id"] == pair_id for item in pairs):
                continue
            label = "1" if evidence_index % 2 == 0 else "0"
            evidence_class = "direct_positive" if label == "1" else "screen_negative"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "uniprot_a": a,
                    "uniprot_b": b,
                    "label": label,
                    "evidence_class": evidence_class,
                    "complex_group_id": "",
                    "pair_status": "eligible",
                    "exclusion_reason": "",
                }
            )
            evidence.append(
                {
                    "evidence_id": f"E{evidence_index}",
                    "pair_id": pair_id,
                    "source_database": "fixture",
                    "source_version": "1",
                    "source_record_id": f"R{evidence_index}",
                    "publication_id": "",
                    "interaction_type_mi": "MI:0407" if label == "1" else "",
                    "detection_method_mi": "MI:0018",
                    "evidence_polarity": "positive" if label == "1" else "negative",
                    "negative_definition": "" if label == "1" else "screen negative",
                    "license": "test",
                    "retrieved_at": "2026-09-15",
                }
            )
            evidence_index += 1

    proteins_path = root / "proteins.csv"
    pairs_path = root / "pairs.csv"
    evidence_path = root / "evidence.csv"
    write_csv(proteins_path, list(proteins[0]), proteins)
    write_csv(pairs_path, list(pairs[0]), pairs)
    write_csv(evidence_path, list(evidence[0]), evidence)
    return proteins_path, pairs_path, evidence_path


class MakeC3SplitTests(unittest.TestCase):
    def test_stable_hash_is_repeatable_and_bounded(self):
        first = stable_unit_interval(42, "cluster-A")
        self.assertEqual(first, stable_unit_interval(42, "cluster-A"))
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_generated_split_is_deterministic_and_leakage_free(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, evidence = build_dense_fixture(root)
            first_dir = root / "first"
            second_dir = root / "second"
            first = generate(proteins, pairs, first_dir, seed_start=0, seed_end=999)
            second = generate(proteins, pairs, second_dir, seed_start=0, seed_end=999)
            self.assertEqual(first, second)
            self.assertEqual((first_dir / "splits.csv").read_bytes(), (second_dir / "splits.csv").read_bytes())
            self.assertGreater(first.retained_pairs, 0)
            self.assertGreater(first.quarantined_pairs, 0)
            self.assertEqual(first.retained_pairs + first.quarantined_pairs, len(read_csv(pairs)))

            validation = validate_tables(proteins, pairs, evidence, first_dir / "splits.csv")
            self.assertEqual(validation.errors, [])

            assignments = read_csv(first_dir / "splits.csv")
            pair_rows = {row["pair_id"]: row for row in read_csv(pairs)}
            nodes = {partition: set() for partition in ("train", "validation", "test")}
            for assignment in assignments:
                pair = pair_rows[assignment["pair_id"]]
                nodes[assignment["partition"]].update((pair["uniprot_a"], pair["uniprot_b"]))
            self.assertFalse(nodes["train"] & nodes["test"])
            self.assertFalse(nodes["train"] & nodes["validation"])
            self.assertFalse(nodes["validation"] & nodes["test"])

            metadata = json.loads((first_dir / "c3_split_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm_version"], "c3_hash_cluster_v0.2")
            self.assertEqual(metadata["stats"]["seed"], first.seed)

            protein_assignments = read_csv(first_dir / "protein_pool_assignments.csv")
            self.assertEqual(len(protein_assignments), 60)
            cluster_partitions = {}
            for row in protein_assignments:
                prior = cluster_partitions.setdefault(row["homology_cluster_30"], row["partition"])
                self.assertEqual(prior, row["partition"])

    def test_requires_cluster_for_every_protein(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins, pairs, _ = build_dense_fixture(root)
            rows = read_csv(proteins)
            rows[0]["homology_cluster_30"] = ""
            write_csv(proteins, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "homology_cluster_30"):
                generate(proteins, pairs, root / "output", seed_start=0, seed_end=2)

    def test_cross_pool_complex_group_is_quarantined_atomically(self):
        proteins = {
            "A": "CA",
            "B": "CB",
            "C": "CC",
            "D": "CD",
        }
        assignment = {"CA": "train", "CB": "train", "CC": "test", "CD": "test"}
        pairs = [
            {"pair_id": "AB", "uniprot_a": "A", "uniprot_b": "B", "complex_group_id": "G1"},
            {"pair_id": "CD", "uniprot_a": "C", "uniprot_b": "D", "complex_group_id": "G1"},
            {"pair_id": "AB2", "uniprot_a": "A", "uniprot_b": "B", "complex_group_id": ""},
        ]
        decisions = pair_retention_decisions(pairs, proteins, assignment)
        self.assertEqual(
            decisions["AB"], (None, "complex_group_cross_partition_quarantine")
        )
        self.assertEqual(
            decisions["CD"], (None, "complex_group_cross_partition_quarantine")
        )
        self.assertEqual(decisions["AB2"], ("train", ""))


if __name__ == "__main__":
    unittest.main()
