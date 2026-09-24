import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.validate_dataset import validate_tables


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protein(accession: str, sequence: str, cluster: str) -> dict[str, str]:
    return {
        "uniprot_accession": accession,
        "gene_symbol": accession,
        "taxid": "9606",
        "sequence": sequence,
        "sequence_length": str(len(sequence)),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "uniprot_release": "2026_01",
        "sequence_version": "1",
        "is_reviewed": "true",
        "homology_cluster_30": cluster,
        "retrieved_at": "2026-09-15",
    }


def build_tables(root: Path):
    proteins = [
        protein("P00001", "ACDEFG", "C1"),
        protein("P00002", "HIKLMN", "C2"),
        protein("P00003", "PQRSTV", "C3"),
        protein("P00004", "WYACDE", "C4"),
        protein("P00005", "FGHIKL", "C5"),
        protein("P00006", "MNPQRS", "C6"),
    ]
    pairs = [
        {
            "pair_id": "HUMAN_P00001_P00002", "uniprot_a": "P00001", "uniprot_b": "P00002",
            "label": "1", "evidence_class": "direct_positive", "complex_group_id": "G1",
            "pair_status": "eligible", "exclusion_reason": "",
        },
        {
            "pair_id": "HUMAN_P00003_P00004", "uniprot_a": "P00003", "uniprot_b": "P00004",
            "label": "0", "evidence_class": "curated_negative", "complex_group_id": "",
            "pair_status": "eligible", "exclusion_reason": "",
        },
        {
            "pair_id": "HUMAN_P00005_P00006", "uniprot_a": "P00005", "uniprot_b": "P00006",
            "label": "0", "evidence_class": "screen_negative", "complex_group_id": "",
            "pair_status": "eligible", "exclusion_reason": "",
        },
    ]
    evidence = [
        {
            "evidence_id": "E1", "pair_id": "HUMAN_P00001_P00002", "source_database": "HuRI",
            "source_version": "test", "source_record_id": "R1", "publication_id": "PMID:1",
            "interaction_type_mi": "MI:0407", "detection_method_mi": "MI:0018",
            "evidence_polarity": "positive", "negative_definition": "", "license": "CC-BY-4.0",
            "retrieved_at": "2026-09-15",
        },
        {
            "evidence_id": "E2", "pair_id": "HUMAN_P00003_P00004", "source_database": "Negatome",
            "source_version": "test", "source_record_id": "R2", "publication_id": "PMID:2",
            "interaction_type_mi": "", "detection_method_mi": "MI:0090",
            "evidence_polarity": "negative", "negative_definition": "curated experimental non-interaction",
            "license": "academic", "retrieved_at": "2026-09-15",
        },
        {
            "evidence_id": "E3", "pair_id": "HUMAN_P00005_P00006", "source_database": "HuRI",
            "source_version": "test", "source_record_id": "R3", "publication_id": "PMID:3",
            "interaction_type_mi": "", "detection_method_mi": "MI:0018",
            "evidence_polarity": "negative", "negative_definition": "explicitly tested negative in screen",
            "license": "CC-BY-4.0", "retrieved_at": "2026-09-15",
        },
    ]
    splits = [
        {"split_scheme": "c3_primary", "fold": "0", "pair_id": "HUMAN_P00001_P00002", "partition": "train", "pm_class": ""},
        {"split_scheme": "c3_primary", "fold": "0", "pair_id": "HUMAN_P00003_P00004", "partition": "validation", "pm_class": ""},
        {"split_scheme": "c3_primary", "fold": "0", "pair_id": "HUMAN_P00005_P00006", "partition": "test", "pm_class": "C3"},
        {"split_scheme": "homology_primary", "fold": "0", "pair_id": "HUMAN_P00001_P00002", "partition": "train", "pm_class": ""},
        {"split_scheme": "homology_primary", "fold": "0", "pair_id": "HUMAN_P00003_P00004", "partition": "validation", "pm_class": ""},
        {"split_scheme": "homology_primary", "fold": "0", "pair_id": "HUMAN_P00005_P00006", "partition": "test", "pm_class": ""},
    ]
    write_csv(root / "proteins.csv", list(proteins[0]), proteins)
    write_csv(root / "pairs.csv", list(pairs[0]), pairs)
    write_csv(root / "evidence.csv", list(evidence[0]), evidence)
    write_csv(root / "splits.csv", list(splits[0]), splits)


class ValidateDatasetTests(unittest.TestCase):
    def evaluate(self, root: Path):
        return validate_tables(root / "proteins.csv", root / "pairs.csv", root / "evidence.csv", root / "splits.csv")

    def test_valid_normalized_tables_pass(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            report = self.evaluate(root)
            self.assertEqual(report.errors, [])
            self.assertEqual(report.counts["pairs"], 3)

    def test_reversed_pair_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            rows = read_rows(root / "pairs.csv")
            rows[0]["uniprot_a"], rows[0]["uniprot_b"] = rows[0]["uniprot_b"], rows[0]["uniprot_a"]
            write_csv(root / "pairs.csv", list(rows[0]), rows)
            report = self.evaluate(root)
            self.assertTrue(any("lexicographically ordered" in error for error in report.errors))

    def test_c3_protein_overlap_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            rows = read_rows(root / "pairs.csv")
            rows[2]["uniprot_a"] = "P00001"
            rows[2]["pair_id"] = "HUMAN_P00001_P00006"
            write_csv(root / "pairs.csv", list(rows[0]), rows)
            split_rows = read_rows(root / "splits.csv")
            for row in split_rows:
                if row["pair_id"] == "HUMAN_P00005_P00006":
                    row["pair_id"] = "HUMAN_P00001_P00006"
            write_csv(root / "splits.csv", list(split_rows[0]), split_rows)
            evidence_rows = read_rows(root / "evidence.csv")
            evidence_rows[2]["pair_id"] = "HUMAN_P00001_P00006"
            write_csv(root / "evidence.csv", list(evidence_rows[0]), evidence_rows)
            report = self.evaluate(root)
            self.assertTrue(any("protein leakage" in error for error in report.errors))

    def test_homology_cluster_overlap_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            rows = read_rows(root / "proteins.csv")
            rows[4]["homology_cluster_30"] = "C1"
            write_csv(root / "proteins.csv", list(rows[0]), rows)
            report = self.evaluate(root)
            self.assertTrue(any("homology cluster leakage" in error for error in report.errors))

    def test_negative_without_definition_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            rows = read_rows(root / "evidence.csv")
            rows[1]["negative_definition"] = ""
            write_csv(root / "evidence.csv", list(rows[0]), rows)
            report = self.evaluate(root)
            self.assertTrue(any("negative evidence definition" in error for error in report.errors))

    def test_quarantined_self_pair_is_preserved_without_primary_scope_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tables(root)
            pair_rows = read_rows(root / "pairs.csv")
            pair_rows.append(
                {
                    "pair_id": "HUMAN_P00001_P00001", "uniprot_a": "P00001", "uniprot_b": "P00001",
                    "label": "1", "evidence_class": "direct_positive", "complex_group_id": "",
                    "pair_status": "quarantine", "exclusion_reason": "self_pair",
                }
            )
            write_csv(root / "pairs.csv", list(pair_rows[0]), pair_rows)
            evidence_rows = read_rows(root / "evidence.csv")
            evidence_rows.append(
                {
                    "evidence_id": "E4", "pair_id": "HUMAN_P00001_P00001", "source_database": "HuRI",
                    "source_version": "test", "source_record_id": "R4", "publication_id": "PMID:4",
                    "interaction_type_mi": "MI:0407", "detection_method_mi": "MI:0018",
                    "evidence_polarity": "positive", "negative_definition": "", "license": "CC-BY-4.0",
                    "retrieved_at": "2026-09-15",
                }
            )
            write_csv(root / "evidence.csv", list(evidence_rows[0]), evidence_rows)
            report = self.evaluate(root)
            self.assertEqual(report.errors, [])


if __name__ == "__main__":
    unittest.main()
