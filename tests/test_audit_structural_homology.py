import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.audit_structural_homology import (
    audit_complexes,
    build_mmseqs_command,
    eligible_hits_by_query,
    parse_hits,
    write_audited_structural_set,
)


class AuditStructuralHomologyTests(unittest.TestCase):
    def test_mmseqs_command_contains_frozen_identity_coverage_and_output_fields(self):
        command = build_mmseqs_command(
            "mmseqs",
            Path("query.fasta"),
            Path("target.fasta"),
            Path("hits.tsv"),
            Path("tmp"),
            8,
        )
        self.assertIn("--min-seq-id", command)
        self.assertIn("--cov-mode", command)
        self.assertIn("--alignment-mode", command)
        self.assertIn("--format-output", command)
        self.assertIn("qcov", command[command.index("--format-output") + 1])
        self.assertIn("tcov", command[command.index("--format-output") + 1])

    def test_hit_parser_normalizes_percent_identity_and_coverage(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hits.tsv"
            path.write_text("q1\tP12345\t35\t80\t90\t1e-20\t100\n", encoding="utf-8")
            hits = parse_hits(path)
            self.assertEqual(hits[0]["pident"], 0.35)
            self.assertEqual(hits[0]["qcov"], 0.80)
            self.assertEqual(hits[0]["tcov"], 0.90)

    def test_only_hits_meeting_identity_and_both_coverages_are_retained(self):
        hits = [
            {"query": "q1", "target": "P1", "pident": 0.30, "qcov": 0.50, "tcov": 0.50, "evalue": 1, "bits": 1},
            {"query": "q1", "target": "P2", "pident": 0.31, "qcov": 0.49, "tcov": 0.99, "evalue": 1, "bits": 2},
        ]
        retained = eligible_hits_by_query(hits, {"q1": "Q123"}, {"P1", "P2"})
        self.assertEqual([item["target"] for item in retained["q1"]], ["P1"])

    def test_direct_and_homology_overlap_are_reported_per_complex(self):
        complexes = {
            "C1": {
                "chains": [
                    {"chain": "A", "query": "C1__chain_A", "accession": "P1", "sequence_length": 100},
                    {"chain": "B", "query": "C1__chain_B", "accession": "Q1", "sequence_length": 100},
                ]
            }
        }
        query_to_accession = {"C1__chain_A": "P1", "C1__chain_B": "Q1"}
        hits = {
            "C1__chain_B": [
                {"query": "C1__chain_B", "target": "P2", "pident": 0.35, "qcov": 0.8, "tcov": 0.7, "evalue": 1e-20, "bits": 50}
            ]
        }
        result = audit_complexes(complexes, query_to_accession, {"P1", "P2"}, hits)
        self.assertTrue(result[0]["ppi_train_protein_overlap"])
        self.assertTrue(result[0]["homology_cluster_overlap"])
        self.assertFalse(result[0]["independence_passed"])

    def test_audited_set_is_a_new_copy_with_flags_from_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "structural_set.csv"
            output = root / "structural_set_audited.csv"
            fields = [
                "complex_id",
                "pdb_id",
                "biological_assembly_id",
                "protein_chain_mapping_json",
                "reference_structure",
                "reference_structure_sha256",
                "source_url",
                "retrieved_at",
                "pdb_release_date",
                "selection_rule_version",
                "bootstrap_group",
                "post_cutoff",
                "homology_audit_passed",
                "ppi_train_overlap",
            ]
            row = {
                key: "false" if key in {"homology_audit_passed", "ppi_train_overlap"} else "value"
                for key in fields
            }
            row["complex_id"] = "C1"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            write_audited_structural_set(
                source,
                output,
                [
                    {
                        "complex_id": "C1",
                        "ppi_train_protein_overlap": False,
                        "homology_cluster_overlap": False,
                        "independence_passed": True,
                    }
                ],
            )
            with output.open(encoding="utf-8", newline="") as handle:
                audited = next(csv.DictReader(handle))
            self.assertEqual(audited["homology_audit_passed"], "true")
            self.assertEqual(audited["ppi_train_overlap"], "false")
            self.assertNotEqual(source.read_bytes(), output.read_bytes())


if __name__ == "__main__":
    unittest.main()
