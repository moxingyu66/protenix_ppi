import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from protenix_ppi.scripts.apply_mmseqs_clusters import apply_clusters
from protenix_ppi.scripts.export_protein_fasta import export_fasta


def protein(accession, sequence):
    return {
        "uniprot_accession": accession,
        "gene_symbol": accession,
        "taxid": "9606",
        "sequence": sequence,
        "sequence_length": str(len(sequence)),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "uniprot_release": "2026_03",
        "sequence_version": "1",
        "is_reviewed": "true",
        "homology_cluster_30": "",
        "retrieved_at": "2026-09-15",
    }


class MmseqsClusteringIoTests(unittest.TestCase):
    def write_proteins(self, path):
        rows = [protein("P00002", "HIKLMN"), protein("P00001", "ACDEFG")]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_fasta_is_sorted_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proteins = root / "proteins.csv"
            self.write_proteins(proteins)
            export_fasta(proteins, root / "proteins.fasta", root / "fasta.json")
            self.assertEqual(
                (root / "proteins.fasta").read_text(encoding="ascii"),
                ">P00001\nACDEFG\n>P00002\nHIKLMN\n",
            )

    def test_cluster_membership_must_cover_every_protein_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proteins = root / "proteins.csv"
            self.write_proteins(proteins)
            clusters = root / "clusters.tsv"
            clusters.write_text("P00001\tP00001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "omits 1 proteins"):
                apply_clusters(proteins, clusters, root / "out.csv", root / "meta.json", "test")

    def test_clusters_are_populated_deterministically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proteins = root / "proteins.csv"
            self.write_proteins(proteins)
            clusters = root / "clusters.tsv"
            clusters.write_text("P00001\tP00001\nP00001\tP00002\n", encoding="utf-8")
            metadata = apply_clusters(
                proteins, clusters, root / "out.csv", root / "meta.json", "test-version"
            )
            self.assertEqual(metadata["counts"]["clusters"], 1)
            with (root / "out.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["homology_cluster_30"] for row in rows}, {"MMSEQ30_P00001"})


if __name__ == "__main__":
    unittest.main()
