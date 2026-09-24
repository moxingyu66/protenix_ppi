import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.apply_mmseqs_clusters import apply_clusters
from protenix_ppi.scripts.export_protein_fasta import export_fasta
from protenix_ppi.scripts.finalize_c3_benchmark import freeze, validate_mmseqs_evidence
from protenix_ppi.scripts.make_c3_split import sha256_file
from protenix_ppi.tests.test_make_c3_split import build_dense_fixture, read_csv, write_csv


def prepare_dataset(root: Path) -> Path:
    fixture_root = root / "fixture"
    fixture_root.mkdir()
    proteins, pairs, evidence = build_dense_fixture(fixture_root)
    protein_rows = read_csv(proteins)
    pair_fields = list(read_csv(pairs)[0])
    evidence_fields = list(read_csv(evidence)[0])
    pair_rows = []
    evidence_rows = []
    for left, protein_a in enumerate(protein_rows):
        for protein_b in protein_rows[left + 1 :]:
            accession_a = protein_a["uniprot_accession"]
            accession_b = protein_b["uniprot_accession"]
            pair_id = f"HUMAN_{accession_a}_{accession_b}"
            digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
            label = "1" if digest[0] % 3 else "0"
            evidence_class = "direct_positive"
            if label == "0":
                evidence_class = "curated_negative" if digest[1] % 2 == 0 else "screen_negative"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "uniprot_a": accession_a,
                    "uniprot_b": accession_b,
                    "label": label,
                    "evidence_class": evidence_class,
                    "complex_group_id": "",
                    "pair_status": "eligible",
                    "exclusion_reason": "",
                }
            )
            evidence_rows.append(
                {
                    "evidence_id": f"E{len(evidence_rows):05d}",
                    "pair_id": pair_id,
                    "source_database": "fixture",
                    "source_version": "1",
                    "source_record_id": f"R{len(evidence_rows):05d}",
                    "publication_id": "",
                    "interaction_type_mi": "MI:0407" if label == "1" else "",
                    "detection_method_mi": "MI:0018",
                    "evidence_polarity": "positive" if label == "1" else "negative",
                    "negative_definition": "" if label == "1" else "fixture explicit negative",
                    "license": "fixture",
                    "retrieved_at": "2026-09-15",
                }
            )
    write_csv(pairs, pair_fields, pair_rows)
    write_csv(evidence, evidence_fields, evidence_rows)

    dataset = root / "dataset"
    dataset.mkdir()
    original_rows = protein_rows
    for row in original_rows:
        row["homology_cluster_30"] = ""
    write_csv(dataset / "proteins.csv", list(original_rows[0]), original_rows)
    (dataset / "pairs.csv").write_bytes(pairs.read_bytes())
    (dataset / "evidence.csv").write_bytes(evidence.read_bytes())
    metadata = {
        "outputs": {
            "proteins.csv": {"sha256": sha256_file(dataset / "proteins.csv")},
            "pairs.csv": {"sha256": sha256_file(dataset / "pairs.csv")},
            "evidence.csv": {"sha256": sha256_file(dataset / "evidence.csv")},
        }
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    mmseqs = dataset / "mmseqs30"
    mmseqs.mkdir()
    export_fasta(
        dataset / "proteins.csv",
        mmseqs / "proteins.fasta",
        mmseqs / "fasta_metadata.json",
    )
    with (mmseqs / "mmseqs30_cluster.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in original_rows:
            writer.writerow([row["uniprot_accession"], row["uniprot_accession"]])
    apply_clusters(
        dataset / "proteins.csv",
        mmseqs / "mmseqs30_cluster.tsv",
        mmseqs / "proteins.clustered.csv",
        mmseqs / "mmseqs_cluster_metadata.json",
        "fixture-mmseqs",
    )
    (mmseqs / "mmseqs.log").write_text("fixture completed\n", encoding="utf-8")
    return dataset


class FinalizeC3BenchmarkTests(unittest.TestCase):
    def test_freezes_validated_split_cohorts_and_manifest_once(self):
        with TemporaryDirectory() as directory:
            dataset = prepare_dataset(Path(directory))
            manifest = freeze(dataset)
            self.assertEqual(manifest["status"], "frozen_before_model_scoring")
            self.assertTrue(manifest["validation"]["passed"])
            self.assertTrue((dataset / "splits" / "c3_primary" / "splits.csv").is_file())
            self.assertTrue((dataset / "evaluation_cohorts" / "cohort_membership.csv").is_file())
            self.assertTrue((dataset / "benchmark_freeze_manifest.json").is_file())
            with self.assertRaisesRegex(ValueError, "never regenerate"):
                freeze(dataset)

    def test_rejects_tampered_mmseqs_cluster_table(self):
        with TemporaryDirectory() as directory:
            dataset = prepare_dataset(Path(directory))
            cluster_tsv = dataset / "mmseqs30" / "mmseqs30_cluster.tsv"
            cluster_tsv.write_text(
                cluster_tsv.read_text(encoding="utf-8") + "X\tX\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cluster_tsv_sha256"):
                validate_mmseqs_evidence(dataset)


if __name__ == "__main__":
    unittest.main()
