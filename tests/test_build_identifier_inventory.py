import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import (
    InventoryKey,
    _uniprot_key,
    iter_huri,
    iter_intact_negatives,
)


class BuildIdentifierInventoryTests(unittest.TestCase):
    def test_isoform_is_quarantined_without_silent_canonicalization(self):
        item = _uniprot_key("Negatome 2.0", "v", "negative_evidence", "P12345-2")
        self.assertEqual(item.mapping_query_id, "P12345")
        self.assertEqual(item.isoform_specific, "true")
        self.assertEqual(item.mapping_status, "quarantine_isoform_specific")

    def test_huri_rejects_non_gene_identifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "huri.tsv"
            path.write_text("ENSG00000000005\tENST00000000001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid Ensembl gene"):
                list(iter_huri(path, "v"))

    def test_intact_negative_requires_both_human_proteins(self):
        fields = [
            "#ID(s) interactor A", "ID(s) interactor B", "Taxid interactor A",
            "Taxid interactor B", "Type(s) interactor A", "Type(s) interactor B", "Negative",
        ]
        rows = [
            {
                "#ID(s) interactor A": "uniprotkb:P12345",
                "ID(s) interactor B": "uniprotkb:Q12345",
                "Taxid interactor A": "taxid:9606(human)",
                "Taxid interactor B": "taxid:9606(human)",
                "Type(s) interactor A": "psi-mi:\"MI:0326\"(protein)",
                "Type(s) interactor B": "psi-mi:\"MI:0326\"(protein)",
                "Negative": "true",
            },
            {
                "#ID(s) interactor A": "uniprotkb:P99999",
                "ID(s) interactor B": "uniprotkb:Q99999",
                "Taxid interactor A": "taxid:9606(human)",
                "Taxid interactor B": "taxid:10090(mouse)",
                "Type(s) interactor A": "psi-mi:\"MI:0326\"(protein)",
                "Type(s) interactor B": "psi-mi:\"MI:0326\"(protein)",
                "Negative": "true",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "negative.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            items = list(iter_intact_negatives(path, "v"))
        self.assertEqual([item.raw_identifier for item in items], ["P12345", "Q12345"])

    def test_non_uniprot_primary_is_manual_review(self):
        fields = [
            "#ID(s) interactor A", "ID(s) interactor B", "Taxid interactor A",
            "Taxid interactor B", "Type(s) interactor A", "Type(s) interactor B", "Negative",
        ]
        row = {
            "#ID(s) interactor A": "refseq:NP_000001.1",
            "ID(s) interactor B": "uniprotkb:Q12345",
            "Taxid interactor A": "taxid:9606(human)",
            "Taxid interactor B": "taxid:9606(human)",
            "Type(s) interactor A": "MI:0326",
            "Type(s) interactor B": "MI:0326",
            "Negative": "true",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "negative.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            items = list(iter_intact_negatives(path, "v"))
        self.assertEqual(items[0].mapping_status, "manual_review_non_uniprot_primary")
        self.assertEqual(items[1].mapping_status, "mapping_candidate")


if __name__ == "__main__":
    unittest.main()
