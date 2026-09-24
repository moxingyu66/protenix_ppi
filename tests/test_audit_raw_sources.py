import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.audit_raw_sources import (
    parse_obo_terms,
    profile_huri,
    profile_intact_negative,
    profile_negatome,
)


class AuditRawSourcesTests(unittest.TestCase):
    def test_huri_profile_keeps_self_pairs_visible(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HuRI.tsv"
            path.write_text(
                "ENSG00000000001\tENSG00000000002\n"
                "ENSG00000000003\tENSG00000000003\n",
                encoding="utf-8",
            )
            profile = profile_huri(path)
            self.assertEqual(profile["rows"], 2)
            self.assertEqual(profile["self_pairs_to_quarantine"], 1)
            self.assertEqual(profile["invalid_rows"], 0)

    def test_negatome_profile_requires_four_columns_and_canonicalizes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manual_stringent.txt"
            path.write_text(
                "P00001\tP00002\t123\tMI:0018 - two hybrid\n"
                "P00002\tP00001\t124\tMI:0018 - two hybrid\n"
                "P00003-2\tP00003-2\t125\tMI:0019 - coimmunoprecipitation\n",
                encoding="utf-8",
            )
            profile = profile_negatome(path)
            self.assertEqual(profile["valid_four_column_rows"], 3)
            self.assertEqual(profile["unique_undirected_pairs"], 2)
            self.assertEqual(profile["duplicate_or_reciprocal_rows"], 1)
            self.assertEqual(profile["self_pairs_to_quarantine"], 1)
            self.assertEqual(profile["isoform_specific_rows_to_quarantine"], 1)

    def test_obo_parser_preserves_parents(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "psi-mi.obo"
            path.write_text(
                "format-version: 1.2\n\n[Term]\n"
                "id: MI:0407\nname: direct interaction\n"
                "is_a: MI:0915 ! physical association\n\n"
                "[Term]\nid: MI:0915\nname: physical association\n",
                encoding="utf-8",
            )
            terms = parse_obo_terms(path)
            self.assertEqual(terms["MI:0407"]["name"], "direct interaction")
            self.assertEqual(terms["MI:0407"]["is_a"], ["MI:0915"])

    def test_intact_profile_requires_both_human_protein_endpoints(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "human_negative.txt"
            fields = [
                "#ID(s) interactor A",
                "ID(s) interactor B",
                "Taxid interactor A",
                "Taxid interactor B",
                "Interaction detection method(s)",
                "Interaction type(s)",
                "Type(s) interactor A",
                "Type(s) interactor B",
                "Interaction identifier(s)",
                "Negative",
            ]
            rows = [
                {
                    fields[0]: "uniprotkb:P00001", fields[1]: "uniprotkb:P00002",
                    fields[2]: "taxid:9606(human)", fields[3]: "taxid:9606(human)",
                    fields[4]: "psi-mi:\"MI:0018\"(two hybrid)",
                    fields[5]: "psi-mi:\"MI:0407\"(direct interaction)",
                    fields[6]: "psi-mi:\"MI:0326\"(protein)",
                    fields[7]: "psi-mi:\"MI:0326\"(protein)",
                    fields[8]: "intact:EBI-1", fields[9]: "true",
                },
                {
                    fields[0]: "uniprotkb:P00001", fields[1]: "uniprotkb:X00001",
                    fields[2]: "taxid:9606(human)", fields[3]: "taxid:10090(mouse)",
                    fields[4]: "psi-mi:\"MI:0018\"(two hybrid)",
                    fields[5]: "psi-mi:\"MI:0407\"(direct interaction)",
                    fields[6]: "psi-mi:\"MI:0326\"(protein)",
                    fields[7]: "psi-mi:\"MI:0326\"(protein)",
                    fields[8]: "intact:EBI-2", fields[9]: "true",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            profile = profile_intact_negative(path)
            self.assertEqual(profile["rows"], 2)
            self.assertEqual(profile["both_interactors_human_rows"], 1)
            self.assertEqual(profile["both_human_protein_rows"], 1)
            self.assertEqual(profile["unique_human_protein_primary_uniprot_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
