import unittest

from protenix_ppi.scripts.extract_luck2020_screen_negatives import build_orf_mapping


class ExtractLuck2020ScreenNegativesTests(unittest.TestCase):
    def test_orf_mapping_strips_only_ensembl_version(self):
        mapping = build_orf_mapping(
            [{"orf_id": "1", "ensembl_gene_id": "ENSG00000143933.16"}]
        )
        self.assertEqual(mapping, {"1": {"ENSG00000143933"}})

    def test_orf_mapping_preserves_ambiguity(self):
        mapping = build_orf_mapping(
            [
                {"orf_id": "1", "ensembl_gene_id": "ENSG00000143933.16"},
                {"orf_id": "1", "ensembl_gene_id": "ENSG00000198668.5"},
            ]
        )
        self.assertEqual(len(mapping["1"]), 2)

    def test_invalid_gene_identifier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid ORF/Ensembl"):
            build_orf_mapping([{"orf_id": "1", "ensembl_gene_id": "ENST00000143933.1"}])


if __name__ == "__main__":
    unittest.main()
