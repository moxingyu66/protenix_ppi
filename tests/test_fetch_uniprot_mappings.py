import tempfile
import unittest
from pathlib import Path

from protenix_ppi.scripts.fetch_uniprot_mappings import (
    _validate_release_headers,
    parse_mapping_tsv,
    read_release_evidence,
    read_request_ids,
)


class FetchUniProtMappingsTests(unittest.TestCase):
    def test_release_evidence_is_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reldate.txt"
            path.write_text(
                "UniProt Knowledgebase Release 2026_03 consists of:\n"
                "UniProtKB/Swiss-Prot Release 2026_03 of 02-Sep-2026\n"
                "UniProtKB/TrEMBL Release 2026_03 of 02-Sep-2026\n",
                encoding="utf-8",
            )
            self.assertEqual(read_release_evidence(path), ("2026_03", "02-Sep-2026"))

    def test_request_ids_must_be_sorted_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ids.txt"
            path.write_text("Q9ZZZ1\nP12345\nP12345\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                read_request_ids(path)

    def test_mapping_summary_exposes_one_to_many_and_unmapped(self):
        payload = b"From\tTo\nA\tP1\nB\tP2\nB\tP3\n"
        rows, summary = parse_mapping_tsv(payload, {"A", "B", "C"})
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["one_to_one_source_ids"], 1)
        self.assertEqual(summary["one_to_many_source_ids"], 1)
        self.assertEqual(summary["unmapped_source_ids"], 1)

    def test_release_header_drift_is_rejected(self):
        headers = {
            "x-uniprot-release": "2026_02",
            "x-uniprot-release-date": "02-September-2026",
        }
        with self.assertRaisesRegex(ValueError, "release drift"):
            _validate_release_headers(headers, "2026_03", "02-Sep-2026")


if __name__ == "__main__":
    unittest.main()
