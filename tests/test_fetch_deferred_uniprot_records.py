import tempfile
import unittest
from pathlib import Path

from protenix_ppi.scripts.fetch_deferred_uniprot_records import (
    deferred_targets,
    parse_record_tsv,
)


class FetchDeferredUniProtRecordsTests(unittest.TestCase):
    def test_only_deferred_targets_are_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "resolution.tsv"
            path.write_text(
                "resolution_status\tall_uniprot_targets\n"
                "accepted\tP1\n"
                "deferred\tA0A1;A0A2\n"
                "deferred\tA0A2;A0A3\n",
                encoding="utf-8",
            )
            self.assertEqual(deferred_targets(path), ["A0A1", "A0A2", "A0A3"])

    def test_record_parser_requires_entry(self):
        with self.assertRaisesRegex(ValueError, "no Entry"):
            parse_record_tsv(b"From\tTo\nA\tB\n")

    def test_record_parser_returns_rows(self):
        header, rows = parse_record_tsv(
            b"Entry\tReviewed\tOrganism (ID)\nA0A1\tunreviewed\t9606\n"
        )
        self.assertEqual(header[0], "Entry")
        self.assertEqual(rows[0]["Entry"], "A0A1")


if __name__ == "__main__":
    unittest.main()
