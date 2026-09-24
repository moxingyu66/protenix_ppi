import unittest

from protenix_ppi.scripts.resolve_identifier_mappings import resolve_one


def inventory_row(status="mapping_candidate", namespace="Ensembl", query="ENSG00000000005"):
    return {
        "source_database": "HuRI",
        "source_version": "v",
        "evidence_role": "positive_evidence",
        "namespace": namespace,
        "raw_identifier": query,
        "mapping_query_id": query,
        "isoform_specific": "false",
        "mapping_status": status,
        "occurrence_count": "1",
    }


class ResolveIdentifierMappingsTests(unittest.TestCase):
    def test_unique_reviewed_target_is_accepted_even_with_unreviewed_alternatives(self):
        row = resolve_one(
            inventory_row(),
            {"ENSG00000000005": {"P12345", "A0A000"}},
            {"P12345"},
        )
        self.assertEqual(row["resolution_status"], "accepted")
        self.assertEqual(row["resolved_accession"], "P12345")

    def test_multiple_reviewed_targets_are_quarantined(self):
        row = resolve_one(
            inventory_row(),
            {"ENSG00000000005": {"P12345", "Q12345"}},
            {"P12345", "Q12345"},
        )
        self.assertEqual(row["resolution_status"], "quarantine")
        self.assertEqual(row["resolution_reason"], "multiple_reviewed_human_targets")

    def test_no_reviewed_target_is_deferred_not_dropped(self):
        row = resolve_one(inventory_row(), {"ENSG00000000005": {"A0A000"}}, set())
        self.assertEqual(row["resolution_status"], "deferred")

    def test_unique_current_unreviewed_human_target_is_accepted_after_audit(self):
        records = {"A0A000": {"Organism (ID)": "9606"}}
        row = resolve_one(
            inventory_row(), {"ENSG00000000005": {"A0A000"}}, set(), records
        )
        self.assertEqual(row["resolution_status"], "accepted")
        self.assertEqual(row["resolved_accession"], "A0A000")

    def test_nonhuman_target_is_explicitly_excluded(self):
        records = {"P99999": {"Organism (ID)": "10090"}}
        row = resolve_one(
            inventory_row(namespace="UniProtKB_AC-ID", query="P99999"),
            {"P99999": {"P99999"}},
            set(),
            records,
        )
        self.assertEqual(row["resolution_status"], "excluded")
        self.assertEqual(row["resolution_reason"], "no_human_target_in_pinned_release")

    def test_one_human_among_multiple_targets_stays_quarantined(self):
        records = {
            "A0A000": {"Organism (ID)": "9606"},
            "P99999": {"Organism (ID)": "10090"},
        }
        row = resolve_one(
            inventory_row(),
            {"ENSG00000000005": {"A0A000", "P99999"}},
            set(),
            records,
        )
        self.assertEqual(row["resolution_status"], "quarantine")

    def test_isoform_remains_quarantined(self):
        row = inventory_row(
            status="quarantine_isoform_specific",
            namespace="UniProtKB_AC-ID",
            query="P12345",
        )
        row["raw_identifier"] = "P12345-2"
        resolved = resolve_one(row, {"P12345": {"P12345"}}, {"P12345"})
        self.assertEqual(resolved["resolution_status"], "quarantine")
        self.assertEqual(resolved["resolved_accession"], "")


if __name__ == "__main__":
    unittest.main()
