import unittest

from protenix_ppi.scripts.build_normalized_benchmark import (
    finalize_observations,
    quarantine_ambiguous_source_records,
)


def observation(pair_id, a, b, polarity, evidence_id, negative_class="", complexes=()):
    return {
        "pair_id": pair_id,
        "uniprot_a": a,
        "uniprot_b": b,
        "negative_class": negative_class,
        "complex_ids": set(complexes),
        "evidence": {
            "evidence_id": evidence_id,
            "pair_id": pair_id,
            "source_database": "X",
            "source_version": "v",
            "source_record_id": evidence_id,
            "publication_id": "",
            "interaction_type_mi": "MI:0407" if polarity == "positive" else "",
            "detection_method_mi": "MI:0018",
            "evidence_polarity": polarity,
            "negative_definition": "tested negative" if polarity == "negative" else "",
            "license": "test",
            "retrieved_at": "2026-09-15",
        },
    }


class BuildNormalizedBenchmarkTests(unittest.TestCase):
    def test_positive_negative_contradiction_is_quarantined(self):
        pair_id = "HUMAN_P00001_P00002"
        rows, _ = finalize_observations(
            [
                observation(pair_id, "P00001", "P00002", "positive", "E1"),
                observation(pair_id, "P00001", "P00002", "negative", "E2", "curated_negative"),
            ]
        )
        self.assertEqual(rows[0]["pair_status"], "quarantine")
        self.assertEqual(rows[0]["evidence_class"], "unlabeled")

    def test_curated_negative_dominates_screen_class_without_changing_label(self):
        pair_id = "HUMAN_P00001_P00002"
        rows, _ = finalize_observations(
            [
                observation(pair_id, "P00001", "P00002", "negative", "E1", "screen_negative"),
                observation(pair_id, "P00001", "P00002", "negative", "E2", "curated_negative"),
            ]
        )
        self.assertEqual(rows[0]["label"], "0")
        self.assertEqual(rows[0]["evidence_class"], "curated_negative")

    def test_self_pair_is_quarantined(self):
        pair_id = "HUMAN_P00001_P00001"
        rows, _ = finalize_observations(
            [observation(pair_id, "P00001", "P00001", "positive", "E1")]
        )
        self.assertEqual(rows[0]["pair_status"], "quarantine")

    def test_pairs_sharing_a_pdb_receive_same_connected_group(self):
        rows, _ = finalize_observations(
            [
                observation("HUMAN_P00001_P00002", "P00001", "P00002", "positive", "E1", complexes={"1ABC"}),
                observation("HUMAN_P00002_P00003", "P00002", "P00003", "positive", "E2", complexes={"1ABC", "2DEF"}),
                observation("HUMAN_P00003_P00004", "P00003", "P00004", "positive", "E3", complexes={"2DEF"}),
            ]
        )
        self.assertEqual(len({row["complex_group_id"] for row in rows}), 1)

    def test_one_source_record_mapping_to_multiple_pairs_is_quarantined(self):
        items = [
            observation("HUMAN_P00001_P00002", "P00001", "P00002", "negative", "R1", "curated_negative"),
            observation("HUMAN_P00001_P00003", "P00001", "P00003", "negative", "R1", "curated_negative"),
        ]
        for item in items:
            item["raw_identifier_a"] = item["uniprot_a"]
            item["raw_identifier_b"] = item["uniprot_b"]
            item["evidence"]["source_record_id"] = "R1"
        quarantine = []
        kept = quarantine_ambiguous_source_records(items, quarantine)
        self.assertEqual(kept, [])
        self.assertEqual(len(quarantine), 2)


if __name__ == "__main__":
    unittest.main()
