import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.freeze_structural_set import freeze
from protenix_ppi.scripts.fetch_structural_candidates import CANDIDATE_FIELDS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path, included: int = 30) -> tuple[Path, Path]:
    references = root / "references"
    rows: list[dict[str, str]] = []
    reviews: list[dict[str, str]] = []
    for index in range(32):
        structure = references / f"8{index:03X}-assembly1.cif"
        structure.parent.mkdir(parents=True, exist_ok=True)
        structure.write_text(
            "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.label_asym_id\n_atom_site.label_entity_id\nATOM A 1\nATOM B 2\n#\n",
            encoding="utf-8",
        )
        complex_id = f"C{index:02d}"
        rows.append(
            {
                "complex_id": complex_id,
                "pdb_id": f"8{index:03X}",
                "biological_assembly_id": "1",
                "protein_chain_mapping_json": json.dumps({"A": "P12345", "B": "Q67890"}),
                "reference_structure": str(structure.relative_to(root)),
                "reference_structure_sha256": digest(structure),
                "source_url": f"https://files.rcsb.org/download/8{index:03X}-assembly1.cif",
                "retrieved_at": "2026-09-15T00:00:00+00:00",
                "pdb_release_date": "2024-01-01",
                "selection_rule_version": "rcsb_human_heteromer_v0.1",
                "automated_eligibility": "true",
                "manual_review_status": "pending",
                "exclusion_reasons": "",
                "protein_chain_entity_mapping_json": '{"A":"1","B":"2"}',
                "uniprot_accessions": "P12345;Q67890",
                "entity_descriptions_json": '{"1":"Protein A","2":"Protein B"}',
                "entity_lengths_json": '{"1":200,"2":200}',
                "total_residues": "400",
                "interface_residues": "50",
                "buried_surface_area": "1000",
                "experimental_method": "X-ray",
                "resolution_angstrom": "2.0",
                "assembly_details": "author_defined",
                "assembly_method_details": "PISA",
            }
        )
        include = index < included
        reviews.append(
            {
                "complex_id": complex_id,
                "include": str(include).lower(),
                "biological_assembly_confirmed": "true" if include else "false",
                "chain_mapping_confirmed": "true" if include else "false",
                "human_heteromer_confirmed": "true" if include else "false",
                "b4_b5_outputs_consulted": "false",
                "reviewer_role": "study lead",
                "reviewed_at": "2026-09-15T12:00:00+00:00",
                "rationale": "Verified biological assembly and distinct human protein chains.",
            }
        )
    candidates = root / "structural_candidates.csv"
    review = root / "structural_review.csv"
    write_csv(candidates, CANDIDATE_FIELDS, rows)
    write_csv(review, list(reviews[0]), reviews)
    return candidates, review


class FreezeStructuralSetTests(unittest.TestCase):
    def test_freezes_reviewed_set_and_keeps_homology_unproven(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, review = build_fixture(root)
            result = freeze(candidates, review, root / "frozen")
            self.assertEqual(result["counts"]["included"], 30)
            self.assertEqual(result["status"], "frozen_for_homology_audit_only")
            self.assertNotIn("freeze_metadata.json", result["outputs"])
            with (root / "frozen" / "structural_set.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["homology_audit_passed"] == "false" for row in rows))

    def test_less_than_30_included_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, review = build_fixture(root, included=29)
            with self.assertRaisesRegex(ValueError, "at least 30"):
                freeze(candidates, review, root / "frozen")

    def test_b4_b5_informed_review_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, review = build_fixture(root)
            with review.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["b4_b5_outputs_consulted"] = "true"
            write_csv(review, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "informed by B4/B5"):
                freeze(candidates, review, root / "frozen")


if __name__ == "__main__":
    unittest.main()
