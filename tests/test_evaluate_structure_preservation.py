import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_structure_preservation import evaluate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reference_mmcif(path: Path) -> None:
    path.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "ATOM A 1\n"
        "ATOM B 2\n"
        "#\n",
        encoding="utf-8",
    )


def build_fixture(root: Path, new_failure: bool = False, overlap: bool = False):
    set_rows = []
    metric_rows = []
    for index in range(2):
        reference = root / f"reference_{index}.cif"
        baseline = root / f"baseline_{index}.cif"
        candidate = root / f"candidate_{index}.cif"
        write_reference_mmcif(reference)
        baseline.write_text(f"baseline {index}", encoding="utf-8")
        candidate.write_text(f"candidate {index}", encoding="utf-8")
        complex_id = f"C{index}"
        pdb_id = f"8{index:03X}"
        set_rows.append(
            {
                "complex_id": complex_id,
                "pdb_id": pdb_id,
                "biological_assembly_id": "1",
                "protein_chain_mapping_json": json.dumps({"A": "P12345", "B": "Q67890"}),
                "reference_structure": reference.name,
                "reference_structure_sha256": digest(reference),
                "source_url": f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif",
                "retrieved_at": "2026-09-15T12:00:00+08:00",
                "pdb_release_date": "2024-01-01",
                "selection_rule_version": "rcsb_human_heteromer_v0.1",
                "bootstrap_group": complex_id,
                "post_cutoff": "true",
                "homology_audit_passed": "true",
                "ppi_train_overlap": "true" if overlap and index == 0 else "false",
            }
        )
        metric_rows.append(
            {
                "complex_id": complex_id,
                "baseline_structure": baseline.name,
                "baseline_structure_sha256": digest(baseline),
                "candidate_structure": candidate.name,
                "candidate_structure_sha256": digest(candidate),
                "baseline_dockq": "0.70",
                "candidate_dockq": "0.69",
                "baseline_irmsd": "1.0",
                "candidate_irmsd": "1.1",
                "baseline_lrmsd": "2.0",
                "candidate_lrmsd": "2.1",
                "baseline_interface_contact_precision": "0.8",
                "candidate_interface_contact_precision": "0.79",
                "baseline_interface_contact_recall": "0.8",
                "candidate_interface_contact_recall": "0.79",
                "baseline_chain_collapse": "false",
                "candidate_chain_collapse": "true" if new_failure and index == 0 else "false",
                "baseline_atomic_overlap": "false",
                "candidate_atomic_overlap": "false",
                "baseline_within_chain_geometry_failure": "false",
                "candidate_within_chain_geometry_failure": "false",
            }
        )
    structural_set = root / "structural_set.csv"
    metrics = root / "metrics.csv"
    write_csv(structural_set, set_rows)
    write_csv(metrics, metric_rows)
    return structural_set, metrics


class EvaluateStructurePreservationTests(unittest.TestCase):
    def test_passes_small_noninferior_fixture(self):
        with TemporaryDirectory() as directory:
            structural_set, metrics = build_fixture(Path(directory))
            result = evaluate(structural_set, metrics, min_complexes=2, bootstrap_replicates=20, bootstrap_seed=7)
            self.assertTrue(result["structural_preservation_passed"])
            self.assertAlmostEqual(result["metrics"]["dockq"]["candidate_minus_baseline_mean"], -0.01)

    def test_new_geometry_failure_fails_gate(self):
        with TemporaryDirectory() as directory:
            structural_set, metrics = build_fixture(Path(directory), new_failure=True)
            result = evaluate(structural_set, metrics, min_complexes=2, bootstrap_replicates=20)
            self.assertFalse(result["structural_preservation_passed"])
            self.assertEqual(result["new_severe_geometry_failures"]["chain_collapse"], 1)

    def test_training_overlap_is_rejected(self):
        with TemporaryDirectory() as directory:
            structural_set, metrics = build_fixture(Path(directory), overlap=True)
            with self.assertRaisesRegex(ValueError, "independence audit failed"):
                evaluate(structural_set, metrics, min_complexes=2, bootstrap_replicates=5)

    def test_asymmetric_unit_url_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            structural_set, metrics = build_fixture(root)
            with structural_set.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["source_url"] = "https://files.rcsb.org/download/8000.cif"
            write_csv(structural_set, rows)
            with self.assertRaisesRegex(ValueError, "RCSB biological assembly"):
                evaluate(structural_set, metrics, min_complexes=2, bootstrap_replicates=5)

    def test_homomeric_chain_mapping_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            structural_set, metrics = build_fixture(root)
            with structural_set.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["protein_chain_mapping_json"] = json.dumps({"A": "P12345", "B": "P12345"})
            write_csv(structural_set, rows)
            with self.assertRaisesRegex(ValueError, "not a heteromeric"):
                evaluate(structural_set, metrics, min_complexes=2, bootstrap_replicates=5)


if __name__ == "__main__":
    unittest.main()
