import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.build_psi_mi_filter import compile_policy


def write_fixture(root: Path) -> tuple[Path, Path]:
    ontology = root / "psi-mi.obo"
    ontology.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: MI:0407\nname: direct interaction\n\n"
        "[Term]\nid: MI:0195\nname: covalent binding\nis_a: MI:0407 ! direct interaction\n\n"
        "[Term]\nid: MI:0018\nname: two hybrid\n\n"
        "[Term]\nid: MI:0397\nname: two hybrid array\nis_a: MI:0018 ! two hybrid\n\n"
        "[Term]\nid: MI:0999\nname: obsolete two hybrid\nis_a: MI:0018 ! two hybrid\n"
        "is_obsolete: true\nreplaced_by: MI:0397\n\n"
        "[Term]\nid: MI:0004\nname: affinity chromatography technology\n\n"
        "[Term]\nid: MI:0096\nname: pull down\nis_a: MI:0004 ! affinity chromatography technology\n\n",
        encoding="utf-8",
    )
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "policy_id": "test",
                "ontology_commit": "a" * 40,
                "ontology_sha256": hashlib.sha256(ontology.read_bytes()).hexdigest(),
                "interaction_type_policy": {
                    "auto_accept_exact": ["MI:0407"],
                    "never_auto_accept_exact": [],
                },
                "detection_method_policy": {
                    "auto_accept_roots_with_descendants": ["MI:0018"],
                    "never_auto_accept_roots_with_descendants": ["MI:0004"],
                    "manual_review_exact_even_if_under_auto_root": [],
                },
                "decision_rule": "test",
                "scientific_scope": "test",
            }
        ),
        encoding="utf-8",
    )
    return ontology, policy


class BuildPsiMiFilterTests(unittest.TestCase):
    def test_compiles_descendants_and_excludes_obsolete_auto_terms(self):
        with TemporaryDirectory() as directory:
            ontology, policy = write_fixture(Path(directory))
            result = compile_policy(ontology, policy)
            accepted = {
                item["id"]
                for item in result["detection_method"]["auto_accept_non_obsolete_closure"]
            }
            obsolete = {
                item["id"]
                for item in result["detection_method"]["excluded_obsolete_members_of_auto_closure"]
            }
            manual_interaction = {
                item["id"]
                for item in result["interaction_type"]["direct_descendants_manual_review"]
            }
            self.assertEqual(accepted, {"MI:0018", "MI:0397"})
            self.assertEqual(obsolete, {"MI:0999"})
            self.assertEqual(manual_interaction, {"MI:0195"})

    def test_rejects_ontology_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            ontology, policy = write_fixture(Path(directory))
            data = json.loads(policy.read_text(encoding="utf-8"))
            data["ontology_sha256"] = "0" * 64
            policy.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ontology SHA-256 mismatch"):
                compile_policy(ontology, policy)

    def test_rejects_overlap_between_accept_and_never_accept(self):
        with TemporaryDirectory() as directory:
            ontology, policy = write_fixture(Path(directory))
            data = json.loads(policy.read_text(encoding="utf-8"))
            data["detection_method_policy"]["never_auto_accept_roots_with_descendants"] = [
                "MI:0018"
            ]
            policy.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closures overlap"):
                compile_policy(ontology, policy)

    def test_explicit_manual_review_removes_descendant_from_auto_set(self):
        with TemporaryDirectory() as directory:
            ontology, policy = write_fixture(Path(directory))
            data = json.loads(policy.read_text(encoding="utf-8"))
            data["detection_method_policy"]["manual_review_exact_even_if_under_auto_root"] = [
                "MI:0397"
            ]
            policy.write_text(json.dumps(data), encoding="utf-8")
            result = compile_policy(ontology, policy)
            accepted = {
                item["id"]
                for item in result["detection_method"]["auto_accept_non_obsolete_closure"]
            }
            manual = {
                item["id"]
                for item in result["detection_method"][
                    "manual_review_members_removed_from_auto_closure"
                ]
            }
            self.assertEqual(accepted, {"MI:0018"})
            self.assertEqual(manual, {"MI:0397"})


if __name__ == "__main__":
    unittest.main()
