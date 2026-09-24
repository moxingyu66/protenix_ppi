import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_structure_preservation import (
    parse_mmcif_chain_sequences,
    parse_mmcif_protein_chains,
)
from protenix_ppi.scripts.fetch_structural_candidates import (
    assess_metadata,
    build_search_query,
    stable_candidate_key,
)


def entity(entity_id: str, accession: str, description: str = "Protein kinase") -> dict:
    return {
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": entity_id,
            "asym_ids": [chr(64 + int(entity_id))],
            "uniprot_ids": [accession],
        },
        "rcsb_entity_source_organism": [{"ncbi_taxonomy_id": 9606}],
        "rcsb_polymer_entity": {"pdbx_description": description},
        "entity_poly": {
            "rcsb_entity_polymer_type": "Protein",
            "rcsb_sample_sequence_length": 200,
        },
    }


def valid_metadata() -> tuple[dict, dict, list[dict]]:
    entry = {
        "rcsb_accession_info": {"initial_release_date": "2024-01-01T00:00:00Z"},
        "rcsb_entry_info": {
            "structure_determination_methodology": "experimental",
            "experimental_method": "X-ray",
            "resolution_combined": [2.5],
        },
    }
    assembly = {
        "pdbx_struct_assembly": {
            "details": "author_defined_assembly",
            "method_details": "PISA",
            "rcsb_candidate_assembly": "Y",
        },
        "rcsb_assembly_info": {
            "polymer_composition": "heteromeric protein",
            "polymer_entity_instance_count_protein": 2,
            "polymer_entity_instance_count_nucleic_acid": 0,
            "total_number_interface_residues": 50,
            "total_assembly_buried_surface_area": 1200,
        },
    }
    return entry, assembly, [entity("1", "P12345"), entity("2", "Q67890")]


class FetchStructuralCandidatesTests(unittest.TestCase):
    def test_search_query_freezes_cutoff_and_assembly_definition(self):
        query = build_search_query(0, 100)
        self.assertEqual(query["return_type"], "assembly")
        nodes = query["query"]["nodes"]
        parameters = [node["parameters"] for node in nodes]
        self.assertIn(
            {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater",
                "value": "2021-09-30T00:00:00Z",
            },
            parameters,
        )
        self.assertTrue(any(item.get("value") == "heteromeric protein" for item in parameters))

    def test_valid_human_heteromer_metadata_passes(self):
        entry_value, assembly, entities = valid_metadata()
        reasons, details = assess_metadata(entry_value, assembly, entities)
        self.assertEqual(reasons, [])
        self.assertEqual(details["release_date"], "2024-01-01")
        self.assertEqual(details["resolution"], 2.5)

    def test_candidate_sampling_key_is_stable_and_seeded(self):
        self.assertEqual(stable_candidate_key("8ABC-1"), stable_candidate_key("8ABC-1"))
        self.assertNotEqual(stable_candidate_key("8ABC-1", 1), stable_candidate_key("8ABC-1", 2))

    def test_antibody_and_missing_uniprot_are_visible_exclusions(self):
        entry_value, assembly, entities = valid_metadata()
        entities[0]["rcsb_polymer_entity"]["pdbx_description"] = "Antibody Heavy Chain"
        entities[0]["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"] = []
        reasons, _ = assess_metadata(entry_value, assembly, entities)
        self.assertIn("entity_1_antibody_like", reasons)
        self.assertIn("entity_1_does_not_have_one_uniprot", reasons)

    def test_nonhuman_entity_is_not_hidden_by_one_human_entity(self):
        entry_value, assembly, entities = valid_metadata()
        entities[1]["rcsb_entity_source_organism"] = [{"ncbi_taxonomy_id": 10090}]
        reasons, _ = assess_metadata(entry_value, assembly, entities)
        self.assertIn("entity_2_not_unambiguously_human", reasons)

    def test_mmcif_parser_binds_every_atom_chain_to_entity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "assembly.cif"
            path.write_text(
                "data_test\n"
                "loop_\n"
                "_atom_site.group_PDB\n"
                "_atom_site.label_asym_id\n"
                "_atom_site.label_entity_id\n"
                "ATOM A 1\n"
                "ATOM B 2\n"
                "HETATM C 3\n"
                "#\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_mmcif_protein_chains(path), {"A": "1", "B": "2"})

    def test_sequence_parser_includes_mse_but_ignores_other_ligands(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "assembly.cif"
            path.write_text(
                "data_test\n"
                "loop_\n"
                "_atom_site.group_PDB\n"
                "_atom_site.label_asym_id\n"
                "_atom_site.label_entity_id\n"
                "_atom_site.label_seq_id\n"
                "_atom_site.label_comp_id\n"
                "ATOM A 1 1 ALA\n"
                "HETATM A 1 2 MSE\n"
                "HETATM A 1 3 ATP\n"
                "ATOM B 2 1 GLY\n"
                "#\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_mmcif_chain_sequences(path), {"A": "AM", "B": "G"})


if __name__ == "__main__":
    unittest.main()
