import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_g4_gate import evaluate


COMMIT = "a" * 40
CHECKPOINT = "b" * 64
COHORTS = [
    "balanced_explicit_1to1",
    "full_evidence_pool",
    "balanced_curated_1to1",
    "balanced_screen_1to1",
]
METHODS = {
    "b0b_vs_b0a": (
        "B0b_frozen_PLM_symmetric_logistic_regression",
        "B0a_symmetric_AAC_logistic_regression",
    ),
    "b1_vs_constant": (
        "B1_Protenix_zero_shot",
        "C0_constant_score_reference",
    ),
    "b2_vs_b0b": (
        "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
        "B0b_frozen_PLM_symmetric_logistic_regression",
    ),
    "b3_vs_b2": (
        "B3_frozen_Protenix_nonlinear_PPI_task_head",
        "B2_frozen_Protenix_pair_features_symmetric_logistic_regression",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def record(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": digest(path)}


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


def make_g1_run(root: Path) -> Path:
    run = root / "g1"
    input_path = run / "inputs" / "input.json"
    write_json(input_path, {"sequences": []})
    lock = {
        "repository_url": "https://github.com/bytedance/Protenix.git",
        "source_commit": COMMIT,
        "package_version": "1.0.0",
        "model_name": "protenix_base_default_v1.0.0",
        "declared_training_cutoff": "2021-09-30",
        "checkpoint_sha256": CHECKPOINT,
        "g0_route": "A",
        "python_version": "3.10.15",
        "torch_version": "2.5.1",
        "cuda_runtime": "12.4",
        "nvidia_driver": "550.54.15",
    }
    write_json(run / "backbone_lock.json", lock)
    for relative in ("logs/command.txt", "logs/stdout.txt", "telemetry/gpu.csv"):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")
    gradient = {
        "loss_value": 1.0,
        "loss_finite": True,
        "trainable_parameter_count": 10,
        "parameters_with_finite_nonzero_gradients": 2,
        "frozen_parameters_with_gradients": 0,
        "optimizer_step_completed": True,
        "peak_gpu_memory_mib": 2048,
    }
    write_json(run / "backward" / "gradient.json", gradient)
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "prediction.cif").write_text("data_test\n", encoding="utf-8")
    write_json(run / "outputs" / "confidence.json", {"iptm": 0.5})
    write_json(
        run / "manifest.json",
        {
            "model_name": lock["model_name"],
            "source_commit": COMMIT,
            "checkpoint_sha256": CHECKPOINT,
            "input_sha256": digest(input_path),
            "command_file": "logs/command.txt",
            "stdout_file": "logs/stdout.txt",
            "telemetry_file": "telemetry/gpu.csv",
            "gradient_summary_file": "backward/gradient.json",
            "wall_time_seconds": 10,
            "exit_code": 0,
        },
    )
    return run


def make_fixture(root: Path) -> Path:
    freeze = root / "benchmark_freeze_manifest.json"
    write_json(
        freeze,
        {
            "status": "frozen_before_model_scoring",
            "validation": {"passed": True, "errors": []},
            "outputs": {
                "splits/c3_primary/splits.csv": {"sha256": "1" * 64},
                "splits/c3_primary/protein_pool_assignments.csv": {"sha256": "4" * 64},
                "evaluation_cohorts/cohort_membership.csv": {"sha256": "2" * 64},
            },
            "inputs": {
                "proteins.clustered.csv": {"sha256": "3" * 64}
            },
        },
    )
    freeze_hash = digest(freeze)

    suites = {}
    for comparison, (candidate, baseline) in METHODS.items():
        path = root / f"{comparison}.json"
        write_json(
            path,
            {
                "comparison": comparison,
                "candidate_method": candidate,
                "baseline_method": baseline,
                "benchmark_freeze_manifest_sha256": freeze_hash,
                "registered_seeds": [42, 123, 999],
                "required_cohorts": COHORTS,
                "mandatory_secondary_results_complete": True,
                "cohort_results": {name: {} for name in COHORTS},
                "primary_support": {"claim_supported": True},
            },
        )
        suites[comparison] = record(path, root)

    references = root / "references"
    references.mkdir()
    structural_set = root / "structural_set.csv"
    rows = []
    for index in range(30):
        structure = references / f"C{index:02d}.cif"
        write_reference_mmcif(structure)
        pdb_id = f"8{index:03X}"
        rows.append(
            {
                "complex_id": f"C{index:02d}",
                "pdb_id": pdb_id,
                "biological_assembly_id": "1",
                "protein_chain_mapping_json": json.dumps(
                    {"A": f"P{index:05d}", "B": f"Q{index:05d}"}
                ),
                "reference_structure": str(structure.relative_to(root)),
                "reference_structure_sha256": digest(structure),
                "source_url": f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif",
                "retrieved_at": "2026-09-15T12:00:00+08:00",
                "pdb_release_date": "2022-01-01",
                "selection_rule_version": "rcsb_human_heteromer_v0.1",
                "bootstrap_group": f"G{index:02d}",
                "post_cutoff": "true",
                "homology_audit_passed": "true",
                "ppi_train_overlap": "false",
            }
        )
    with structural_set.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    command = root / "homology_command.txt"
    command.write_text("mmseqs easy-search --min-seq-id 0.30 -c 0.50\n", encoding="utf-8")
    homology = root / "homology_audit.json"
    audited_set = root / "structural_set_audited.csv"
    audited_set.write_bytes(structural_set.read_bytes())
    write_json(
        homology,
        {
            "schema_version": "1.0",
            "structural_set_sha256": digest(audited_set),
            "benchmark_freeze_manifest_sha256": freeze_hash,
            "ppi_train_proteins_sha256": "3" * 64,
            "protein_pool_assignments_sha256": "4" * 64,
            "tool": {
                "name": "MMseqs2",
                "version": "15.6f452",
                "parameters": {
                    "min_seq_id": 0.30,
                    "coverage": 0.50,
                    "cov_mode": 0,
                    "alignment_mode": 3,
                    "sensitivity": 7.5,
                },
                "command_evidence": record(command, root),
            },
            "outputs": {
                "structural_set_audited.csv": record(audited_set, root)
            },
            "thresholds": {
                "maximum_sequence_identity": 0.30,
                "minimum_coverage": 0.50,
            },
            "b4_b5_outputs_consulted": False,
            "complexes": [
                {
                    "complex_id": row["complex_id"],
            "ppi_train_protein_overlap": False,
            "homology_cluster_overlap": False,
                }
                for row in rows
            ],
        },
    )

    limitation = root / "b3_error_analysis.json"
    write_json(limitation, {"pattern": "errors concentrate in interface-remodelling complexes"})
    run = make_g1_run(root)
    trainable = root / "trainable_parameters.txt"
    trainable.write_text("pairformer.block.47.projection.adapter_A\n", encoding="utf-8")
    capacity_command = root / "capacity_command.txt"
    capacity_command.write_text("python run_b4_capacity_probe.py --steps 1\n", encoding="utf-8")
    capacity_stdout = root / "capacity_stdout_stderr.txt"
    capacity_stdout.write_text("loss=1.25 optimizer_step=complete\n", encoding="utf-8")
    capacity_telemetry = root / "capacity_gpu.csv"
    capacity_telemetry.write_text("timestamp,memory.used\n0,60000\n", encoding="utf-8")
    capacity = root / "capacity.json"
    write_json(
        capacity,
        {
            "schema_version": "1.0",
            "method_id": "B4",
            "model_name": "protenix_base_default_v1.0.0",
            "source_commit": COMMIT,
            "checkpoint_sha256": CHECKPOINT,
            "graph_description": "LoRA adapters on the registered late pair/interface projections only",
            "gpu_model": "NVIDIA A100 80GB",
            "measured_on_target_machine": True,
            "graph_matches_proposed_method": True,
            "representative_input_meets_registered_target": True,
            "trainable_parameter_evidence": record(trainable, root),
            "command_evidence": record(capacity_command, root),
            "stdout_stderr_evidence": record(capacity_stdout, root),
            "telemetry_evidence": record(capacity_telemetry, root),
            "measurements": {
                "representative_total_residues": 800,
                "microbatch_size": 1,
                "gradient_accumulation_steps": 8,
                "intended_trainable_parameter_count": 1000,
                "parameters_with_finite_nonzero_gradients": 20,
                "frozen_parameters_with_gradients": 0,
                "peak_gpu_memory_mib": 60000,
                "available_gpu_memory_mib": 80000,
                "elapsed_seconds": 120,
                "loss_value": 1.25,
                "forward_completed": True,
                "backward_completed": True,
                "optimizer_step_completed": True,
                "loss_finite": True,
                "oom_observed": False,
            },
        },
    )

    decision = root / "decision.json"
    write_json(
        decision,
        {
            "schema_version": "1.0",
            "decision_id": "g4_test_decision",
            "created_at": "2026-09-15T12:00:00+08:00",
            "benchmark_freeze_manifest": record(freeze, root),
            "comparison_suites": suites,
            "b3_scientific_limitation": {
                "category": "interface_representation_bottleneck",
                "affected_cohort": "balanced_explicit_1to1",
                "observed_pattern": "Registered subgroup analysis shows repeated ranking errors in interface-remodelling pairs.",
                "mechanistic_hypothesis": "Frozen pair states omit task-adapted interface reorganization needed for this subgroup.",
                "why_b3_cannot_address_it": "B3 only remaps an immutable vector and cannot alter the pair representation supplied upstream.",
                "why_structural_tuning_may_address_it": "Adapters on late pair projections can test whether interface-aware representation changes recover ranking.",
                "scientific_bottleneck": True,
                "engineering_only": False,
                "b4_b5_outputs_consulted": False,
                "evidence": record(limitation, root),
                "human_review": {
                    "status": "approved",
                    "reviewer_role": "PPI study lead",
                    "reviewed_at": "2026-09-15T11:00:00+08:00",
                },
            },
            "structural_preservation": {
                "structural_set": record(structural_set, root),
                "homology_audit": record(homology, root),
                "frozen_before_structural_tuning": True,
                "b4_b5_outputs_consulted": False,
            },
            "g1_run": {
                "path": str(run.relative_to(root)),
                "manifest_sha256": digest(run / "manifest.json"),
            },
            "proposed_structural_tuning": {
                "method_id": "B4",
                "backward_capacity_report": record(capacity, root),
            },
        },
    )
    return decision


class EvaluateG4GateTests(unittest.TestCase):
    def test_complete_evidence_authorizes_only_named_method(self):
        with TemporaryDirectory() as directory:
            decision = make_fixture(Path(directory))
            result = evaluate(decision)
            self.assertTrue(result["structural_tuning_authorized"])
            self.assertEqual(result["authorized_method"], "B4")
            self.assertTrue(all(result["gate_checks"].values()))

    def test_unsupported_b2_comparison_keeps_gate_locked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            decision = make_fixture(root)
            decision_value = json.loads(decision.read_text(encoding="utf-8"))
            suite_path = root / "b2_vs_b0b.json"
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            suite["primary_support"]["claim_supported"] = False
            write_json(suite_path, suite)
            decision_value["comparison_suites"]["b2_vs_b0b"] = record(suite_path, root)
            write_json(decision, decision_value)
            result = evaluate(decision)
            self.assertFalse(result["structural_tuning_authorized"])
            self.assertFalse(result["gate_checks"]["b2_beats_strong_b0b_sequence_baseline"])

    def test_unreviewed_limitation_keeps_gate_locked(self):
        with TemporaryDirectory() as directory:
            decision = make_fixture(Path(directory))
            value = json.loads(decision.read_text(encoding="utf-8"))
            value["b3_scientific_limitation"]["human_review"]["status"] = "rejected"
            write_json(decision, value)
            result = evaluate(decision)
            self.assertFalse(result["gate_checks"]["b3_has_reviewed_scientific_limitation"])

    def test_structural_training_overlap_keeps_gate_locked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            decision = make_fixture(root)
            structural_set = root / "structural_set.csv"
            with structural_set.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["ppi_train_overlap"] = "true"
            with structural_set.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            audited_set = root / "structural_set_audited.csv"
            audited_set.write_bytes(structural_set.read_bytes())
            value = json.loads(decision.read_text(encoding="utf-8"))
            value["structural_preservation"]["structural_set"] = record(structural_set, root)
            audit_path = root / "homology_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["structural_set_sha256"] = digest(structural_set)
            audit["outputs"]["structural_set_audited.csv"] = record(audited_set, root)
            write_json(audit_path, audit)
            value["structural_preservation"]["homology_audit"] = record(audit_path, root)
            write_json(decision, value)
            result = evaluate(decision)
            self.assertFalse(
                result["gate_checks"]["structural_set_has_at_least_30_independent_post_cutoff_complexes"]
            )

    def test_oom_capacity_probe_keeps_gate_locked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            decision = make_fixture(root)
            capacity_path = root / "capacity.json"
            capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
            capacity["measurements"]["oom_observed"] = True
            write_json(capacity_path, capacity)
            value = json.loads(decision.read_text(encoding="utf-8"))
            value["proposed_structural_tuning"]["backward_capacity_report"] = record(
                capacity_path, root
            )
            write_json(decision, value)
            result = evaluate(decision)
            self.assertFalse(
                result["gate_checks"]["target_machine_supports_proposed_backward_graph"]
            )

    def test_hash_drift_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            decision = make_fixture(root)
            (root / "b3_vs_b2.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                evaluate(decision)


if __name__ == "__main__":
    unittest.main()
