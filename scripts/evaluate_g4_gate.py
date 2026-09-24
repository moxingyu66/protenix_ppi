#!/usr/bin/env python3
"""Evaluate the evidence gate that authorizes one proposed B4/B5 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.evaluate_structure_preservation import (
    MIN_COMPLEXES,
    PRIMARY_CUTOFF,
    SET_FIELDS,
    parse_bool,
    resolve_hashed_file,
    validate_structural_set_provenance,
)
from protenix_ppi.scripts.validate_g1_artifacts import (
    HEX40,
    HEX64,
    PRIMARY_MODEL,
    sha256_file,
    validate_run,
)


SCHEMA_VERSION = "1.0"
REGISTERED_SEEDS = [42, 123, 999]
REQUIRED_COHORTS = {
    "balanced_explicit_1to1",
    "full_evidence_pool",
    "balanced_curated_1to1",
    "balanced_screen_1to1",
}
REQUIRED_COMPARISONS = {
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
LIMITATION_CATEGORIES = {
    "interface_representation_bottleneck",
    "ranking_error_subgroup",
    "calibration_failure",
    "long_range_pair_context",
    "other_scientific",
}
HEX64_TEXT = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_HOMOLOGY_MMSEQS_PARAMETERS = {
    "min_seq_id": 0.30,
    "coverage": 0.50,
    "cov_mode": 0,
    "alignment_mode": 3,
    "sensitivity": 7.5,
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _is_placeholder(value: Any) -> bool:
    text = str(value).strip()
    return not text or "REPLACE" in text.upper() or text.upper() in {"TODO", "TBD"}


def _require_text(mapping: dict[str, Any], key: str, source: str, minimum: int = 1) -> str:
    value = str(mapping.get(key, "")).strip()
    if _is_placeholder(value) or len(value) < minimum:
        raise ValueError(f"{source} requires a substantive {key}")
    return value


def _parse_iso_datetime(value: Any, source: str) -> str:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{source} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{source} must include a timezone")
    return str(value).strip()


def _resolve_path(value: Any, base: Path, source: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or _is_placeholder(value):
        raise ValueError(f"{source} path is missing or a placeholder")
    path = Path(value.strip())
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{source} {kind} does not exist: {path}")
    return path


def resolve_file_record(record: Any, manifest_path: Path, source: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{source} must be an object with path and sha256")
    path = _resolve_path(record.get("path"), manifest_path.parent, source)
    digest = str(record.get("sha256", "")).strip().lower()
    if not HEX64_TEXT.fullmatch(digest):
        raise ValueError(f"{source} has an invalid SHA-256")
    if sha256_file(path) != digest:
        raise ValueError(f"{source} SHA-256 mismatch")
    return path


def validate_freeze_manifest(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    if freeze.get("status") != "frozen_before_model_scoring":
        raise ValueError("benchmark is not marked frozen_before_model_scoring")
    validation = freeze.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True or validation.get("errors"):
        raise ValueError("benchmark freeze manifest does not prove a passing C3 validation")
    outputs = freeze.get("outputs")
    required_outputs = {
        "splits/c3_primary/splits.csv",
        "splits/c3_primary/protein_pool_assignments.csv",
        "evaluation_cohorts/cohort_membership.csv",
    }
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError("benchmark freeze manifest lacks required C3/cohort outputs")
    inputs = freeze.get("inputs")
    if not isinstance(inputs, dict) or "proteins.clustered.csv" not in inputs:
        raise ValueError("benchmark freeze manifest lacks clustered protein input")
    for source, mapping, keys in (
        ("freeze inputs", inputs, {"proteins.clustered.csv"}),
        ("freeze outputs", outputs, required_outputs),
    ):
        for key in keys:
            digest = str((mapping.get(key) or {}).get("sha256", "")).strip().lower()
            if not HEX64_TEXT.fullmatch(digest):
                raise ValueError(f"{source} has an invalid SHA-256 for {key}")
    return freeze


def validate_comparison_suite(
    path: Path,
    expected_comparison: str,
    freeze_sha256: str,
) -> dict[str, Any]:
    suite = load_json(path)
    expected_candidate, expected_baseline = REQUIRED_COMPARISONS[expected_comparison]
    if suite.get("comparison") != expected_comparison:
        raise ValueError(f"comparison suite is not {expected_comparison}")
    if suite.get("candidate_method") != expected_candidate:
        raise ValueError(f"{expected_comparison} candidate method identity drifted")
    if suite.get("baseline_method") != expected_baseline:
        raise ValueError(f"{expected_comparison} baseline method identity drifted")
    if suite.get("benchmark_freeze_manifest_sha256") != freeze_sha256:
        raise ValueError(f"{expected_comparison} is not bound to the supplied benchmark freeze")
    if suite.get("registered_seeds") != REGISTERED_SEEDS:
        raise ValueError(f"{expected_comparison} does not contain exactly the registered seeds")
    if suite.get("mandatory_secondary_results_complete") is not True:
        raise ValueError(f"{expected_comparison} mandatory cohort results are incomplete")
    required = set(suite.get("required_cohorts", []))
    observed = set((suite.get("cohort_results") or {}).keys())
    if required != REQUIRED_COHORTS or observed != REQUIRED_COHORTS:
        raise ValueError(f"{expected_comparison} does not contain exactly the four frozen cohorts")
    support = suite.get("primary_support")
    if not isinstance(support, dict) or not isinstance(support.get("claim_supported"), bool):
        raise ValueError(f"{expected_comparison} lacks a valid primary support decision")
    return suite


def validate_limitation(
    limitation: Any, decision_manifest_path: Path
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(limitation, dict):
        raise ValueError("b3_scientific_limitation must be an object")
    category = str(limitation.get("category", "")).strip()
    if category not in LIMITATION_CATEGORIES:
        raise ValueError("b3_scientific_limitation category is not registered")
    affected = str(limitation.get("affected_cohort", "")).strip()
    if affected not in REQUIRED_COHORTS:
        raise ValueError("b3_scientific_limitation affected_cohort is not a frozen cohort")
    fields = {
        key: _require_text(limitation, key, "b3_scientific_limitation", 20)
        for key in (
            "observed_pattern",
            "mechanistic_hypothesis",
            "why_b3_cannot_address_it",
            "why_structural_tuning_may_address_it",
        )
    }
    evidence_path = resolve_file_record(
        limitation.get("evidence"), decision_manifest_path, "B3 limitation evidence"
    )
    review = limitation.get("human_review")
    if not isinstance(review, dict):
        raise ValueError("B3 limitation requires a structured human_review")
    _parse_iso_datetime(review.get("reviewed_at"), "B3 limitation reviewed_at")
    reviewer_role = _require_text(review, "reviewer_role", "B3 limitation human_review", 3)
    checks = {
        "scientific_bottleneck_confirmed": limitation.get("scientific_bottleneck") is True,
        "not_engineering_only": limitation.get("engineering_only") is False,
        "review_approved": review.get("status") == "approved",
        "no_b4_b5_output_used": limitation.get("b4_b5_outputs_consulted") is False,
    }
    return all(checks.values()), {
        "category": category,
        "affected_cohort": affected,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "reviewer_role": reviewer_role,
        "text_fields_present": sorted(fields),
        "checks": checks,
    }


def validate_structural_set(path: Path) -> tuple[bool, dict[str, Any], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = SET_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    seen: set[str] = set()
    post_cutoff_count = 0
    independent_count = 0
    for line, row in enumerate(rows, start=2):
        complex_id = row["complex_id"].strip()
        if not complex_id or complex_id in seen:
            raise ValueError(f"{path.name}:{line}: missing or duplicate complex_id")
        seen.add(complex_id)
        try:
            release_date = date.fromisoformat(row["pdb_release_date"].strip())
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}: invalid pdb_release_date") from exc
        post_cutoff = parse_bool(row["post_cutoff"], f"{path.name}:{line}:post_cutoff")
        homology_passed = parse_bool(
            row["homology_audit_passed"], f"{path.name}:{line}:homology_audit_passed"
        )
        train_overlap = parse_bool(
            row["ppi_train_overlap"], f"{path.name}:{line}:ppi_train_overlap"
        )
        if release_date > PRIMARY_CUTOFF and post_cutoff:
            post_cutoff_count += 1
        if homology_passed and not train_overlap:
            independent_count += 1
        if not row["bootstrap_group"].strip():
            raise ValueError(f"{path.name}:{line}: bootstrap_group is empty")
        reference_path = resolve_hashed_file(
            row["reference_structure"],
            row["reference_structure_sha256"],
            path,
            f"{path.name}:{line}:reference",
        )
        validate_structural_set_provenance(row, path, line, reference_path)
    checks = {
        "at_least_30_complexes": len(rows) >= MIN_COMPLEXES,
        "all_references_post_cutoff": post_cutoff_count == len(rows),
        "all_rows_independent_of_ppi_train": independent_count == len(rows),
    }
    return all(checks.values()), {
        "complex_count": len(rows),
        "post_cutoff_count": post_cutoff_count,
        "independent_count": independent_count,
        "checks": checks,
    }, seen


def validate_homology_audit(
    path: Path,
    *,
    structural_set_sha256: str,
    freeze_sha256: str,
    ppi_proteins_sha256: str,
    protein_pool_assignments_sha256: str,
    expected_complex_ids: set[str],
) -> tuple[bool, dict[str, Any]]:
    report = load_json(path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structural homology audit schema_version is invalid")
    if report.get("structural_set_sha256") != structural_set_sha256:
        raise ValueError("homology audit is not bound to the structural set")
    if report.get("benchmark_freeze_manifest_sha256") != freeze_sha256:
        raise ValueError("homology audit is not bound to the frozen PPI benchmark")
    if report.get("ppi_train_proteins_sha256") != ppi_proteins_sha256:
        raise ValueError("homology audit used PPI proteins different from the frozen benchmark")
    if report.get("protein_pool_assignments_sha256") != protein_pool_assignments_sha256:
        raise ValueError("homology audit used protein assignments different from the frozen C3 split")
    audited_output = (report.get("outputs") or {}).get("structural_set_audited.csv")
    audited_path = resolve_file_record(
        audited_output, path, "audited structural set output"
    )
    if sha256_file(audited_path) != report.get("structural_set_sha256"):
        raise ValueError("homology audit structural_set_sha256 differs from audited output")
    tool = report.get("tool")
    if not isinstance(tool, dict):
        raise ValueError("homology audit requires tool provenance")
    tool_name = _require_text(tool, "name", "homology audit tool", 2)
    tool_version = _require_text(tool, "version", "homology audit tool", 1)
    if tool_name != "MMseqs2":
        raise ValueError("homology audit must use MMseqs2")
    if tool.get("parameters") != EXPECTED_HOMOLOGY_MMSEQS_PARAMETERS:
        raise ValueError("homology audit MMseqs2 parameters differ from the frozen rule")
    command_path = resolve_file_record(tool.get("command_evidence"), path, "homology command evidence")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("homology audit requires thresholds")
    try:
        max_identity = float(thresholds.get("maximum_sequence_identity"))
        minimum_coverage = float(thresholds.get("minimum_coverage"))
    except (TypeError, ValueError) as exc:
        raise ValueError("homology audit thresholds must be numeric") from exc
    if not math.isclose(max_identity, 0.30) or not math.isclose(minimum_coverage, 0.50):
        raise ValueError("homology audit thresholds differ from the frozen 30%/50% rule")
    entries = report.get("complexes")
    if not isinstance(entries, list):
        raise ValueError("homology audit complexes must be a list")
    observed: set[str] = set()
    overlap_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"homology audit complex entry {index} is not an object")
        complex_id = str(entry.get("complex_id", "")).strip()
        if not complex_id or complex_id in observed:
            raise ValueError("homology audit has a missing or duplicate complex_id")
        observed.add(complex_id)
        if entry.get("ppi_train_protein_overlap") is not False:
            overlap_count += 1
        if entry.get("homology_cluster_overlap") is not False:
            overlap_count += 1
    exact_coverage = observed == expected_complex_ids
    checks = {
        "all_structural_complexes_audited": exact_coverage,
        "no_protein_or_homology_overlap": overlap_count == 0,
        "audit_completed_before_b4_b5": report.get("b4_b5_outputs_consulted") is False,
    }
    return all(checks.values()), {
        "tool": {"name": tool_name, "version": tool_version},
        "command_evidence_path": str(command_path),
        "audited_structural_set_path": str(audited_path),
        "audited_complex_count": len(observed),
        "overlap_flag_count": overlap_count,
        "checks": checks,
    }


def validate_g1_record(record: Any, decision_manifest_path: Path) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not isinstance(record, dict):
        raise ValueError("g1_run must be an object")
    run_dir = _resolve_path(record.get("path"), decision_manifest_path.parent, "G1 run", directory=True)
    manifest_path = run_dir / "manifest.json"
    expected_digest = str(record.get("manifest_sha256", "")).strip().lower()
    if not HEX64_TEXT.fullmatch(expected_digest) or sha256_file(manifest_path) != expected_digest:
        raise ValueError("G1 manifest SHA-256 mismatch")
    errors = validate_run(run_dir)
    lock = load_json(run_dir / "backbone_lock.json")
    return not errors, {
        "run_directory": str(run_dir),
        "manifest_sha256": expected_digest,
        "validation_errors": errors,
    }, lock


def validate_capacity_report(
    path: Path,
    *,
    expected_method: str,
    backbone_lock: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    report = load_json(path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("backward capacity report schema_version is invalid")
    if report.get("method_id") != expected_method:
        raise ValueError("backward capacity report method differs from the proposal")
    if report.get("model_name") != PRIMARY_MODEL:
        raise ValueError("backward capacity report uses the wrong Protenix model")
    if report.get("source_commit") != backbone_lock.get("source_commit"):
        raise ValueError("capacity report source commit differs from G1")
    if report.get("checkpoint_sha256") != backbone_lock.get("checkpoint_sha256"):
        raise ValueError("capacity report checkpoint differs from G1")
    if not HEX40.fullmatch(str(report.get("source_commit", ""))):
        raise ValueError("capacity report source_commit is invalid")
    _require_text(report, "graph_description", "backward capacity report", 20)
    _require_text(report, "gpu_model", "backward capacity report", 3)
    parameter_path = resolve_file_record(
        report.get("trainable_parameter_evidence"), path, "trainable parameter evidence"
    )
    command_path = resolve_file_record(
        report.get("command_evidence"), path, "capacity-probe command evidence"
    )
    stdout_path = resolve_file_record(
        report.get("stdout_stderr_evidence"), path, "capacity-probe stdout/stderr evidence"
    )
    telemetry_path = resolve_file_record(
        report.get("telemetry_evidence"), path, "capacity-probe GPU telemetry evidence"
    )
    for evidence_path, label in (
        (parameter_path, "trainable parameter evidence"),
        (command_path, "capacity-probe command evidence"),
        (stdout_path, "capacity-probe stdout/stderr evidence"),
        (telemetry_path, "capacity-probe GPU telemetry evidence"),
    ):
        if evidence_path.stat().st_size <= 0:
            raise ValueError(f"{label} is empty")
    measurements = report.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("backward capacity report requires measurements")
    numeric_keys = (
        "representative_total_residues",
        "microbatch_size",
        "gradient_accumulation_steps",
        "intended_trainable_parameter_count",
        "parameters_with_finite_nonzero_gradients",
        "peak_gpu_memory_mib",
        "available_gpu_memory_mib",
        "elapsed_seconds",
    )
    numbers: dict[str, float] = {}
    for key in numeric_keys:
        try:
            value = float(measurements.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"capacity report measurement {key} must be numeric") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"capacity report measurement {key} must be finite and positive")
        numbers[key] = value
    for key in (
        "representative_total_residues",
        "microbatch_size",
        "gradient_accumulation_steps",
        "intended_trainable_parameter_count",
        "parameters_with_finite_nonzero_gradients",
    ):
        if not numbers[key].is_integer():
            raise ValueError(f"capacity report measurement {key} must be an integer")
    try:
        loss_value = float(measurements.get("loss_value"))
    except (TypeError, ValueError) as exc:
        raise ValueError("capacity report loss_value must be numeric") from exc
    if not math.isfinite(loss_value):
        raise ValueError("capacity report loss_value must be finite")
    try:
        frozen_with_gradients = int(measurements.get("frozen_parameters_with_gradients"))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen_parameters_with_gradients must be an integer") from exc
    headroom = numbers["available_gpu_memory_mib"] - numbers["peak_gpu_memory_mib"]
    checks = {
        "measured_on_target_machine": report.get("measured_on_target_machine") is True,
        "graph_matches_proposed_method": report.get("graph_matches_proposed_method") is True,
        "representative_input_meets_registered_target": (
            report.get("representative_input_meets_registered_target") is True
        ),
        "forward_completed": measurements.get("forward_completed") is True,
        "backward_completed": measurements.get("backward_completed") is True,
        "optimizer_step_completed": measurements.get("optimizer_step_completed") is True,
        "finite_loss": measurements.get("loss_finite") is True,
        "nonzero_intended_gradients": numbers["parameters_with_finite_nonzero_gradients"] > 0,
        "no_frozen_gradients": frozen_with_gradients == 0,
        "no_oom": measurements.get("oom_observed") is False,
        "positive_memory_headroom": headroom > 0,
    }
    return all(checks.values()), {
        "method_id": expected_method,
        "trainable_parameter_evidence_path": str(parameter_path),
        "command_evidence_path": str(command_path),
        "stdout_stderr_evidence_path": str(stdout_path),
        "telemetry_evidence_path": str(telemetry_path),
        "loss_value": loss_value,
        "peak_gpu_memory_mib": numbers["peak_gpu_memory_mib"],
        "available_gpu_memory_mib": numbers["available_gpu_memory_mib"],
        "memory_headroom_mib": headroom,
        "checks": checks,
    }


def evaluate(decision_manifest_path: Path) -> dict[str, Any]:
    decision_manifest_path = decision_manifest_path.resolve()
    decision = load_json(decision_manifest_path)
    if decision.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("G4 decision manifest schema_version is invalid")
    decision_id = _require_text(decision, "decision_id", "G4 decision manifest", 4)
    _parse_iso_datetime(decision.get("created_at"), "G4 decision manifest created_at")

    freeze_path = resolve_file_record(
        decision.get("benchmark_freeze_manifest"),
        decision_manifest_path,
        "benchmark freeze manifest",
    )
    freeze_manifest = validate_freeze_manifest(freeze_path)
    freeze_sha256 = sha256_file(freeze_path)

    suite_records = decision.get("comparison_suites")
    if not isinstance(suite_records, dict) or set(suite_records) != set(REQUIRED_COMPARISONS):
        raise ValueError("comparison_suites must contain exactly the four G4 comparisons")
    suites: dict[str, dict[str, Any]] = {}
    suite_paths: dict[str, str] = {}
    for comparison in REQUIRED_COMPARISONS:
        suite_path = resolve_file_record(
            suite_records[comparison], decision_manifest_path, f"{comparison} comparison suite"
        )
        suites[comparison] = validate_comparison_suite(
            suite_path, comparison, freeze_sha256
        )
        suite_paths[comparison] = str(suite_path)

    limitation_passed, limitation_details = validate_limitation(
        decision.get("b3_scientific_limitation"), decision_manifest_path
    )

    structural = decision.get("structural_preservation")
    if not isinstance(structural, dict):
        raise ValueError("structural_preservation must be an object")
    structural_set_path = resolve_file_record(
        structural.get("structural_set"), decision_manifest_path, "structural preservation set"
    )
    structural_set_passed, structural_set_details, complex_ids = validate_structural_set(
        structural_set_path
    )
    homology_path = resolve_file_record(
        structural.get("homology_audit"), decision_manifest_path, "structural homology audit"
    )
    homology_passed, homology_details = validate_homology_audit(
        homology_path,
        structural_set_sha256=sha256_file(structural_set_path),
        freeze_sha256=freeze_sha256,
        ppi_proteins_sha256=str(
            (freeze_manifest.get("inputs") or {})
            .get("proteins.clustered.csv", {})
            .get("sha256", "")
        ),
        protein_pool_assignments_sha256=str(
            (freeze_manifest.get("outputs") or {})
            .get("splits/c3_primary/protein_pool_assignments.csv", {})
            .get("sha256", "")
        ),
        expected_complex_ids=complex_ids,
    )
    structural_frozen_before_tuning = (
        structural.get("frozen_before_structural_tuning") is True
        and structural.get("b4_b5_outputs_consulted") is False
    )

    g1_passed, g1_details, backbone_lock = validate_g1_record(
        decision.get("g1_run"), decision_manifest_path
    )

    proposal = decision.get("proposed_structural_tuning")
    if not isinstance(proposal, dict):
        raise ValueError("proposed_structural_tuning must be an object")
    method_id = str(proposal.get("method_id", "")).strip()
    if method_id not in {"B4", "B5"}:
        raise ValueError("proposed structural tuning method_id must be B4 or B5")
    capacity_path = resolve_file_record(
        proposal.get("backward_capacity_report"),
        decision_manifest_path,
        "backward capacity report",
    )
    capacity_passed, capacity_details = validate_capacity_report(
        capacity_path,
        expected_method=method_id,
        backbone_lock=backbone_lock,
    )

    gate_checks = {
        "frozen_c3_benchmark_valid": True,
        "b0_through_b3_four_cohort_results_complete": True,
        "b2_beats_strong_b0b_sequence_baseline": suites["b2_vs_b0b"]["primary_support"]["claim_supported"],
        "b3_beats_b2_frozen_linear_probe": suites["b3_vs_b2"]["primary_support"]["claim_supported"],
        "b3_has_reviewed_scientific_limitation": limitation_passed,
        "structural_set_has_at_least_30_independent_post_cutoff_complexes": structural_set_passed,
        "structural_homology_audit_has_no_train_overlap": homology_passed,
        "structural_set_frozen_before_b4_b5": structural_frozen_before_tuning,
        "g1_real_forward_backward_optimizer_evidence_passes": g1_passed,
        "target_machine_supports_proposed_backward_graph": capacity_passed,
    }
    authorized = all(gate_checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "G4_permission_to_test_structural_tuning",
        "decision_id": decision_id,
        "authorized_method": method_id if authorized else None,
        "structural_tuning_authorized": authorized,
        "gate_checks": gate_checks,
        "input_sha256": {
            "decision_manifest": sha256_file(decision_manifest_path),
            "benchmark_freeze_manifest": freeze_sha256,
            "structural_set": sha256_file(structural_set_path),
            "homology_audit": sha256_file(homology_path),
            "backward_capacity_report": sha256_file(capacity_path),
        },
        "comparison_suite_paths": suite_paths,
        "b3_limitation": limitation_details,
        "structural_set": structural_set_details,
        "homology_audit": homology_details,
        "g1": g1_details,
        "backward_capacity": capacity_details,
        "interpretation": (
            "PASS authorizes only the named, evidence-bound B4/B5 experiment; it does not "
            "establish screening benefit or structural preservation."
            if authorized
            else "Structural-module tuning remains locked. Resolve failed checks without changing "
            "the frozen C3 benchmark or selecting evidence after inspecting B4/B5 results."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("G4 output already exists; use a new evidence-bound decision file")
    result = evaluate(args.decision_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["structural_tuning_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
