#!/usr/bin/env python3
"""Validate the evidence-bound Protenix pair-feature hook lock for B2."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.embedding_bundle import sha256_file
from protenix_ppi.scripts.make_b1_predictions import PRIMARY_CUTOFF, PRIMARY_MODEL


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = "1.0"
EXPECTED_MODULE_CALL = "Protenix.get_pairformer_output"
EXPECTED_TENSOR = "z"
EXPECTED_CAPTURE_STAGE = "after_final_recycle_before_diffusion"
EXPECTED_POOLING = "cross_chain_direction_symmetrized_mean_and_std"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def require_digest(value: Any, source: str, pattern: re.Pattern[str] = HEX64) -> str:
    digest = str(value or "").strip().lower()
    if not pattern.fullmatch(digest):
        raise ValueError(f"{source} must be a lowercase hexadecimal digest")
    return digest


def require_text(value: Any, source: str, minimum: int = 1) -> str:
    text = str(value or "").strip()
    if not text or "REPLACE" in text.upper() or text.upper() in {"TODO", "TBD"} or len(text) < minimum:
        raise ValueError(f"{source} is missing or a placeholder")
    return text


def require_bool(value: Any, source: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source} must be boolean")
    return value


def require_positive_number(value: Any, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{source} must be finite and positive")
    return number


def require_nonnegative_number(value: Any, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{source} must be finite and nonnegative")
    return number


def require_positive_integer(value: Any, source: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{source} must be positive")
    return number


def validate_source_files(evidence: dict[str, Any]) -> dict[str, str]:
    files = evidence.get("source_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("hook evidence requires nonempty source_files")
    result: dict[str, str] = {}
    for path, digest in files.items():
        result[require_text(path, "hook evidence source file path")] = require_digest(
            digest, f"hook evidence source file {path} SHA-256"
        )
    required = {"protenix/model/protenix.py", "protenix/model/modules/pairformer.py"}
    if not required.issubset(result):
        raise ValueError("hook evidence must include protenix.py and pairformer.py source hashes")
    return result


def validate_hook_evidence(
    evidence: dict[str, Any],
    *,
    backbone_lock: dict[str, Any],
    g1_manifest: dict[str, Any],
) -> dict[str, Any]:
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("hook evidence schema_version is invalid")
    if evidence.get("model_name") != PRIMARY_MODEL or evidence.get("model_name") != backbone_lock.get("model_name"):
        raise ValueError("hook evidence uses the wrong primary model")
    if evidence.get("declared_training_cutoff") != PRIMARY_CUTOFF:
        raise ValueError("hook evidence uses the wrong training cutoff")
    source_commit = require_digest(evidence.get("source_commit"), "hook evidence source_commit", HEX40)
    if source_commit != backbone_lock.get("source_commit") or source_commit != g1_manifest.get("source_commit"):
        raise ValueError("hook evidence source_commit differs from G1/backbone lock")
    if require_digest(evidence.get("backbone_lock_sha256"), "hook evidence backbone_lock_sha256") != backbone_lock.get("_file_sha256"):
        raise ValueError("hook evidence is not bound to the supplied backbone lock")
    if require_digest(evidence.get("g1_manifest_sha256"), "hook evidence g1_manifest_sha256") != g1_manifest.get("_file_sha256"):
        raise ValueError("hook evidence is not bound to the supplied G1 manifest")
    if evidence.get("module_call") != EXPECTED_MODULE_CALL:
        raise ValueError("hook evidence module_call is not the registered Pairformer call")
    if evidence.get("candidate_tensor") != EXPECTED_TENSOR:
        raise ValueError("hook evidence candidate_tensor must be z")
    if evidence.get("capture_stage") != EXPECTED_CAPTURE_STAGE:
        raise ValueError("hook evidence capture_stage is not after_final_recycle_before_diffusion")
    require_bool(evidence.get("last_recycle_confirmed"), "hook evidence last_recycle_confirmed")
    if evidence["last_recycle_confirmed"] is not True:
        raise ValueError("hook evidence does not confirm the final recycle")
    require_bool(evidence.get("before_diffusion_confirmed"), "hook evidence before_diffusion_confirmed")
    if evidence["before_diffusion_confirmed"] is not True:
        raise ValueError("hook evidence does not confirm capture before diffusion")
    require_bool(evidence.get("label_blind"), "hook evidence label_blind")
    require_bool(evidence.get("labels_read"), "hook evidence labels_read")
    if evidence["label_blind"] is not True or evidence["labels_read"] is not False:
        raise ValueError("hook evidence is not label-blind")
    require_bool(evidence.get("split_assignments_read"), "hook evidence split_assignments_read")
    if evidence["split_assignments_read"] is not False:
        raise ValueError("hook evidence read split assignments during extraction")
    require_bool(evidence.get("use_msa"), "hook evidence use_msa")
    require_bool(evidence.get("use_template"), "hook evidence use_template")
    tensor_shape = evidence.get("tensor_shape_before_pooling")
    if not isinstance(tensor_shape, list) or len(tensor_shape) < 3:
        raise ValueError("hook evidence tensor_shape_before_pooling must have at least three dimensions")
    dimensions = [require_positive_integer(value, "hook evidence tensor shape") for value in tensor_shape[-3:]]
    if dimensions[0] != dimensions[1]:
        raise ValueError("hook evidence pair tensor is not square in token dimensions")
    c_z = require_positive_integer(evidence.get("c_z"), "hook evidence c_z")
    if dimensions[2] != c_z:
        raise ValueError("hook evidence c_z does not match tensor shape")
    if c_z != 128:
        raise ValueError("primary B2 hook lock currently requires c_z=128")
    swap = evidence.get("swap_invariance")
    if not isinstance(swap, dict):
        raise ValueError("hook evidence requires swap_invariance")
    pair_count = require_positive_integer(swap.get("pair_count"), "swap audit pair_count")
    tolerance = require_positive_number(swap.get("tolerance"), "swap audit tolerance")
    maximum_delta = require_nonnegative_number(swap.get("maximum_absolute_delta"), "swap audit maximum_absolute_delta")
    if maximum_delta > tolerance or swap.get("passed") is not True:
        raise ValueError("hook evidence swap-invariance audit failed")
    source_files = validate_source_files(evidence)
    return {
        "source_commit": source_commit,
        "tensor_shape_before_pooling": tensor_shape,
        "c_z": c_z,
        "swap_pair_count": pair_count,
        "swap_tolerance": tolerance,
        "swap_maximum_absolute_delta": maximum_delta,
        "source_files": source_files,
    }


def validate_lock(
    lock: dict[str, Any],
    *,
    lock_path: Path,
    backbone_lock: dict[str, Any],
    g1_manifest: dict[str, Any],
    hook_evidence: dict[str, Any],
) -> dict[str, Any]:
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("feature extraction lock schema_version is invalid")
    if lock.get("model_name") != PRIMARY_MODEL or lock.get("model_name") != backbone_lock.get("model_name"):
        raise ValueError("feature extraction lock uses the wrong primary model")
    if lock.get("declared_training_cutoff") != PRIMARY_CUTOFF:
        raise ValueError("feature extraction lock uses the wrong training cutoff")
    source_commit = require_digest(lock.get("source_commit"), "feature extraction lock source_commit", HEX40)
    if source_commit != backbone_lock.get("source_commit") or source_commit != g1_manifest.get("source_commit"):
        raise ValueError("feature extraction lock source_commit differs from G1/backbone lock")
    hook_hash = require_digest(lock.get("hook_evidence_sha256"), "feature extraction lock hook_evidence_sha256")
    if hook_hash != hook_evidence.get("_file_sha256"):
        raise ValueError("feature extraction lock hook evidence hash mismatch")
    g1_hash = require_digest(lock.get("g1_manifest_sha256"), "feature extraction lock g1_manifest_sha256")
    if g1_hash != g1_manifest.get("_file_sha256"):
        raise ValueError("feature extraction lock G1 manifest hash mismatch")
    if require_digest(lock.get("backbone_lock_sha256"), "feature extraction lock backbone_lock_sha256") != backbone_lock.get("_file_sha256"):
        raise ValueError("feature extraction lock backbone hash mismatch")
    require_digest(
        lock.get("inference_condition_lock_sha256"),
        "feature extraction lock inference_condition_lock_sha256",
    )
    require_text(lock.get("seed_policy"), "feature extraction lock seed_policy")
    require_text(lock.get("sample_policy"), "feature extraction lock sample_policy")
    require_bool(lock.get("label_blind"), "feature extraction lock label_blind")
    require_bool(lock.get("swap_invariant"), "feature extraction lock swap_invariant")
    if lock["label_blind"] is not True or lock["swap_invariant"] is not True:
        raise ValueError("feature extraction lock must be label-blind and swap-invariant")
    components = lock.get("feature_components")
    if not isinstance(components, list) or len(components) != 1:
        raise ValueError("primary feature extraction lock requires exactly one component")
    component = components[0]
    if not isinstance(component, dict):
        raise ValueError("feature component must be an object")
    if component.get("name") != "protenix_pair_z_cross_chain_mean_std":
        raise ValueError("feature component name is not the registered primary B2 feature")
    if component.get("source_tensor") != "Protenix.get_pairformer_output:return[2]:z":
        raise ValueError("feature component source tensor is not the registered z hook")
    if component.get("pooling") != EXPECTED_POOLING:
        raise ValueError("feature component pooling differs from the registered symmetric policy")
    audit = lock.get("swap_invariance_audit")
    if not isinstance(audit, dict):
        raise ValueError("feature extraction lock requires swap_invariance_audit")
    if require_positive_integer(audit.get("pair_count"), "feature lock swap pair_count") != hook_evidence["_swap_pair_count"]:
        raise ValueError("feature lock swap pair count differs from hook evidence")
    tolerance = require_positive_number(audit.get("tolerance"), "feature lock swap tolerance")
    maximum_delta = require_nonnegative_number(audit.get("maximum_absolute_delta"), "feature lock swap maximum_absolute_delta")
    if maximum_delta > tolerance or maximum_delta != hook_evidence["_swap_maximum_absolute_delta"]:
        raise ValueError("feature lock swap audit differs from hook evidence")
    return {
        "lock_sha256": sha256_file(lock_path),
        "source_commit": source_commit,
        "hook_evidence_sha256": hook_hash,
        "g1_manifest_sha256": g1_hash,
        "backbone_lock_sha256": lock.get("backbone_lock_sha256"),
        "feature_component": component["name"],
        "tensor_shape_before_pooling": hook_evidence["tensor_shape_before_pooling"],
        "c_z": hook_evidence["c_z"],
    }


def validate(
    lock_path: Path,
    backbone_path: Path,
    g1_manifest_path: Path,
    hook_evidence_path: Path,
) -> dict[str, Any]:
    lock = load_json(lock_path)
    backbone = load_json(backbone_path)
    g1_manifest = load_json(g1_manifest_path)
    hook = load_json(hook_evidence_path)
    backbone["_file_sha256"] = sha256_file(backbone_path)
    g1_manifest["_file_sha256"] = sha256_file(g1_manifest_path)
    hook["_file_sha256"] = sha256_file(hook_evidence_path)
    hook_summary = validate_hook_evidence(
        hook, backbone_lock=backbone, g1_manifest=g1_manifest
    )
    hook["_swap_pair_count"] = hook_summary["swap_pair_count"]
    hook["_swap_maximum_absolute_delta"] = hook_summary["swap_maximum_absolute_delta"]
    lock_summary = validate_lock(
        lock,
        lock_path=lock_path,
        backbone_lock=backbone,
        g1_manifest=g1_manifest,
        hook_evidence=hook,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "lock": lock_summary,
        "hook_evidence": hook_summary,
        "backbone_lock_sha256": backbone["_file_sha256"],
        "g1_manifest_sha256": g1_manifest["_file_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--backbone-lock", type=Path, required=True)
    parser.add_argument("--g1-manifest", type=Path, required=True)
    parser.add_argument("--hook-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("B2 hook validation output already exists")
    result = validate(args.lock, args.backbone_lock, args.g1_manifest, args.hook_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
