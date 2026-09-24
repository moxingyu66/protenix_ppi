#!/usr/bin/env python3
"""Validate the minimum evidence bundle for the Protenix-PPI G1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PRIMARY_MODEL = "protenix_base_default_v1.0.0"
PRIMARY_CUTOFF = "2021-09-30"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lock(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if lock.get("repository_url") != "https://github.com/bytedance/Protenix.git":
        errors.append("backbone lock repository_url is not the official Protenix repository")
    if not HEX40.fullmatch(str(lock.get("source_commit", ""))):
        errors.append("backbone lock source_commit must be a 40-character lowercase hexadecimal commit")
    if lock.get("model_name") != PRIMARY_MODEL:
        errors.append(f"primary model must be {PRIMARY_MODEL}")
    if lock.get("declared_training_cutoff") != PRIMARY_CUTOFF:
        errors.append(f"declared training cutoff must be {PRIMARY_CUTOFF}")
    if not HEX64.fullmatch(str(lock.get("checkpoint_sha256", ""))):
        errors.append("checkpoint_sha256 must be a 64-character lowercase hexadecimal digest")
    if lock.get("g0_route") not in {"A", "B", "C"}:
        errors.append("g0_route must be A, B, or C for a primary-model G1 run")
    for key in ("package_version", "python_version", "torch_version", "cuda_runtime", "nvidia_driver"):
        value = str(lock.get(key, "")).strip()
        if not value or value == "REPLACE":
            errors.append(f"backbone lock field is missing: {key}")
    return errors


def validate_gradient_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    loss = summary.get("loss_value")
    if not summary.get("loss_finite") or not isinstance(loss, (int, float)) or not math.isfinite(loss):
        errors.append("backward loss is not recorded as finite")
    if int(summary.get("trainable_parameter_count", 0)) <= 0:
        errors.append("no trainable parameters were recorded")
    if int(summary.get("parameters_with_finite_nonzero_gradients", 0)) <= 0:
        errors.append("no finite nonzero gradient was recorded")
    if int(summary.get("frozen_parameters_with_gradients", 0)) != 0:
        errors.append("one or more frozen parameters received gradients")
    if not summary.get("optimizer_step_completed"):
        errors.append("optimizer step was not completed")
    if int(summary.get("peak_gpu_memory_mib", 0)) <= 0:
        errors.append("peak GPU memory was not recorded")
    return errors


def resolve_required_file(run_dir: Path, relative_value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(relative_value, str) or not relative_value.strip():
        errors.append(f"manifest is missing {label}")
        return None
    candidate = (run_dir / relative_value).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        errors.append(f"{label} escapes the G1 run directory")
        return None
    if not candidate.is_file():
        errors.append(f"required {label} does not exist: {relative_value}")
        return None
    return candidate


def validate_run(run_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = run_dir / "manifest.json"
    lock_path = run_dir / "backbone_lock.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    if not lock_path.is_file():
        return ["backbone_lock.json is missing"]

    manifest = load_json(manifest_path)
    lock = load_json(lock_path)
    errors.extend(validate_lock(lock))

    if manifest.get("model_name") != lock.get("model_name"):
        errors.append("manifest model_name does not match backbone lock")
    if manifest.get("source_commit") != lock.get("source_commit"):
        errors.append("manifest source_commit does not match backbone lock")
    if manifest.get("checkpoint_sha256") != lock.get("checkpoint_sha256"):
        errors.append("manifest checkpoint digest does not match backbone lock")
    if int(manifest.get("exit_code", -1)) != 0:
        errors.append("inference exit code is not zero")
    if float(manifest.get("wall_time_seconds", 0)) <= 0:
        errors.append("positive wall_time_seconds was not recorded")
    if not HEX64.fullmatch(str(manifest.get("input_sha256", ""))):
        errors.append("manifest input_sha256 is invalid")

    for key, label in (
        ("command_file", "command log"),
        ("stdout_file", "stdout/stderr log"),
        ("telemetry_file", "GPU telemetry"),
    ):
        resolve_required_file(run_dir, manifest.get(key), label, errors)

    gradient_path = resolve_required_file(
        run_dir, manifest.get("gradient_summary_file"), "gradient summary", errors
    )
    if gradient_path is not None:
        errors.extend(validate_gradient_summary(load_json(gradient_path)))

    output_dir = run_dir / "outputs"
    structures = list(output_dir.rglob("*.cif")) + list(output_dir.rglob("*.pdb")) if output_dir.is_dir() else []
    if not structures:
        errors.append("no predicted .cif or .pdb structure found under outputs/")

    confidence_files = list(output_dir.rglob("*confidence*.json")) if output_dir.is_dir() else []
    if not confidence_files:
        errors.append("no confidence JSON found under outputs/")
    else:
        for path in confidence_files:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid confidence JSON {path.relative_to(run_dir)}: {exc}")

    input_candidates = list((run_dir / "inputs").glob("*.json")) if (run_dir / "inputs").is_dir() else []
    if len(input_candidates) != 1:
        errors.append("exactly one frozen input JSON is required under inputs/")
    elif HEX64.fullmatch(str(manifest.get("input_sha256", ""))):
        if sha256_file(input_candidates[0]) != manifest["input_sha256"]:
            errors.append("input JSON SHA-256 does not match manifest")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()
    errors = validate_run(run_dir)
    if errors:
        print("G1 VALIDATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("G1 VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

