#!/usr/bin/env python3
"""Validate a frozen, label-blind Protenix pair-feature bundle for B2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from protenix_ppi.scripts.make_b1_predictions import (
    PRIMARY_CUTOFF,
    PRIMARY_MODEL,
    validate_backbone_lock,
)


HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PairFeatureBundleSummary:
    pair_count: int
    feature_dim: int
    dtype: str
    model_name: str
    declared_training_cutoff: str
    feature_definition_version: str
    pairs_sha256: str
    splits_sha256: str
    backbone_lock_sha256: str
    feature_extraction_lock_sha256: str
    b2_hook_validation_sha256: str


@dataclass
class LoadedPairFeatureBundle:
    features: np.ndarray
    pair_to_row: dict[str, int]
    summary: PairFeatureBundleSummary

    def vector(self, pair_id: str) -> np.ndarray:
        if pair_id not in self.pair_to_row:
            raise KeyError(pair_id)
        return np.asarray(self.features[self.pair_to_row[pair_id]], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_vector(vector: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(vector)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def expected_split_pairs(
    pairs_path: Path,
    splits_path: Path,
    split_scheme: str,
    fold: str,
) -> dict[str, str]:
    pair_rows = read_csv(pairs_path)
    known_pairs = {
        row.get("pair_id", "").strip(): row
        for row in pair_rows
        if row.get("pair_id", "").strip()
    }
    expected: dict[str, str] = {}
    for line, row in enumerate(read_csv(splits_path), start=2):
        if row.get("split_scheme", "").strip() != split_scheme or row.get("fold", "").strip() != fold:
            continue
        pair_id = row.get("pair_id", "").strip()
        partition = row.get("partition", "").strip()
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"splits.csv:{line}: invalid partition {partition!r}")
        if not pair_id or pair_id in expected:
            raise ValueError(f"splits.csv:{line}: missing or duplicate pair_id {pair_id!r}")
        if pair_id not in known_pairs:
            raise ValueError(f"splits.csv:{line}: unknown pair_id {pair_id}")
        pair = known_pairs[pair_id]
        if pair.get("pair_status", "").strip() != "eligible" or pair.get("label", "").strip() not in {"0", "1"}:
            raise ValueError(f"splits.csv:{line}: {pair_id} is not an eligible labeled pair")
        expected[pair_id] = partition
    if not expected:
        raise ValueError(f"no rows found for split_scheme={split_scheme!r}, fold={fold!r}")
    return expected


def _require_nonplaceholder_text(mapping: dict, key: str, table_name: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value or value.upper().startswith("REPLACE"):
        raise ValueError(f"{table_name} field is missing: {key}")
    return value


def validate_hook_validation_report(
    report_path: Path,
    extraction_lock_path: Path,
    backbone_lock_path: Path,
) -> dict:
    """Fail closed unless the persisted G1 hook-validation report is self-consistent."""
    report = load_json(report_path)
    extraction_lock = load_json(extraction_lock_path)
    backbone_lock = load_json(backbone_lock_path)
    if report.get("schema_version") != "1.0" or report.get("status") != "passed":
        raise ValueError("B2 hook validation report must have schema_version=1.0 and status=passed")

    lock_summary = report.get("lock")
    hook_summary = report.get("hook_evidence")
    if not isinstance(lock_summary, dict) or not isinstance(hook_summary, dict):
        raise ValueError("B2 hook validation report is missing lock or hook_evidence summaries")

    expected_lock_hash = sha256_file(extraction_lock_path)
    expected_backbone_hash = sha256_file(backbone_lock_path)
    if str(lock_summary.get("lock_sha256", "")).strip().lower() != expected_lock_hash:
        raise ValueError("B2 hook validation report does not match feature_extraction_lock.json")
    if str(report.get("backbone_lock_sha256", "")).strip().lower() != expected_backbone_hash:
        raise ValueError("B2 hook validation report does not match the supplied backbone lock")
    if str(lock_summary.get("backbone_lock_sha256", "")).strip().lower() != expected_backbone_hash:
        raise ValueError("B2 hook validation lock summary has a different backbone hash")

    source_commit = str(extraction_lock.get("source_commit", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("feature extraction lock source_commit is invalid")
    if source_commit != str(backbone_lock.get("source_commit", "")).strip().lower():
        raise ValueError("feature extraction lock source_commit differs from the backbone lock")
    if source_commit != str(lock_summary.get("source_commit", "")).strip().lower():
        raise ValueError("B2 hook validation lock summary has a different source_commit")
    if source_commit != str(hook_summary.get("source_commit", "")).strip().lower():
        raise ValueError("B2 hook validation hook summary has a different source_commit")

    for field in ("g1_manifest_sha256", "hook_evidence_sha256"):
        expected = str(extraction_lock.get(field, "")).strip().lower()
        observed = str(lock_summary.get(field, "")).strip().lower()
        if not HEX64.fullmatch(expected) or observed != expected:
            raise ValueError(f"B2 hook validation report {field} is missing or inconsistent")
    if str(report.get("g1_manifest_sha256", "")).strip().lower() != str(
        extraction_lock.get("g1_manifest_sha256", "")
    ).strip().lower():
        raise ValueError("B2 hook validation report has a different G1 manifest hash")

    components = extraction_lock.get("feature_components")
    if not isinstance(components, list) or len(components) != 1 or not isinstance(components[0], dict):
        raise ValueError("validated B2 extraction lock must contain exactly one feature component")
    component_name = components[0].get("name")
    if component_name != "protenix_pair_z_cross_chain_mean_std":
        raise ValueError("validated B2 extraction lock does not use the registered z feature")
    if lock_summary.get("feature_component") != component_name:
        raise ValueError("B2 hook validation report has a different feature component")
    if hook_summary.get("c_z") != 128 or lock_summary.get("c_z") != 128:
        raise ValueError("B2 hook validation report must confirm c_z=128")
    tensor_shape = hook_summary.get("tensor_shape_before_pooling")
    if not isinstance(tensor_shape, list) or len(tensor_shape) < 3 or tensor_shape[-1] != 128:
        raise ValueError("B2 hook validation report has an invalid pair tensor shape")
    if lock_summary.get("tensor_shape_before_pooling") != tensor_shape:
        raise ValueError("B2 hook validation report tensor-shape summaries disagree")
    source_files = hook_summary.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("B2 hook validation report has no source-file hashes")
    for source_path, digest in source_files.items():
        if not str(source_path).strip() or not HEX64.fullmatch(str(digest).strip().lower()):
            raise ValueError("B2 hook validation report contains an invalid source-file hash")
    return report


def load_pair_feature_bundle(
    bundle_dir: Path,
    pairs_path: Path,
    splits_path: Path,
    backbone_lock_path: Path,
    split_scheme: str = "c3_primary",
    fold: str = "0",
) -> LoadedPairFeatureBundle:
    metadata_path = bundle_dir / "feature_metadata.json"
    index_path = bundle_dir / "feature_index.csv"
    features_path = bundle_dir / "features.npy"
    extraction_lock_path = bundle_dir / "feature_extraction_lock.json"
    hook_validation_path = bundle_dir / "b2_hook_validation.json"
    for path in (
        metadata_path,
        index_path,
        features_path,
        extraction_lock_path,
        hook_validation_path,
        backbone_lock_path,
    ):
        if not path.is_file():
            raise ValueError(f"pair-feature bundle file is missing: {path.name}")

    backbone = validate_backbone_lock(backbone_lock_path)
    extraction_lock = load_json(extraction_lock_path)
    metadata = load_json(metadata_path)
    if extraction_lock.get("model_name") != PRIMARY_MODEL or extraction_lock.get("declared_training_cutoff") != PRIMARY_CUTOFF:
        raise ValueError("feature extraction lock does not use the primary Protenix model/cutoff")
    if extraction_lock.get("label_blind") is not True:
        raise ValueError("feature extraction lock must declare label_blind=true")
    if extraction_lock.get("swap_invariant") is not True:
        raise ValueError("feature extraction lock must declare swap_invariant=true")
    components = extraction_lock.get("feature_components")
    if not isinstance(components, list) or not components:
        raise ValueError("feature extraction lock requires nonempty feature_components")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("each feature component must be an object")
        for key in ("name", "source_tensor", "pooling"):
            _require_nonplaceholder_text(component, key, "feature component")
    validate_hook_validation_report(
        hook_validation_path, extraction_lock_path, backbone_lock_path
    )

    expected_hashes = {
        "pairs_sha256": sha256_file(pairs_path),
        "splits_sha256": sha256_file(splits_path),
        "backbone_lock_sha256": sha256_file(backbone_lock_path),
        "feature_extraction_lock_sha256": sha256_file(extraction_lock_path),
        "b2_hook_validation_sha256": sha256_file(hook_validation_path),
    }
    for key, expected in expected_hashes.items():
        observed = str(metadata.get(key, "")).strip().lower()
        if observed != expected:
            raise ValueError(f"feature metadata {key} does not match the supplied file")
    if metadata.get("model_name") != backbone["model_name"]:
        raise ValueError("feature metadata/backbone model_name mismatch")
    if metadata.get("declared_training_cutoff") != backbone["declared_training_cutoff"]:
        raise ValueError("feature metadata/backbone cutoff mismatch")
    if metadata.get("checkpoint_sha256") != backbone["checkpoint_sha256"]:
        raise ValueError("feature metadata/backbone checkpoint mismatch")
    feature_definition_version = _require_nonplaceholder_text(
        metadata, "feature_definition_version", "feature metadata"
    )

    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.floating):
        raise ValueError("features.npy must be a two-dimensional floating-point array")
    if not np.isfinite(features).all():
        raise ValueError("features.npy contains NaN or infinite values")
    if str(features.dtype) != str(metadata.get("dtype", "")):
        raise ValueError("features.npy dtype disagrees with metadata")
    if features.shape != (int(metadata.get("pair_count", -1)), int(metadata.get("feature_dim", -1))):
        raise ValueError("features.npy shape disagrees with metadata")
    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != features.shape[1]:
        raise ValueError("feature_names must contain one name per feature column")
    normalized_names = [str(name).strip() for name in feature_names]
    if any(not name or name.upper().startswith("REPLACE") for name in normalized_names):
        raise ValueError("feature_names contains a missing/placeholder name")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("feature_names contains duplicates")

    expected = expected_split_pairs(pairs_path, splits_path, split_scheme, fold)
    index_rows = read_csv(index_path)
    required = {"row_index", "pair_id", "partition", "input_sha256", "feature_sha256"}
    if not index_rows or required - set(index_rows[0]):
        raise ValueError("feature_index.csv is empty or missing required columns")
    pair_to_row: dict[str, int] = {}
    observed_indices: set[int] = set()
    for line, row in enumerate(index_rows, start=2):
        pair_id = row.get("pair_id", "").strip()
        if pair_id in pair_to_row or pair_id not in expected:
            raise ValueError(f"feature_index.csv:{line}: duplicate or unexpected pair_id {pair_id!r}")
        try:
            row_index = int(row.get("row_index", ""))
        except ValueError as exc:
            raise ValueError(f"feature_index.csv:{line}: invalid row_index") from exc
        if row_index in observed_indices or not 0 <= row_index < features.shape[0]:
            raise ValueError(f"feature_index.csv:{line}: duplicate or out-of-range row_index")
        if row.get("partition", "").strip() != expected[pair_id]:
            raise ValueError(f"feature_index.csv:{line}: partition mismatch for {pair_id}")
        input_digest = row.get("input_sha256", "").strip().lower()
        feature_digest = row.get("feature_sha256", "").strip().lower()
        if not HEX64.fullmatch(input_digest):
            raise ValueError(f"feature_index.csv:{line}: invalid input_sha256")
        if not HEX64.fullmatch(feature_digest) or feature_digest != sha256_vector(features[row_index]):
            raise ValueError(f"feature_index.csv:{line}: feature_sha256 mismatch")
        pair_to_row[pair_id] = row_index
        observed_indices.add(row_index)
    if set(pair_to_row) != set(expected):
        missing = sorted(set(expected) - set(pair_to_row))[:5]
        raise ValueError(f"pair-feature bundle does not cover the frozen split; missing {missing}")
    if observed_indices != set(range(features.shape[0])):
        raise ValueError("feature row indices are not an exact permutation of array rows")

    summary = PairFeatureBundleSummary(
        pair_count=features.shape[0],
        feature_dim=features.shape[1],
        dtype=str(features.dtype),
        model_name=str(metadata["model_name"]),
        declared_training_cutoff=str(metadata["declared_training_cutoff"]),
        feature_definition_version=feature_definition_version,
        **expected_hashes,
    )
    return LoadedPairFeatureBundle(features=features, pair_to_row=pair_to_row, summary=summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--backbone-lock", type=Path, required=True)
    parser.add_argument("--split-scheme", default="c3_primary")
    parser.add_argument("--fold", default="0")
    args = parser.parse_args()
    bundle = load_pair_feature_bundle(
        args.bundle_dir,
        args.pairs,
        args.splits,
        args.backbone_lock,
        args.split_scheme,
        args.fold,
    )
    print(json.dumps({**asdict(bundle.summary), "passed": True}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
