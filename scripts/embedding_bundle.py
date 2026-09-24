#!/usr/bin/env python3
"""Validate and load a frozen per-protein embedding bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
PRIMARY_POOLING = "mean_residue_tokens_after_overlap_average"


@dataclass(frozen=True)
class BundleSummary:
    sequence_count: int
    embedding_dim: int
    dtype: str
    model_id: str
    model_revision: str
    protein_table_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_bundle(
    bundle_dir: Path,
    proteins_path: Path,
) -> tuple[dict[str, np.ndarray], BundleSummary]:
    metadata_path = bundle_dir / "embedding_metadata.json"
    index_path = bundle_dir / "embedding_index.csv"
    embeddings_path = bundle_dir / "embeddings.npy"
    for path in (metadata_path, index_path, embeddings_path):
        if not path.is_file():
            raise ValueError(f"embedding bundle file is missing: {path.name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("embedding_metadata.json must contain an object")
    required_text = (
        "model_id",
        "model_revision",
        "pooling",
        "dtype",
        "protein_table_sha256",
        "tokenizer_version",
        "python_version",
        "torch_version",
        "transformers_version",
        "device",
        "compute_dtype",
        "created_at",
        "window_tail_policy",
    )
    for field_name in required_text:
        value = str(metadata.get(field_name, "")).strip()
        if not value or value == "REPLACE" or value.startswith("REPLACE_"):
            raise ValueError(f"embedding metadata field is missing: {field_name}")
    if not IMMUTABLE_REVISION.fullmatch(str(metadata["model_revision"]).lower()):
        raise ValueError("model_revision must be an immutable hexadecimal revision")
    if metadata["pooling"] != PRIMARY_POOLING:
        raise ValueError("embedding pooling policy disagrees with the frozen B0b contract")
    if metadata["dtype"] != "float32":
        raise ValueError("B0b embedding storage dtype must be float32")
    try:
        datetime.fromisoformat(str(metadata["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at is not a valid ISO-8601 timestamp") from exc
    if not HEX64.fullmatch(str(metadata["protein_table_sha256"])):
        raise ValueError("protein_table_sha256 is invalid")
    if metadata["protein_table_sha256"] != sha256_file(proteins_path):
        raise ValueError("protein table SHA-256 does not match embedding metadata")

    checkpoint_digests = metadata.get("checkpoint_files_sha256")
    if not isinstance(checkpoint_digests, dict) or not checkpoint_digests:
        raise ValueError("checkpoint_files_sha256 is missing")
    for filename, digest in checkpoint_digests.items():
        if not filename or not HEX64.fullmatch(str(digest)):
            raise ValueError("checkpoint_files_sha256 contains an invalid filename or digest")

    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy must be a two-dimensional array")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError("embeddings.npy must contain floating-point values")
    if str(embeddings.dtype) != str(metadata["dtype"]):
        raise ValueError("embedding dtype disagrees with metadata")
    try:
        expected_dim = int(metadata.get("embedding_dim", -1))
        expected_count = int(metadata.get("sequence_count", -1))
        layer = int(metadata.get("layer", -1))
        window_size = int(metadata.get("max_residues_per_window", -1))
        window_stride = int(metadata.get("window_stride", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding metadata contains invalid numeric fields") from exc
    if expected_dim <= 0 or expected_count <= 0 or layer <= 0:
        raise ValueError("embedding dimension, sequence count, and layer must be positive")
    if window_size <= 0 or not 0 < window_stride <= window_size:
        raise ValueError("embedding window size/stride metadata is invalid")
    if embeddings.shape[1] != expected_dim:
        raise ValueError("embedding dimension disagrees with metadata")
    if embeddings.shape[0] != expected_count:
        raise ValueError("embedding sequence count disagrees with metadata")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding array contains NaN or infinite values")

    protein_rows = read_csv(proteins_path)
    protein_hashes = {
        row["uniprot_accession"].strip(): row["sequence_sha256"].strip().lower()
        for row in protein_rows
    }
    index_rows = read_csv(index_path)
    if len(index_rows) != embeddings.shape[0]:
        raise ValueError("embedding index length does not match array rows")
    observed_indices: set[int] = set()
    observed_accessions: set[str] = set()
    mapping: dict[str, np.ndarray] = {}
    for line, row in enumerate(index_rows, start=2):
        accession = row.get("uniprot_accession", "").strip()
        sequence_hash = row.get("sequence_sha256", "").strip().lower()
        try:
            row_index = int(row.get("row_index", ""))
        except ValueError as exc:
            raise ValueError(f"embedding_index.csv:{line}: invalid row_index") from exc
        if accession in observed_accessions:
            raise ValueError(f"embedding_index.csv:{line}: duplicate accession {accession}")
        if row_index in observed_indices:
            raise ValueError(f"embedding_index.csv:{line}: duplicate row_index {row_index}")
        if accession not in protein_hashes:
            raise ValueError(f"embedding_index.csv:{line}: unknown accession {accession}")
        if sequence_hash != protein_hashes[accession]:
            raise ValueError(f"embedding_index.csv:{line}: sequence hash mismatch for {accession}")
        if not 0 <= row_index < embeddings.shape[0]:
            raise ValueError(f"embedding_index.csv:{line}: row_index out of bounds")
        observed_accessions.add(accession)
        observed_indices.add(row_index)
        mapping[accession] = np.asarray(embeddings[row_index], dtype=np.float64)
    if observed_indices != set(range(embeddings.shape[0])):
        raise ValueError("embedding row indices are not an exact permutation of array rows")
    if observed_accessions != set(protein_hashes):
        missing = sorted(set(protein_hashes) - observed_accessions)[:5]
        raise ValueError(f"embedding bundle does not cover every protein; missing {missing}")

    summary = BundleSummary(
        sequence_count=embeddings.shape[0],
        embedding_dim=embeddings.shape[1],
        dtype=str(embeddings.dtype),
        model_id=str(metadata["model_id"]),
        model_revision=str(metadata["model_revision"]),
        protein_table_sha256=str(metadata["protein_table_sha256"]),
    )
    return mapping, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--proteins", type=Path, required=True)
    args = parser.parse_args()
    _, summary = load_bundle(args.bundle_dir, args.proteins)
    print(json.dumps({**asdict(summary), "passed": True}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
