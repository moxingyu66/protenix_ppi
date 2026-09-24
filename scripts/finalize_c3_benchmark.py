#!/usr/bin/env python3
"""Validate MMseqs2 evidence and atomically freeze the primary C3 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.apply_mmseqs_clusters import PARAMETERS as MMSEQS_PARAMETERS
from protenix_ppi.scripts.build_evaluation_cohorts import (
    ALGORITHM_VERSION as COHORT_ALGORITHM_VERSION,
    build_cohorts,
)
from protenix_ppi.scripts.embedding_bundle import sha256_file
from protenix_ppi.scripts.make_c3_split import (
    ALGORITHM_VERSION as C3_ALGORITHM_VERSION,
    generate,
)
from protenix_ppi.scripts.validate_dataset import validate_tables


COHORT_SEED = 20260915
SEED_START = 0
SEED_END = 999
MIN_PER_CLASS = 1
FREEZE_SCHEMA_VERSION = "1.0"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader)


def _require_hash(mapping: dict[str, Any], key: str, expected: str, source: str) -> None:
    observed = str(mapping.get(key, "")).strip().lower()
    if observed != expected:
        raise ValueError(f"{source} {key} does not match the supplied file")


def validate_mmseqs_evidence(dataset_dir: Path) -> dict[str, Any]:
    original_proteins = dataset_dir / "proteins.csv"
    mmseqs_dir = dataset_dir / "mmseqs30"
    clustered_proteins = mmseqs_dir / "proteins.clustered.csv"
    cluster_tsv = mmseqs_dir / "mmseqs30_cluster.tsv"
    cluster_metadata_path = mmseqs_dir / "mmseqs_cluster_metadata.json"
    fasta_path = mmseqs_dir / "proteins.fasta"
    fasta_metadata_path = mmseqs_dir / "fasta_metadata.json"
    log_path = mmseqs_dir / "mmseqs.log"
    required = (
        original_proteins,
        clustered_proteins,
        cluster_tsv,
        cluster_metadata_path,
        fasta_path,
        fasta_metadata_path,
        log_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("MMseqs2 evidence is incomplete; missing: " + ", ".join(missing))

    cluster_metadata = load_json(cluster_metadata_path)
    fasta_metadata = load_json(fasta_metadata_path)
    if cluster_metadata.get("method") != "mmseqs2_easy_cluster_30pct_v0.1":
        raise ValueError("unexpected MMseqs2 clustering method")
    if cluster_metadata.get("parameters") != MMSEQS_PARAMETERS:
        raise ValueError("MMseqs2 parameters differ from the frozen C3 protocol")
    mmseqs_version = str(cluster_metadata.get("mmseqs_version", "")).strip()
    if not mmseqs_version or mmseqs_version.upper().startswith("REPLACE"):
        raise ValueError("MMseqs2 version is missing")

    _require_hash(
        cluster_metadata.get("inputs", {}),
        "proteins_sha256",
        sha256_file(original_proteins),
        "MMseqs2 metadata",
    )
    _require_hash(
        cluster_metadata.get("inputs", {}),
        "cluster_tsv_sha256",
        sha256_file(cluster_tsv),
        "MMseqs2 metadata",
    )
    _require_hash(
        cluster_metadata,
        "output_sha256",
        sha256_file(clustered_proteins),
        "MMseqs2 metadata",
    )
    _require_hash(
        fasta_metadata,
        "proteins_sha256",
        sha256_file(original_proteins),
        "FASTA metadata",
    )
    _require_hash(
        fasta_metadata,
        "fasta_sha256",
        sha256_file(fasta_path),
        "FASTA metadata",
    )

    originals = read_csv(original_proteins)
    clustered = read_csv(clustered_proteins)
    if len(originals) != len(clustered):
        raise ValueError("clustered protein table row count differs from proteins.csv")
    original_by_id = {row.get("uniprot_accession", "").strip(): row for row in originals}
    clustered_by_id = {row.get("uniprot_accession", "").strip(): row for row in clustered}
    if len(original_by_id) != len(originals) or set(original_by_id) != set(clustered_by_id):
        raise ValueError("clustered protein accession set differs from proteins.csv")
    for accession, original in original_by_id.items():
        candidate = clustered_by_id[accession]
        cluster_id = candidate.get("homology_cluster_30", "").strip()
        if not cluster_id:
            raise ValueError(f"clustered protein lacks homology_cluster_30: {accession}")
        for field, value in original.items():
            if field == "homology_cluster_30":
                continue
            if candidate.get(field) != value:
                raise ValueError(f"clustered protein changed field {field!r} for {accession}")
    if int(fasta_metadata.get("sequence_count", -1)) != len(originals):
        raise ValueError("FASTA metadata sequence count differs from proteins.csv")

    return {
        "mmseqs_version": mmseqs_version,
        "parameters": MMSEQS_PARAMETERS,
        "counts": cluster_metadata.get("counts"),
        "files": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (
                fasta_path,
                fasta_metadata_path,
                cluster_tsv,
                log_path,
                clustered_proteins,
                cluster_metadata_path,
            )
        },
    }


def _file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        record["rows"] = len(read_csv(path))
    return record


def freeze(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    pairs_path = dataset_dir / "pairs.csv"
    evidence_path = dataset_dir / "evidence.csv"
    normalized_metadata_path = dataset_dir / "metadata.json"
    clustered_proteins = dataset_dir / "mmseqs30" / "proteins.clustered.csv"
    for path in (pairs_path, evidence_path, normalized_metadata_path):
        if not path.is_file():
            raise ValueError(f"required normalized dataset file is missing: {path}")

    split_dir = dataset_dir / "splits" / "c3_primary"
    cohort_dir = dataset_dir / "evaluation_cohorts"
    manifest_path = dataset_dir / "benchmark_freeze_manifest.json"
    validation_path = dataset_dir / "c3_validation.json"
    targets = (split_dir, cohort_dir, manifest_path, validation_path)
    if any(path.exists() for path in targets):
        raise ValueError(
            "C3 freeze outputs already exist; never regenerate a benchmark after scoring"
        )

    normalized_metadata = load_json(normalized_metadata_path)
    outputs = normalized_metadata.get("outputs", {})
    for filename, path in (
        ("proteins.csv", dataset_dir / "proteins.csv"),
        ("pairs.csv", pairs_path),
        ("evidence.csv", evidence_path),
    ):
        expected = str(outputs.get(filename, {}).get("sha256", "")).strip().lower()
        if expected != sha256_file(path):
            raise ValueError(f"normalized metadata hash mismatch for {filename}")

    mmseqs_evidence = validate_mmseqs_evidence(dataset_dir)
    protocol_root = Path(__file__).resolve().parents[1] / "protocol"
    with tempfile.TemporaryDirectory(prefix="g2_freeze_", dir=dataset_dir) as temporary:
        staging = Path(temporary)
        staged_splits = staging / "c3_primary"
        staged_cohorts = staging / "evaluation_cohorts"
        stats = generate(
            clustered_proteins,
            pairs_path,
            staged_splits,
            seed_start=SEED_START,
            seed_end=SEED_END,
            min_per_class=MIN_PER_CLASS,
        )
        validation = validate_tables(
            clustered_proteins,
            pairs_path,
            evidence_path,
            staged_splits / "splits.csv",
        )
        if not validation.passed:
            raise ValueError(
                "generated C3 benchmark failed validation:\n- " + "\n- ".join(validation.errors)
            )
        cohort_metadata = build_cohorts(
            pairs_path,
            staged_splits / "splits.csv",
            staged_cohorts,
            seed=COHORT_SEED,
        )
        validation_payload = {**asdict(validation), "passed": validation.passed}

        output_files: dict[str, dict[str, Any]] = {}
        for base_name, directory in (
            ("splits/c3_primary", staged_splits),
            ("evaluation_cohorts", staged_cohorts),
        ):
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    output_files[f"{base_name}/{path.name}"] = _file_record(path)

        manifest = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "status": "frozen_before_model_scoring",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_dir_name": dataset_dir.name,
            "algorithms": {
                "c3": C3_ALGORITHM_VERSION,
                "evaluation_cohorts": COHORT_ALGORITHM_VERSION,
                "split_seed_search": {"start": SEED_START, "end": SEED_END},
                "minimum_per_class_per_partition": MIN_PER_CLASS,
                "cohort_seed": COHORT_SEED,
            },
            "protocol_sha256": {
                "c3_split_protocol_v0.1.md": sha256_file(
                    protocol_root / "c3_split_protocol_v0.1.md"
                ),
                "evaluation_protocol_v0.2.md": sha256_file(
                    protocol_root / "evaluation_protocol_v0.2.md"
                ),
                "data_contract_v0.1.md": sha256_file(
                    protocol_root / "data_contract_v0.1.md"
                ),
            },
            "inputs": {
                "proteins.csv": _file_record(dataset_dir / "proteins.csv"),
                "proteins.clustered.csv": _file_record(clustered_proteins),
                "pairs.csv": _file_record(pairs_path),
                "evidence.csv": _file_record(evidence_path),
                "metadata.json": _file_record(normalized_metadata_path),
            },
            "mmseqs2": mmseqs_evidence,
            "c3_stats": asdict(stats),
            "cohort_counts": cohort_metadata["counts"],
            "validation": validation_payload,
            "outputs": output_files,
            "invariants": [
                "No unobserved pair is labeled negative.",
                "Homology clusters are atomic across train/validation/test.",
                "Cross-pool complex evidence groups are quarantined atomically.",
                "Cohort membership was generated without model scores.",
                "Balanced cohorts contain only explicit negatives and deterministic positives.",
            ],
        }

        split_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_splits.replace(split_dir)
        staged_cohorts.replace(cohort_dir)
        validation_path.write_text(
            json.dumps(validation_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["outputs"]["c3_validation.json"] = _file_record(validation_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.dataset_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
