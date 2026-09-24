#!/usr/bin/env python3
"""Audit a frozen structural set against the C3 PPI training protein pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.embedding_bundle import sha256_file
from protenix_ppi.scripts.evaluate_structure_preservation import (
    SET_FIELDS,
    parse_mmcif_chain_sequences,
    resolve_hashed_file,
    validate_structural_set_provenance,
)


SCHEMA_VERSION = "1.0"
MMSEQS_PARAMETERS = {
    "min_seq_id": 0.30,
    "coverage": 0.50,
    "cov_mode": 0,
    "alignment_mode": 3,
    "sensitivity": 7.5,
}
HIT_FIELDS = ["query", "target", "pident", "qcov", "tcov", "evalue", "bits"]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=",", restval="")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=HIT_FIELDS, delimiter="\t")
        rows = list(reader)
    return rows


def parse_fraction(value: str, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid fraction") from exc
    if not math.isfinite(number):
        raise ValueError(f"{source}: fraction is not finite")
    if number > 1.0:
        number /= 100.0
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{source}: fraction outside [0, 1]")
    return number


def parse_hits(path: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=HIT_FIELDS, delimiter="\t")
        for line, row in enumerate(reader, start=1):
            if not row.get("query", "").strip() or not row.get("target", "").strip():
                raise ValueError(f"{path.name}:{line}: hit lacks query or target")
            try:
                evalue = float(row["evalue"])
                bits = float(row["bits"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line}: invalid evalue/bits") from exc
            if not math.isfinite(evalue) or not math.isfinite(bits):
                raise ValueError(f"{path.name}:{line}: evalue/bits must be finite")
            hits.append(
                {
                    "query": row["query"].strip(),
                    "target": row["target"].strip(),
                    "pident": parse_fraction(row["pident"], f"{path.name}:{line}:pident"),
                    "qcov": parse_fraction(row["qcov"], f"{path.name}:{line}:qcov"),
                    "tcov": parse_fraction(row["tcov"], f"{path.name}:{line}:tcov"),
                    "evalue": evalue,
                    "bits": bits,
                }
            )
    return hits


def build_mmseqs_command(
    mmseqs_bin: str,
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    temporary_dir: Path,
    threads: int,
) -> list[str]:
    return [
        mmseqs_bin,
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(temporary_dir),
        "--min-seq-id",
        str(MMSEQS_PARAMETERS["min_seq_id"]),
        "-c",
        str(MMSEQS_PARAMETERS["coverage"]),
        "--cov-mode",
        str(MMSEQS_PARAMETERS["cov_mode"]),
        "--alignment-mode",
        str(MMSEQS_PARAMETERS["alignment_mode"]),
        "-s",
        str(MMSEQS_PARAMETERS["sensitivity"]),
        "--format-output",
        ",".join(HIT_FIELDS),
        "--threads",
        str(threads),
    ]


def eligible_hits_by_query(
    hits: list[dict[str, Any]],
    query_to_accession: dict[str, str],
    training_accessions: set[str],
) -> dict[str, list[dict[str, Any]]]:
    by_query: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        if hit["query"] not in query_to_accession:
            raise ValueError(f"MMseqs hit references unknown query: {hit['query']}")
        if hit["target"] not in training_accessions:
            raise ValueError(f"MMseqs hit references a non-training target: {hit['target']}")
        # MMseqs -c 0.50/cov-mode 0 is relative to the longer sequence; require
        # both reported coverages as an auditable conservative check.
        if hit["pident"] >= MMSEQS_PARAMETERS["min_seq_id"] and min(hit["qcov"], hit["tcov"]) >= MMSEQS_PARAMETERS["coverage"]:
            by_query[hit["query"]].append(hit)
    return by_query


def build_training_fasta(
    proteins_path: Path, assignments_path: Path, output_path: Path
) -> tuple[set[str], dict[str, str]]:
    proteins = read_csv(proteins_path, {"uniprot_accession", "sequence", "sequence_sha256"})
    assignments = read_csv(assignments_path, {"uniprot_accession", "partition"})
    assignment_by_accession: dict[str, str] = {}
    for line, row in enumerate(assignments, start=2):
        accession = row["uniprot_accession"].strip()
        partition = row["partition"].strip()
        if not accession or accession in assignment_by_accession:
            raise ValueError(f"{assignments_path.name}:{line}: missing or duplicate accession")
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"{assignments_path.name}:{line}: invalid partition")
        assignment_by_accession[accession] = partition
    training = {accession for accession, partition in assignment_by_accession.items() if partition == "train"}
    if not training:
        raise ValueError("protein pool assignments contain no train proteins")
    sequence_by_accession: dict[str, str] = {}
    for line, row in enumerate(proteins, start=2):
        accession = row["uniprot_accession"].strip()
        sequence = row["sequence"].strip().upper()
        if not accession or accession in sequence_by_accession:
            raise ValueError(f"{proteins_path.name}:{line}: missing or duplicate accession")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"].strip().lower():
            raise ValueError(f"{proteins_path.name}:{line}: sequence SHA-256 mismatch")
        sequence_by_accession[accession] = sequence
    missing = sorted(training - set(sequence_by_accession))
    if missing:
        raise ValueError(f"training assignments reference missing proteins: {missing[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        for accession in sorted(training):
            sequence = sequence_by_accession[accession]
            if not sequence or any(char not in "ACDEFGHIKLMNPQRSTVWYUXOBZ" for char in sequence):
                raise ValueError(f"training protein has invalid sequence: {accession}")
            handle.write(f">{accession}\n{sequence}\n")
    return training, sequence_by_accession


def build_query_fasta(
    structural_set_path: Path, output_path: Path
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    rows = read_csv(structural_set_path, SET_FIELDS)
    query_to_accession: dict[str, str] = {}
    complexes: dict[str, dict[str, Any]] = {}
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        for line, row in enumerate(rows, start=2):
            complex_id = row["complex_id"].strip()
            reference_path = resolve_hashed_file(
                row["reference_structure"],
                row["reference_structure_sha256"],
                structural_set_path,
                f"{structural_set_path.name}:{line}:reference",
            )
            validate_structural_set_provenance(row, structural_set_path, line, reference_path)
            try:
                mapping = json.loads(row["protein_chain_mapping_json"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{complex_id}: invalid chain mapping JSON") from exc
            sequences = parse_mmcif_chain_sequences(reference_path)
            if set(mapping) != set(sequences):
                raise ValueError(f"{complex_id}: chain mapping and extracted sequence chains differ")
            chain_records = []
            for chain in sorted(mapping):
                accession = str(mapping[chain]).strip()
                query_id = f"{complex_id}__chain_{chain}"
                if query_id in query_to_accession:
                    raise ValueError(f"duplicate query ID: {query_id}")
                query_to_accession[query_id] = accession
                sequence = sequences[chain]
                if len(sequence) < 30:
                    raise ValueError(f"{complex_id}:{chain}: extracted sequence is shorter than 30 residues")
                handle.write(f">{query_id}\n{sequence}\n")
                chain_records.append(
                    {"chain": chain, "query": query_id, "accession": accession, "sequence_length": len(sequence)}
                )
            complexes[complex_id] = {"chains": chain_records}
    return query_to_accession, complexes


def audit_complexes(
    complexes: dict[str, dict[str, Any]],
    query_to_accession: dict[str, str],
    training_accessions: set[str],
    eligible_hits: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for complex_id in sorted(complexes):
        chain_results = []
        direct_overlap = False
        homology_overlap = False
        for chain in complexes[complex_id]["chains"]:
            query = chain["query"]
            accession = query_to_accession[query]
            direct = accession in training_accessions
            hits = sorted(
                eligible_hits.get(query, []),
                key=lambda hit: (-hit["pident"], -hit["bits"], hit["target"]),
            )
            direct_overlap = direct_overlap or direct
            homology_overlap = homology_overlap or bool(hits)
            chain_results.append(
                {
                    **chain,
                    "direct_training_accession_overlap": direct,
                    "eligible_training_homology_hits": hits,
                }
            )
        output.append(
            {
                "complex_id": complex_id,
                "ppi_train_protein_overlap": direct_overlap,
                "homology_cluster_overlap": homology_overlap,
                "chains": chain_results,
                "independence_passed": not direct_overlap and not homology_overlap,
            }
        )
    return output


def write_audited_structural_set(
    structural_set_path: Path,
    output_path: Path,
    complex_results: list[dict[str, Any]],
) -> None:
    """Write a new immutable copy with audit flags populated from this run."""
    with structural_set_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = set(SET_FIELDS)
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{structural_set_path.name} is missing columns: {', '.join(sorted(missing))}")
    result_by_id = {item["complex_id"]: item for item in complex_results}
    if set(result_by_id) != {row["complex_id"].strip() for row in rows}:
        raise ValueError("homology results do not cover the structural set exactly")
    for row in rows:
        result = result_by_id[row["complex_id"].strip()]
        row["homology_audit_passed"] = "true" if result["independence_passed"] else "false"
        row["ppi_train_overlap"] = "true" if result["ppi_train_protein_overlap"] else "false"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(
    structural_set_path: Path,
    proteins_path: Path,
    assignments_path: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
    *,
    mmseqs_bin: str = "mmseqs",
    threads: int = 8,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("structural homology audit output directory already exists")
    if threads < 1:
        raise ValueError("threads must be positive")
    try:
        freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read benchmark freeze manifest") from exc
    if freeze_manifest.get("status") != "frozen_before_model_scoring":
        raise ValueError("benchmark freeze manifest is not frozen_before_model_scoring")
    expected_proteins_hash = str(
        (freeze_manifest.get("inputs") or {}).get("proteins.clustered.csv", {}).get("sha256", "")
    ).strip().lower()
    expected_assignments_hash = str(
        (freeze_manifest.get("outputs") or {})
        .get("splits/c3_primary/protein_pool_assignments.csv", {})
        .get("sha256", "")
    ).strip().lower()
    if expected_proteins_hash != sha256_file(proteins_path):
        raise ValueError("PPI proteins hash differs from benchmark freeze manifest")
    if expected_assignments_hash != sha256_file(assignments_path):
        raise ValueError("protein-pool assignment hash differs from benchmark freeze manifest")
    output_dir.mkdir(parents=True)
    query_fasta = output_dir / "query_chains.fasta"
    target_fasta = output_dir / "ppi_train_proteins.fasta"
    hits_tsv = output_dir / "mmseqs_hits.tsv"
    temporary_dir = output_dir / "mmseqs_tmp"
    query_to_accession, complexes = build_query_fasta(structural_set_path, query_fasta)
    training_accessions, _ = build_training_fasta(proteins_path, assignments_path, target_fasta)
    command = build_mmseqs_command(mmseqs_bin, query_fasta, target_fasta, hits_tsv, temporary_dir, threads)
    command_text = " ".join(shlex.quote(part) for part in command) + "\n"
    command_path = output_dir / "command.txt"
    command_path.write_text(command_text, encoding="utf-8")
    version_command = [mmseqs_bin, "version"]
    try:
        version_result = subprocess.run(version_command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"failed to execute MMseqs2 version: {exc}") from exc
    mmseqs_version = version_result.stdout.strip() or version_result.stderr.strip()
    version_path = output_dir / "mmseqs_version.txt"
    version_path.write_text(mmseqs_version + "\n", encoding="utf-8")
    stdout_path = output_dir / "mmseqs_stdout_stderr.txt"
    try:
        with stdout_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT, text=True)
    except OSError as exc:
        raise ValueError(f"failed to execute MMseqs2 search: {exc}") from exc
    if result.returncode != 0:
        raise ValueError(f"MMseqs2 search failed with exit code {result.returncode}")
    if not hits_tsv.is_file():
        raise ValueError("MMseqs2 search did not produce a hit table")
    hits = parse_hits(hits_tsv)
    eligible_hits = eligible_hits_by_query(hits, query_to_accession, training_accessions)
    complex_results = audit_complexes(complexes, query_to_accession, training_accessions, eligible_hits)
    audited_set_path = output_dir / "structural_set_audited.csv"
    write_audited_structural_set(structural_set_path, audited_set_path, complex_results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "mmseqs2_structural_set_homology_audit_v0.1",
        "input_structural_set_sha256": sha256_file(structural_set_path),
        "structural_set_sha256": sha256_file(audited_set_path),
        "benchmark_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "ppi_train_proteins_sha256": sha256_file(proteins_path),
        "protein_pool_assignments_sha256": sha256_file(assignments_path),
        "tool": {
            "name": "MMseqs2",
            "version": mmseqs_version,
            "parameters": MMSEQS_PARAMETERS,
            "command": command,
            "command_evidence": {"path": command_path.name, "sha256": sha256_file(command_path)},
            "version_evidence": {"path": version_path.name, "sha256": sha256_file(version_path)},
            "stdout_stderr_evidence": {"path": stdout_path.name, "sha256": sha256_file(stdout_path)},
        },
        "thresholds": {
            "maximum_sequence_identity": MMSEQS_PARAMETERS["min_seq_id"],
            "minimum_coverage": MMSEQS_PARAMETERS["coverage"],
            "coverage_interpretation": "both reported query and target coverage must be at least 0.50",
        },
        "b4_b5_outputs_consulted": False,
        "counts": {
            "structural_complexes": len(complex_results),
            "query_chains": len(query_to_accession),
            "training_proteins": len(training_accessions),
            "raw_mmseqs_hits": len(hits),
            "eligible_homology_hits": sum(len(value) for value in eligible_hits.values()),
            "independent_complexes": sum(item["independence_passed"] for item in complex_results),
        },
        "complexes": complex_results,
        "outputs": {
            "structural_set_audited.csv": {
                "path": audited_set_path.name,
                "sha256": sha256_file(audited_set_path),
            },
            "query_chains.fasta": {"path": query_fasta.name, "sha256": sha256_file(query_fasta)},
            "ppi_train_proteins.fasta": {"path": target_fasta.name, "sha256": sha256_file(target_fasta)},
            "mmseqs_hits.tsv": {"path": hits_tsv.name, "sha256": sha256_file(hits_tsv)},
        },
    }
    report_path = output_dir / "structural_homology_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-set", type=Path, required=True)
    parser.add_argument("--ppi-proteins", type=Path, required=True)
    parser.add_argument("--protein-pool-assignments", type=Path, required=True)
    parser.add_argument("--benchmark-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result = audit(
        args.structural_set,
        args.ppi_proteins,
        args.protein_pool_assignments,
        args.benchmark_freeze_manifest,
        args.output_dir,
        mmseqs_bin=args.mmseqs_bin,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["counts"]["independent_complexes"] == result["counts"]["structural_complexes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
