#!/usr/bin/env python3
"""Validate MMseqs2 cluster membership and populate homology_cluster_30."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import sha256_file


PARAMETERS = {
    "min_seq_id": 0.30,
    "coverage": 0.50,
    "cov_mode": 0,
    "alignment_mode": 3,
    "cluster_mode": 0,
    "sensitivity": 7.5,
}


def apply_clusters(
    proteins_path: Path,
    cluster_tsv_path: Path,
    output_path: Path,
    metadata_path: Path,
    mmseqs_version: str,
) -> dict:
    if output_path.exists() or metadata_path.exists():
        raise ValueError("clustered output already exists; preserve immutable split inputs")
    with proteins_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "homology_cluster_30" not in fieldnames:
        raise ValueError("proteins table lacks homology_cluster_30")
    proteins = {row["uniprot_accession"].strip(): row for row in rows}
    if len(proteins) != len(rows):
        raise ValueError("proteins table contains duplicate accessions")

    membership: dict[str, str] = {}
    with cluster_tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"MMseqs cluster line {line_number} must have two columns")
            representative, member = row[0].strip(), row[1].strip()
            if representative not in proteins or member not in proteins:
                raise ValueError(f"MMseqs cluster line {line_number} references an unknown accession")
            if member in membership:
                raise ValueError(f"MMseqs member appears more than once: {member}")
            membership[member] = representative
    missing = sorted(set(proteins) - set(membership))
    if missing:
        raise ValueError(f"MMseqs cluster output omits {len(missing)} proteins; first: {missing[0]}")

    for accession, row in proteins.items():
        row["homology_cluster_30"] = f"MMSEQ30_{membership[accession]}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(proteins[key] for key in sorted(proteins))
    sizes = Counter(membership.values())
    metadata = {
        "schema_version": "1.0",
        "method": "mmseqs2_easy_cluster_30pct_v0.1",
        "mmseqs_version": mmseqs_version.strip(),
        "parameters": PARAMETERS,
        "inputs": {
            "proteins_sha256": sha256_file(proteins_path),
            "cluster_tsv_sha256": sha256_file(cluster_tsv_path),
        },
        "counts": {
            "proteins": len(proteins),
            "clusters": len(sizes),
            "largest_cluster": max(sizes.values()),
            "singletons": sum(size == 1 for size in sizes.values()),
        },
        "output_sha256": sha256_file(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--cluster-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    args = parser.parse_args()
    metadata = apply_clusters(
        args.proteins, args.cluster_tsv, args.output, args.metadata, args.mmseqs_version
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
