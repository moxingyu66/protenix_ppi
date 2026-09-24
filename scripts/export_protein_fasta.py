#!/usr/bin/env python3
"""Export validated canonical protein sequences to deterministic FASTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import sha256_file


def export_fasta(proteins_path: Path, fasta_path: Path, metadata_path: Path) -> dict:
    if fasta_path.exists() or metadata_path.exists():
        raise ValueError("FASTA output already exists; preserve immutable clustering inputs")
    with proteins_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("proteins table is empty")
    seen = set()
    entries = []
    for line_number, row in enumerate(rows, start=2):
        accession = row["uniprot_accession"].strip()
        sequence = row["sequence"].strip().upper()
        if not accession or accession in seen:
            raise ValueError(f"missing or duplicate accession at proteins line {line_number}")
        seen.add(accession)
        observed_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if observed_hash != row["sequence_sha256"].strip().lower():
            raise ValueError(f"sequence hash mismatch for {accession}")
        if len(sequence) != int(row["sequence_length"]):
            raise ValueError(f"sequence length mismatch for {accession}")
        entries.append((accession, sequence))
    entries.sort()
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("x", encoding="ascii", newline="\n") as handle:
        for accession, sequence in entries:
            handle.write(f">{accession}\n{sequence}\n")
    metadata = {
        "schema_version": "1.0",
        "method": "export_protein_fasta_v0.1",
        "proteins_path": str(proteins_path.resolve()),
        "proteins_sha256": sha256_file(proteins_path),
        "sequence_count": len(entries),
        "fasta_path": str(fasta_path.resolve()),
        "fasta_sha256": sha256_file(fasta_path),
        "header_policy": "exact_uniprot_accession_only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_fasta(args.proteins, args.fasta, args.metadata), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
