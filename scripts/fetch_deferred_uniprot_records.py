#!/usr/bin/env python3
"""Fetch exact UniProt records needed to audit deferred mapping targets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.parse
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import sha256_file
from protenix_ppi.scripts.fetch_uniprot_mappings import (
    API_ROOT,
    _request,
    _validate_release_headers,
    read_release_evidence,
)


FIELDS = [
    "accession",
    "id",
    "reviewed",
    "gene_primary",
    "organism_id",
    "length",
    "sequence",
    "sequence_version",
    "xref_proteomes",
]


def deferred_targets(resolution_path: Path) -> list[str]:
    targets: set[str] = set()
    with resolution_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"resolution_status", "all_uniprot_targets"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("identifier resolution table is missing deferred-target fields")
        for row in reader:
            if row["resolution_status"] == "deferred":
                targets.update(item for item in row["all_uniprot_targets"].split(";") if item)
    if not targets:
        raise ValueError("identifier resolution contains no deferred targets")
    return sorted(targets)


def parse_record_tsv(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter="\t")
    if not reader.fieldnames or "Entry" not in reader.fieldnames:
        raise ValueError("UniProt record response has no Entry column")
    rows = list(reader)
    if any(not row["Entry"].strip() for row in rows):
        raise ValueError("UniProt record response contains an empty accession")
    return list(reader.fieldnames), rows


def fetch_records(
    resolution_path: Path,
    release_evidence_path: Path,
    output_dir: Path,
    batch_size: int,
) -> dict:
    if batch_size < 1 or batch_size > 200:
        raise ValueError("batch size must be between 1 and 200")
    release, release_date = read_release_evidence(release_evidence_path)
    targets = deferred_targets(resolution_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "target_ids.txt"
    records_path = output_dir / "records.tsv"
    missing_path = output_dir / "missing_target_ids.txt"
    metadata_path = output_dir / "metadata.json"
    if any(path.exists() for path in (request_path, records_path, missing_path, metadata_path)):
        raise ValueError(f"deferred-record output already exists: {output_dir}")

    request_path.write_text("\n".join(targets) + "\n", encoding="utf-8")
    records: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    api_headers = None
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        query = "(" + " OR ".join(f"accession:{item}" for item in batch) + ")"
        params = urllib.parse.urlencode(
            {"format": "tsv", "query": query, "fields": ",".join(FIELDS)}
        )
        payload, response_headers = _request(f"{API_ROOT}/uniprotkb/stream?{params}", timeout=300)
        _validate_release_headers(response_headers, release, release_date)
        batch_header, batch_rows = parse_record_tsv(payload)
        if header is None:
            header = batch_header
        elif batch_header != header:
            raise ValueError("UniProt record columns changed between batches")
        for row in batch_rows:
            accession = row["Entry"].strip()
            if accession not in set(batch):
                raise ValueError(f"UniProt returned an unrequested target: {accession}")
            if accession in records and records[accession] != row:
                raise ValueError(f"UniProt returned conflicting records for {accession}")
            records[accession] = row
        api_headers = response_headers

    if header is None:
        raise ValueError("no UniProt record response was received")
    records_partial = records_path.with_suffix(".tsv.partial")
    with records_partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    os.replace(records_partial, records_path)
    missing = sorted(set(targets) - set(records))
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    reviewed_counts: dict[str, int] = {}
    organism_counts: dict[str, int] = {}
    for row in records.values():
        reviewed = row["Reviewed"].strip().lower()
        organism = row["Organism (ID)"].strip()
        reviewed_counts[reviewed] = reviewed_counts.get(reviewed, 0) + 1
        organism_counts[organism] = organism_counts.get(organism, 0) + 1
    metadata = {
        "schema_version": "1.0",
        "method": "fetch_deferred_uniprot_records_v0.1",
        "uniprot_release": release,
        "uniprot_release_date": release_date,
        "resolution_input": {
            "path": str(resolution_path.resolve()),
            "sha256": sha256_file(resolution_path),
        },
        "release_evidence_sha256": sha256_file(release_evidence_path),
        "request": {"count": len(targets), "sha256": sha256_file(request_path)},
        "records": {
            "count": len(records),
            "sha256": sha256_file(records_path),
            "reviewed_counts": dict(sorted(reviewed_counts.items())),
            "organism_counts": dict(sorted(organism_counts.items())),
        },
        "missing": {"count": len(missing), "sha256": sha256_file(missing_path)},
        "api_headers": {
            "x-uniprot-release": (api_headers or {}).get("x-uniprot-release"),
            "x-uniprot-release-date": (api_headers or {}).get("x-uniprot-release-date"),
            "x-api-deployment-date": (api_headers or {}).get("x-api-deployment-date"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--release-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    metadata = fetch_records(
        args.resolution, args.release_evidence, args.output_dir, args.batch_size
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
