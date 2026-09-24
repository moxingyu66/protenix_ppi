#!/usr/bin/env python3
"""Resolve the first, conservative tier of PPI endpoint identifiers.

An identifier is accepted automatically only when its UniProt mapping has one
and only one reviewed human target in the pinned snapshot.  Isoform-specific,
unmapped, non-UniProt, and multiply reviewed cases remain quarantined.  The
result is an identifier-resolution table plus a pre-clustering protein table;
it is not yet a pair-label benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import sha256_file


RESOLUTION_FIELDS = [
    "source_database",
    "source_version",
    "evidence_role",
    "namespace",
    "raw_identifier",
    "mapping_query_id",
    "isoform_specific",
    "occurrence_count",
    "all_uniprot_targets",
    "reviewed_human_targets",
    "resolved_accession",
    "resolution_status",
    "resolution_reason",
]
PROTEIN_FIELDS = [
    "uniprot_accession",
    "gene_symbol",
    "taxid",
    "sequence",
    "sequence_length",
    "sequence_sha256",
    "uniprot_release",
    "sequence_version",
    "is_reviewed",
    "homology_cluster_30",
    "retrieved_at",
]


def load_mapping(path: Path) -> dict[str, set[str]]:
    mapping: defaultdict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["From", "To"]:
            raise ValueError(f"unexpected mapping columns in {path}: {reader.fieldnames}")
        for row in reader:
            source = row["From"].strip()
            target = row["To"].strip()
            if not source or not target:
                raise ValueError(f"empty mapping endpoint in {path}")
            mapping[source].add(target)
    return dict(mapping)


def load_reviewed_snapshot(path: Path, manifest_path: Path) -> tuple[dict[str, dict[str, str]], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["human_reviewed_snapshot"]
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError("reviewed UniProt snapshot does not match its manifest")
    records: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "Entry", "Reviewed", "Gene Names (primary)", "Organism (ID)",
            "Length", "Sequence", "Sequence version",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("reviewed UniProt snapshot is missing required fields")
        for line_number, row in enumerate(reader, start=2):
            accession = row["Entry"].strip()
            if accession in records:
                raise ValueError(f"duplicate UniProt accession at line {line_number}: {accession}")
            if row["Reviewed"].strip().lower() != "reviewed":
                raise ValueError(f"non-reviewed record in reviewed snapshot: {accession}")
            if row["Organism (ID)"].strip() != "9606":
                raise ValueError(f"non-human record in human snapshot: {accession}")
            sequence = row["Sequence"].strip().upper()
            if int(row["Length"]) != len(sequence):
                raise ValueError(f"sequence length mismatch for {accession}")
            records[accession] = row
    if len(records) != entry["data_rows"]:
        raise ValueError("reviewed UniProt snapshot row count does not match its manifest")
    return records, manifest


def resolve_one(
    inventory_row: dict[str, str],
    mapping: dict[str, set[str]],
    reviewed: set[str],
    current_records: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    original_status = inventory_row["mapping_status"].strip()
    query_id = inventory_row["mapping_query_id"].strip()
    all_targets = sorted(mapping.get(query_id, set())) if query_id else []
    reviewed_targets = sorted(set(all_targets) & reviewed)
    if original_status == "quarantine_isoform_specific":
        status = "quarantine"
        reason = "isoform_specific_source_not_folded_to_canonical"
        resolved = ""
    elif original_status == "manual_review_non_uniprot_primary":
        status = "manual_review"
        reason = "non_uniprot_primary_identifier"
        resolved = ""
    elif not all_targets:
        status = "quarantine"
        reason = "unmapped_in_pinned_uniprot_release"
        resolved = ""
    elif len(reviewed_targets) == 1:
        status = "accepted"
        reason = "unique_reviewed_human_target"
        resolved = reviewed_targets[0]
    elif not reviewed_targets:
        if current_records is None:
            status = "deferred"
            reason = "no_reviewed_human_target_requires_unreviewed_record_audit"
            resolved = ""
        else:
            known_targets = [target for target in all_targets if target in current_records]
            human_targets = [
                target
                for target in known_targets
                if current_records[target]["Organism (ID)"].strip() == "9606"
            ]
            if len(all_targets) == 1 and len(human_targets) == 1:
                status = "accepted"
                reason = "unique_current_human_target_unreviewed"
                resolved = human_targets[0]
            elif len(known_targets) < len(all_targets):
                status = "deferred"
                reason = "target_record_missing_from_pinned_snapshot"
                resolved = ""
            elif not human_targets:
                status = "excluded"
                reason = "no_human_target_in_pinned_release"
                resolved = ""
            else:
                status = "quarantine"
                reason = "multiple_or_cross_species_targets_without_unique_human_mapping"
                resolved = ""
    else:
        status = "quarantine"
        reason = "multiple_reviewed_human_targets"
        resolved = ""
    return {
        **{field: inventory_row[field].strip() for field in (
            "source_database", "source_version", "evidence_role", "namespace",
            "raw_identifier", "mapping_query_id", "isoform_specific", "occurrence_count",
        )},
        "all_uniprot_targets": ";".join(all_targets),
        "reviewed_human_targets": ";".join(reviewed_targets),
        "resolved_accession": resolved,
        "resolution_status": status,
        "resolution_reason": reason,
    }


def resolve_inventory(
    inventory_path: Path,
    ensembl_mapping_path: Path,
    uniprot_mapping_path: Path | None,
    reviewed_snapshot_path: Path,
    uniprot_manifest_path: Path,
    output_dir: Path,
    deferred_snapshot_path: Path | None = None,
    deferred_metadata_path: Path | None = None,
) -> dict:
    reviewed_records, uniprot_manifest = load_reviewed_snapshot(
        reviewed_snapshot_path, uniprot_manifest_path
    )
    current_records: dict[str, dict[str, str]] | None = None
    deferred_metadata = None
    if (deferred_snapshot_path is None) != (deferred_metadata_path is None):
        raise ValueError("deferred snapshot and metadata must be supplied together")
    if deferred_snapshot_path is not None and deferred_metadata_path is not None:
        deferred_metadata = json.loads(deferred_metadata_path.read_text(encoding="utf-8"))
        expected = deferred_metadata["records"]
        if sha256_file(deferred_snapshot_path) != expected["sha256"]:
            raise ValueError("deferred UniProt snapshot does not match its metadata")
        current_records = {}
        with deferred_snapshot_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required_deferred = {
                "Entry", "Reviewed", "Gene Names (primary)", "Organism (ID)",
                "Length", "Sequence", "Sequence version",
            }
            if not required_deferred.issubset(reader.fieldnames or []):
                raise ValueError("deferred UniProt snapshot is missing required fields")
            for row in reader:
                accession = row["Entry"].strip()
                sequence = row["Sequence"].strip().upper()
                if int(row["Length"]) != len(sequence):
                    raise ValueError(f"deferred sequence length mismatch for {accession}")
                current_records[accession] = row
        if len(current_records) != expected["count"]:
            raise ValueError("deferred UniProt snapshot row count does not match metadata")

    mappings = {
        "Ensembl": load_mapping(ensembl_mapping_path),
        "UniProtKB_AC-ID": (
            load_mapping(uniprot_mapping_path) if uniprot_mapping_path is not None else {}
        ),
    }
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "source_database", "source_version", "evidence_role", "namespace",
        "raw_identifier", "mapping_query_id", "isoform_specific", "mapping_status",
        "occurrence_count",
    }
    if inventory and not required.issubset(inventory[0]):
        raise ValueError("identifier inventory is missing required fields")

    rows = []
    for row in inventory:
        namespace = row["namespace"].strip()
        mapping = mappings.get(namespace, {})
        rows.append(resolve_one(row, mapping, set(reviewed_records), current_records))
    rows.sort(key=lambda row: tuple(row[field] for field in RESOLUTION_FIELDS[:8]))

    output_dir.mkdir(parents=True, exist_ok=True)
    resolution_path = output_dir / "identifier_resolution.tsv"
    quarantine_path = output_dir / "identifier_quarantine.tsv"
    with resolution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLUTION_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with quarantine_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLUTION_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in rows if row["resolution_status"] != "accepted")

    accepted_accessions = sorted({row["resolved_accession"] for row in rows if row["resolved_accession"]})
    proteins_path = output_dir / "proteins_precluster.csv"
    with proteins_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROTEIN_FIELDS, lineterminator="\n")
        writer.writeheader()
        for accession in accepted_accessions:
            record = reviewed_records.get(accession)
            is_reviewed = record is not None
            if record is None and current_records is not None:
                record = current_records.get(accession)
            if record is None:
                raise ValueError(f"accepted accession lacks a pinned record: {accession}")
            sequence = record["Sequence"].strip().upper()
            writer.writerow(
                {
                    "uniprot_accession": accession,
                    "gene_symbol": record["Gene Names (primary)"].strip(),
                    "taxid": "9606",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "uniprot_release": uniprot_manifest["release"],
                    "sequence_version": record["Sequence version"].strip(),
                    "is_reviewed": "true" if is_reviewed else "false",
                    "homology_cluster_30": "",
                    "retrieved_at": uniprot_manifest["retrieved_at"],
                }
            )

    reason_counts = Counter(row["resolution_reason"] for row in rows)
    status_counts = Counter(row["resolution_status"] for row in rows)
    source_accepted = Counter(
        row["source_database"] for row in rows if row["resolution_status"] == "accepted"
    )
    metadata = {
        "schema_version": "1.0",
        "method": (
            "resolve_identifier_mappings_v0.2"
            if current_records is not None
            else "resolve_identifier_mappings_v0.1"
        ),
        "uniprot_release": uniprot_manifest["release"],
        "policy": "accept_only_unique_reviewed_human_target_first_tier",
        "inputs": {
            "inventory_sha256": sha256_file(inventory_path),
            "ensembl_mapping_sha256": sha256_file(ensembl_mapping_path),
            "uniprot_mapping_sha256": (
                sha256_file(uniprot_mapping_path) if uniprot_mapping_path is not None else None
            ),
            "reviewed_snapshot_sha256": sha256_file(reviewed_snapshot_path),
            "uniprot_manifest_sha256": sha256_file(uniprot_manifest_path),
            "deferred_snapshot_sha256": (
                sha256_file(deferred_snapshot_path) if deferred_snapshot_path is not None else None
            ),
            "deferred_metadata_sha256": (
                sha256_file(deferred_metadata_path) if deferred_metadata_path is not None else None
            ),
        },
        "counts": {
            "inventory_rows": len(rows),
            "unique_accepted_proteins": len(accepted_accessions),
            "status": dict(sorted(status_counts.items())),
            "reason": dict(sorted(reason_counts.items())),
            "accepted_rows_by_source": dict(sorted(source_accepted.items())),
        },
        "outputs": {
            "identifier_resolution.tsv": {"sha256": sha256_file(resolution_path), "rows": len(rows)},
            "identifier_quarantine.tsv": {
                "sha256": sha256_file(quarantine_path),
                "rows": sum(row["resolution_status"] != "accepted" for row in rows),
            },
            "proteins_precluster.csv": {
                "sha256": sha256_file(proteins_path),
                "rows": len(accepted_accessions),
            },
        },
        "next_step": (
            "Construct traceable pair/evidence tables, resolve positive-negative contradictions, "
            "and preserve all quarantine rows outside the eligible benchmark."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ensembl-mapping", type=Path, required=True)
    parser.add_argument("--uniprot-mapping", type=Path)
    parser.add_argument("--reviewed-snapshot", type=Path, required=True)
    parser.add_argument("--uniprot-manifest", type=Path, required=True)
    parser.add_argument("--deferred-snapshot", type=Path)
    parser.add_argument("--deferred-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = resolve_inventory(
        args.inventory,
        args.ensembl_mapping,
        args.uniprot_mapping,
        args.reviewed_snapshot,
        args.uniprot_manifest,
        args.output_dir,
        args.deferred_snapshot,
        args.deferred_metadata,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
