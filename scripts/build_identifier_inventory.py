#!/usr/bin/env python3
"""Build a deterministic identifier inventory before UniProt normalization.

This stage deliberately does not choose biological labels or guess identifier
mappings.  It extracts the endpoint identifiers that must be resolved from the
pinned HuRI, IntAct, and Negatome evidence sources and separates isoform and
non-UniProt cases for quarantine/manual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENSEMBL_GENE_RE = re.compile(r"^ENSG\d{11}$")
UNIPROT_ISOFORM_RE = re.compile(r"^([A-Z0-9]{6,10})-(\d+)$")
INVENTORY_FIELDS = [
    "source_database",
    "source_version",
    "evidence_role",
    "namespace",
    "raw_identifier",
    "mapping_query_id",
    "isoform_specific",
    "mapping_status",
    "occurrence_count",
]


@dataclass(frozen=True)
class InventoryKey:
    source_database: str
    source_version: str
    evidence_role: str
    namespace: str
    raw_identifier: str
    mapping_query_id: str
    isoform_specific: str
    mapping_status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_from_audit(audit: dict, source_id: str) -> tuple[Path, dict, dict]:
    if audit.get("status") != "pass":
        raise ValueError("raw-source audit must pass before identifier extraction")
    item = audit.get("integrity", {}).get(source_id)
    if not item or item.get("status") != "pass":
        raise ValueError(f"raw-source audit does not pass for {source_id}")
    path = Path(item["path"])
    if not path.is_file():
        raise ValueError(f"audited source is missing: {path}")
    observed = item.get("observed", {})
    if path.stat().st_size != observed.get("bytes"):
        raise ValueError(f"audited byte count changed for {source_id}")
    if sha256_file(path) != observed.get("sha256"):
        raise ValueError(f"audited SHA-256 changed for {source_id}")
    manifest = json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
    return path, manifest, item


def _split_database_identifier(raw: str) -> tuple[str, str]:
    token = raw.strip().split("|", 1)[0]
    if ":" not in token:
        return "unknown", token
    namespace, identifier = token.split(":", 1)
    return namespace.lower(), identifier.strip()


def _uniprot_key(
    source_database: str,
    source_version: str,
    evidence_role: str,
    raw_identifier: str,
) -> InventoryKey:
    match = UNIPROT_ISOFORM_RE.fullmatch(raw_identifier)
    if match:
        return InventoryKey(
            source_database,
            source_version,
            evidence_role,
            "UniProtKB_AC-ID",
            raw_identifier,
            match.group(1),
            "true",
            "quarantine_isoform_specific",
        )
    return InventoryKey(
        source_database,
        source_version,
        evidence_role,
        "UniProtKB_AC-ID",
        raw_identifier,
        raw_identifier,
        "false",
        "mapping_candidate",
    )


def iter_huri(path: Path, version: str) -> Iterable[InventoryKey]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"HuRI line {line_number} does not contain two columns")
            for identifier in row:
                identifier = identifier.strip()
                if not ENSEMBL_GENE_RE.fullmatch(identifier):
                    raise ValueError(f"HuRI line {line_number} has invalid Ensembl gene {identifier!r}")
                yield InventoryKey(
                    "HuRI",
                    version,
                    "positive_evidence",
                    "Ensembl",
                    identifier,
                    identifier,
                    "false",
                    "mapping_candidate",
                )


def iter_negatome(path: Path, version: str) -> Iterable[InventoryKey]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 4:
                raise ValueError(f"Negatome line {line_number} does not contain four columns")
            for identifier in row[:2]:
                yield _uniprot_key("Negatome 2.0", version, "negative_evidence", identifier.strip())


def iter_intact_candidates(path: Path, version: str) -> Iterable[InventoryKey]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"primary_id_a", "primary_id_b", "decision"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("IntAct candidate table is missing identifier columns")
        for line_number, row in enumerate(reader, start=2):
            if row["decision"] != "candidate":
                raise ValueError(f"IntAct candidate line {line_number} is not marked candidate")
            for field in ("primary_id_a", "primary_id_b"):
                namespace, identifier = _split_database_identifier(row[field])
                if namespace != "uniprotkb" or not identifier:
                    raise ValueError(f"IntAct candidate line {line_number} has non-UniProt primary ID")
                yield _uniprot_key("IntAct", version, "positive_evidence", identifier)


def _is_human(raw: str) -> bool:
    return "taxid:9606" in raw


def _is_protein(raw: str) -> bool:
    return "MI:0326" in raw


def iter_intact_negatives(path: Path, version: str) -> Iterable[InventoryKey]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "#ID(s) interactor A",
            "ID(s) interactor B",
            "Taxid interactor A",
            "Taxid interactor B",
            "Type(s) interactor A",
            "Type(s) interactor B",
            "Negative",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("IntAct negative table is missing required MITAB columns")
        for row in reader:
            if row["Negative"].strip().lower() != "true":
                continue
            if not (_is_human(row["Taxid interactor A"]) and _is_human(row["Taxid interactor B"])):
                continue
            if not (_is_protein(row["Type(s) interactor A"]) and _is_protein(row["Type(s) interactor B"])):
                continue
            for field in ("#ID(s) interactor A", "ID(s) interactor B"):
                namespace, identifier = _split_database_identifier(row[field])
                if namespace == "uniprotkb" and identifier:
                    yield _uniprot_key("IntAct", version, "negative_evidence", identifier)
                else:
                    yield InventoryKey(
                        "IntAct",
                        version,
                        "negative_evidence",
                        namespace or "unknown",
                        identifier,
                        "",
                        "false",
                        "manual_review_non_uniprot_primary",
                    )


def build_inventory(raw_audit_path: Path, intact_interim_dir: Path, output_dir: Path) -> dict:
    audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    huri_path, huri_manifest, _ = _source_from_audit(audit, "huri_pairs")
    negatome_path, negatome_manifest, _ = _source_from_audit(audit, "negatome2_manual_stringent")
    negative_path, negative_manifest, _ = _source_from_audit(audit, "intact_human_negative")

    interim_metadata_path = intact_interim_dir / "metadata.json"
    candidate_path = intact_interim_dir / "intact_direct_candidates.tsv"
    interim_metadata = json.loads(interim_metadata_path.read_text(encoding="utf-8"))
    expected_candidate = interim_metadata.get("outputs", {}).get("intact_direct_candidates.tsv", {})
    if not candidate_path.is_file():
        raise ValueError("IntAct direct-candidate table is missing")
    if candidate_path.stat().st_size != expected_candidate.get("bytes"):
        raise ValueError("IntAct direct-candidate byte count does not match metadata")
    if sha256_file(candidate_path) != expected_candidate.get("sha256"):
        raise ValueError("IntAct direct-candidate SHA-256 does not match metadata")

    counters: defaultdict[InventoryKey, int] = defaultdict(int)
    streams = (
        iter_huri(huri_path, huri_manifest["source_version"]),
        iter_negatome(negatome_path, negatome_manifest["source_version"]),
        iter_intact_candidates(candidate_path, str(interim_metadata["source_version"])),
        iter_intact_negatives(negative_path, negative_manifest["source_version"]),
    )
    for stream in streams:
        for item in stream:
            counters[item] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "identifier_inventory.tsv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for key in sorted(counters, key=lambda item: tuple(item.__dict__.values())):
            writer.writerow({**key.__dict__, "occurrence_count": counters[key]})

    request_ids: defaultdict[str, set[str]] = defaultdict(set)
    status_counts: Counter[str] = Counter()
    source_unique_counts: Counter[str] = Counter()
    for key in counters:
        status_counts[key.mapping_status] += 1
        source_unique_counts[key.source_database] += 1
        if key.mapping_status == "mapping_candidate":
            request_ids[key.namespace].add(key.mapping_query_id)

    request_files = {}
    for namespace, filename in (
        ("Ensembl", "ensembl_gene_ids.txt"),
        ("UniProtKB_AC-ID", "uniprotkb_accession_ids.txt"),
    ):
        path = output_dir / filename
        path.write_text("\n".join(sorted(request_ids[namespace])) + "\n", encoding="utf-8")
        request_files[namespace] = {
            "path": str(path.resolve()),
            "count": len(request_ids[namespace]),
            "sha256": sha256_file(path),
        }

    metadata = {
        "schema_version": "1.0",
        "method": "build_identifier_inventory_v0.1",
        "raw_audit": str(raw_audit_path.resolve()),
        "raw_audit_sha256": sha256_file(raw_audit_path),
        "intact_interim_metadata": str(interim_metadata_path.resolve()),
        "intact_interim_metadata_sha256": sha256_file(interim_metadata_path),
        "inventory": {
            "path": str(inventory_path.resolve()),
            "rows": len(counters),
            "sha256": sha256_file(inventory_path),
        },
        "unique_rows_by_source": dict(sorted(source_unique_counts.items())),
        "mapping_status_counts": dict(sorted(status_counts.items())),
        "request_files": request_files,
        "policy_warning": (
            "This inventory contains mapping requests, not resolved proteins or PPI labels. "
            "Isoforms and non-UniProt primary identifiers remain outside automatic acceptance."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--intact-interim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = build_inventory(args.raw_audit, args.intact_interim_dir, args.output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
