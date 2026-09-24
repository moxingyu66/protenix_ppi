#!/usr/bin/env python3
"""Build provisional normalized PPI tables from pinned, resolved evidence.

The output remains pre-clustering and pre-split.  It preserves source evidence,
quarantines unresolved/self/contradictory cases, and never manufactures
negatives from unobserved protein pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from protenix_ppi.scripts.build_identifier_inventory import (
    _source_from_audit,
    _split_database_identifier,
    sha256_file,
)


MI_RE = re.compile(r"MI:\d{4}")
PDB_RE = re.compile(r"rcsb pdb:([0-9A-Za-z]{4})", re.IGNORECASE)
PAIR_FIELDS = [
    "pair_id", "uniprot_a", "uniprot_b", "label", "evidence_class",
    "complex_group_id", "pair_status", "exclusion_reason",
]
EVIDENCE_FIELDS = [
    "evidence_id", "pair_id", "source_database", "source_version", "source_record_id",
    "publication_id", "interaction_type_mi", "detection_method_mi", "evidence_polarity",
    "negative_definition", "license", "retrieved_at",
]
QUARANTINE_FIELDS = [
    "source_database", "source_version", "evidence_role", "source_record_id",
    "raw_identifier_a", "raw_identifier_b", "endpoint_a_status", "endpoint_b_status",
    "reason",
]


def _one_mi(raw: str) -> str:
    terms = sorted(set(MI_RE.findall(raw or "")))
    return terms[0] if len(terms) == 1 else ""


def _publication_ids(raw: str) -> str:
    values = sorted(set(re.findall(r"pubmed:(\d+)", raw or "", flags=re.IGNORECASE)))
    return ";".join(f"PMID:{value}" for value in values)


def _evidence_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"EVI_{digest}"


def _pair_id(a: str, b: str) -> tuple[str, str, str]:
    left, right = sorted((a, b))
    return f"HUMAN_{left}_{right}", left, right


def load_resolution(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    lookup = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (
                row["source_database"].strip(),
                row["source_version"].strip(),
                row["evidence_role"].strip(),
                row["namespace"].strip(),
                row["raw_identifier"].strip(),
            )
            if key in lookup:
                raise ValueError(f"duplicate identifier-resolution key: {key}")
            lookup[key] = row
    return lookup


def resolve_endpoint(
    lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    source_database: str,
    source_version: str,
    evidence_role: str,
    namespace: str,
    raw_identifier: str,
) -> tuple[str, str]:
    row = lookup.get((source_database, source_version, evidence_role, namespace, raw_identifier))
    if row is None:
        return "", "identifier_not_in_resolution_table"
    if row["resolution_status"] != "accepted":
        return "", row["resolution_reason"]
    return row["resolved_accession"], "accepted"


def add_observation(
    observations: list[dict],
    quarantine: list[dict[str, str]],
    evidence: dict[str, str],
    raw_a: str,
    raw_b: str,
    resolved_a: str,
    resolved_b: str,
    reason_a: str,
    reason_b: str,
    negative_class: str = "",
    complex_ids: Iterable[str] = (),
) -> None:
    if not resolved_a or not resolved_b:
        quarantine.append(
            {
                "source_database": evidence["source_database"],
                "source_version": evidence["source_version"],
                "evidence_role": "positive_evidence" if evidence["evidence_polarity"] == "positive" else "negative_evidence",
                "source_record_id": evidence["source_record_id"],
                "raw_identifier_a": raw_a,
                "raw_identifier_b": raw_b,
                "endpoint_a_status": reason_a,
                "endpoint_b_status": reason_b,
                "reason": "unresolved_endpoint",
            }
        )
        return
    pair_id, left, right = _pair_id(resolved_a, resolved_b)
    evidence = dict(evidence)
    evidence["pair_id"] = pair_id
    evidence["evidence_id"] = _evidence_id(
        evidence["source_database"], evidence["source_version"], evidence["source_record_id"],
        pair_id, evidence["evidence_polarity"], evidence["detection_method_mi"],
    )
    observations.append(
        {
            "pair_id": pair_id,
            "uniprot_a": left,
            "uniprot_b": right,
            "evidence": evidence,
            "negative_class": negative_class,
            "complex_ids": set(complex_ids),
            "raw_identifier_a": raw_a,
            "raw_identifier_b": raw_b,
        }
    )


def quarantine_ambiguous_source_records(
    observations: list[dict], quarantine: list[dict[str, str]]
) -> list[dict]:
    pairs_by_source_record: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    for item in observations:
        evidence = item["evidence"]
        key = (
            evidence["source_database"],
            evidence["source_version"],
            evidence["source_record_id"],
        )
        pairs_by_source_record[key].add(item["pair_id"])
    ambiguous = {key for key, pair_ids in pairs_by_source_record.items() if len(pair_ids) > 1}
    kept = []
    for item in observations:
        evidence = item["evidence"]
        key = (
            evidence["source_database"],
            evidence["source_version"],
            evidence["source_record_id"],
        )
        if key not in ambiguous:
            kept.append(item)
            continue
        quarantine.append(
            {
                "source_database": evidence["source_database"],
                "source_version": evidence["source_version"],
                "evidence_role": (
                    "positive_evidence"
                    if evidence["evidence_polarity"] == "positive"
                    else "negative_evidence"
                ),
                "source_record_id": evidence["source_record_id"],
                "raw_identifier_a": item["raw_identifier_a"],
                "raw_identifier_b": item["raw_identifier_b"],
                "endpoint_a_status": "accepted",
                "endpoint_b_status": "accepted",
                "reason": "source_record_maps_to_multiple_normalized_pairs",
            }
        )
    return kept


class DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def finalize_observations(observations: list[dict]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for observation in observations:
        grouped[observation["pair_id"]].append(observation)

    dsu = DisjointSet(grouped)
    pairs_by_complex: defaultdict[str, set[str]] = defaultdict(set)
    for pair_id, items in grouped.items():
        for item in items:
            for complex_id in item["complex_ids"]:
                pairs_by_complex[complex_id].add(pair_id)
    for pair_ids in pairs_by_complex.values():
        ordered = sorted(pair_ids)
        for pair_id in ordered[1:]:
            dsu.union(ordered[0], pair_id)
    complexes_by_component: defaultdict[str, set[str]] = defaultdict(set)
    for complex_id, pair_ids in pairs_by_complex.items():
        for pair_id in pair_ids:
            complexes_by_component[dsu.find(pair_id)].add(complex_id)

    pair_rows = []
    evidence_by_id: dict[str, dict[str, str]] = {}
    for pair_id in sorted(grouped):
        items = grouped[pair_id]
        polarities = {item["evidence"]["evidence_polarity"] for item in items}
        a, b = items[0]["uniprot_a"], items[0]["uniprot_b"]
        if polarities == {"positive"}:
            label, evidence_class, status, reason = "1", "direct_positive", "eligible", ""
        elif polarities == {"negative"}:
            negative_classes = {item["negative_class"] for item in items}
            evidence_class = "curated_negative" if "curated_negative" in negative_classes else "screen_negative"
            label, status, reason = "0", "eligible", ""
        else:
            label, evidence_class, status, reason = "", "unlabeled", "quarantine", "positive_negative_contradiction"
        if a == b:
            status = "quarantine"
            reason = "self_pair_outside_primary_heterotypic_scope"
        complex_ids = complexes_by_component.get(dsu.find(pair_id), set())
        if len(complex_ids) == 1:
            complex_group = "PDB_" + next(iter(complex_ids))
        elif complex_ids:
            digest = hashlib.sha256(";".join(sorted(complex_ids)).encode("ascii")).hexdigest()[:12]
            complex_group = f"PDB_COMPONENT_{digest}"
        else:
            complex_group = ""
        pair_rows.append(
            {
                "pair_id": pair_id,
                "uniprot_a": a,
                "uniprot_b": b,
                "label": label,
                "evidence_class": evidence_class,
                "complex_group_id": complex_group,
                "pair_status": status,
                "exclusion_reason": reason,
            }
        )
        for item in items:
            evidence = item["evidence"]
            evidence_id = evidence["evidence_id"]
            prior = evidence_by_id.get(evidence_id)
            if prior is not None and prior != evidence:
                raise ValueError(f"evidence ID collision: {evidence_id}")
            evidence_by_id[evidence_id] = evidence
    return pair_rows, [evidence_by_id[key] for key in sorted(evidence_by_id)]


def build_benchmark(
    raw_audit_path: Path,
    intact_interim_dir: Path,
    resolution_path: Path,
    resolution_metadata_path: Path,
    proteins_precluster_path: Path,
    output_dir: Path,
    luck2020_candidates_path: Path | None = None,
    luck2020_metadata_path: Path | None = None,
    luck2020_resolution_path: Path | None = None,
    luck2020_resolution_metadata_path: Path | None = None,
    luck2020_proteins_precluster_path: Path | None = None,
    luck2020_source_quarantine_path: Path | None = None,
) -> dict:
    audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    huri_path, huri_manifest, _ = _source_from_audit(audit, "huri_pairs")
    negatome_path, negatome_manifest, _ = _source_from_audit(audit, "negatome2_manual_stringent")
    intact_negative_path, intact_negative_manifest, _ = _source_from_audit(audit, "intact_human_negative")
    intact_metadata_path = intact_interim_dir / "metadata.json"
    intact_metadata = json.loads(intact_metadata_path.read_text(encoding="utf-8"))
    intact_candidates_path = intact_interim_dir / "intact_direct_candidates.tsv"
    expected_candidates = intact_metadata["outputs"]["intact_direct_candidates.tsv"]
    if sha256_file(intact_candidates_path) != expected_candidates["sha256"]:
        raise ValueError("IntAct direct candidates do not match metadata")
    resolution_metadata = json.loads(resolution_metadata_path.read_text(encoding="utf-8"))
    if sha256_file(resolution_path) != resolution_metadata["outputs"]["identifier_resolution.tsv"]["sha256"]:
        raise ValueError("identifier resolution does not match metadata")
    if sha256_file(proteins_precluster_path) != resolution_metadata["outputs"]["proteins_precluster.csv"]["sha256"]:
        raise ValueError("precluster proteins do not match resolution metadata")
    resolution = load_resolution(resolution_path)
    luck_paths = (
        luck2020_candidates_path,
        luck2020_metadata_path,
        luck2020_resolution_path,
        luck2020_resolution_metadata_path,
        luck2020_proteins_precluster_path,
        luck2020_source_quarantine_path,
    )
    if any(path is not None for path in luck_paths) and not all(path is not None for path in luck_paths):
        raise ValueError("all Luck2020 supplemental paths must be supplied together")
    luck_metadata = None
    luck_resolution_metadata = None
    if all(path is not None for path in luck_paths):
        assert luck2020_metadata_path is not None
        assert luck2020_candidates_path is not None
        assert luck2020_resolution_path is not None
        assert luck2020_resolution_metadata_path is not None
        assert luck2020_proteins_precluster_path is not None
        luck_metadata = json.loads(luck2020_metadata_path.read_text(encoding="utf-8"))
        if sha256_file(luck2020_candidates_path) != luck_metadata["outputs"]["screen_negative_candidates.tsv"]["sha256"]:
            raise ValueError("Luck2020 candidate table does not match metadata")
        luck_resolution_metadata = json.loads(
            luck2020_resolution_metadata_path.read_text(encoding="utf-8")
        )
        if sha256_file(luck2020_resolution_path) != luck_resolution_metadata["outputs"]["identifier_resolution.tsv"]["sha256"]:
            raise ValueError("Luck2020 resolution does not match metadata")
        if sha256_file(luck2020_proteins_precluster_path) != luck_resolution_metadata["outputs"]["proteins_precluster.csv"]["sha256"]:
            raise ValueError("Luck2020 protein table does not match resolution metadata")
        for key, row in load_resolution(luck2020_resolution_path).items():
            if key in resolution and resolution[key] != row:
                raise ValueError(f"conflicting supplemental resolution key: {key}")
            resolution[key] = row

    observations: list[dict] = []
    quarantine: list[dict[str, str]] = []
    retrieved_at = "2026-09-15"

    huri_version = huri_manifest["source_version"]
    with huri_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            raw_a, raw_b = row
            a, reason_a = resolve_endpoint(resolution, "HuRI", huri_version, "positive_evidence", "Ensembl", raw_a)
            b, reason_b = resolve_endpoint(resolution, "HuRI", huri_version, "positive_evidence", "Ensembl", raw_b)
            source_record = f"HuRI:{raw_a}:{raw_b}"
            evidence = {
                "evidence_id": "", "pair_id": "", "source_database": "HuRI",
                "source_version": huri_version, "source_record_id": source_record,
                "publication_id": "PMID:32296183", "interaction_type_mi": "MI:0407",
                "detection_method_mi": "MI:0018", "evidence_polarity": "positive",
                "negative_definition": "", "license": "CC_BY_4.0", "retrieved_at": retrieved_at,
            }
            add_observation(observations, quarantine, evidence, raw_a, raw_b, a, b, reason_a, reason_b)

    intact_version = str(intact_metadata["source_version"])
    with intact_candidates_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            _, raw_a = _split_database_identifier(row["primary_id_a"])
            _, raw_b = _split_database_identifier(row["primary_id_b"])
            a, reason_a = resolve_endpoint(resolution, "IntAct", intact_version, "positive_evidence", "UniProtKB_AC-ID", raw_a)
            b, reason_b = resolve_endpoint(resolution, "IntAct", intact_version, "positive_evidence", "UniProtKB_AC-ID", raw_b)
            source_record = row["source_record_id"].strip()
            evidence = {
                "evidence_id": "", "pair_id": "", "source_database": "IntAct",
                "source_version": intact_version, "source_record_id": source_record,
                "publication_id": _publication_ids(row["publication_ids"]),
                "interaction_type_mi": row["interaction_type_mi"].strip(),
                "detection_method_mi": row["detection_method_mi"].strip(),
                "evidence_polarity": "positive", "negative_definition": "",
                "license": "EMBL_EBI_open_data_terms", "retrieved_at": retrieved_at,
            }
            complexes = {match.upper() for match in PDB_RE.findall(source_record)}
            add_observation(observations, quarantine, evidence, raw_a, raw_b, a, b, reason_a, reason_b, complex_ids=complexes)

    intact_negative_version = intact_negative_manifest["source_version"]
    with intact_negative_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=2):
            if row["Negative"].strip().lower() != "true":
                continue
            if "taxid:9606" not in row["Taxid interactor A"] or "taxid:9606" not in row["Taxid interactor B"]:
                continue
            if "MI:0326" not in row["Type(s) interactor A"] or "MI:0326" not in row["Type(s) interactor B"]:
                continue
            namespace_a, raw_a = _split_database_identifier(row["#ID(s) interactor A"])
            namespace_b, raw_b = _split_database_identifier(row["ID(s) interactor B"])
            expansion = row["Expansion method(s)"].strip()
            if expansion not in {"", "-"}:
                quarantine.append(
                    {
                        "source_database": "IntAct",
                        "source_version": intact_negative_version,
                        "evidence_role": "negative_evidence",
                        "source_record_id": row["Interaction identifier(s)"].strip(),
                        "raw_identifier_a": raw_a,
                        "raw_identifier_b": raw_b,
                        "endpoint_a_status": "not_evaluated",
                        "endpoint_b_status": "not_evaluated",
                        "reason": "participant_expansion_requires_review",
                    }
                )
                continue
            norm_namespace_a = "UniProtKB_AC-ID" if namespace_a == "uniprotkb" else namespace_a
            norm_namespace_b = "UniProtKB_AC-ID" if namespace_b == "uniprotkb" else namespace_b
            a, reason_a = resolve_endpoint(resolution, "IntAct", intact_negative_version, "negative_evidence", norm_namespace_a, raw_a)
            b, reason_b = resolve_endpoint(resolution, "IntAct", intact_negative_version, "negative_evidence", norm_namespace_b, raw_b)
            detection_method = _one_mi(row["Interaction detection method(s)"])
            source_record = row["Interaction identifier(s)"].strip()
            if not source_record or source_record == "-":
                source_record = f"IntAct-negative-line:{line_number}"
            negative_class = "screen_negative" if detection_method == "MI:0397" else "curated_negative"
            evidence = {
                "evidence_id": "", "pair_id": "", "source_database": "IntAct",
                "source_version": intact_negative_version, "source_record_id": source_record,
                "publication_id": _publication_ids(row["Publication Identifier(s)"]),
                "interaction_type_mi": _one_mi(row["Interaction type(s)"]),
                "detection_method_mi": detection_method, "evidence_polarity": "negative",
                "negative_definition": (
                    "explicit_screen_negative_in_intact_export"
                    if negative_class == "screen_negative"
                    else "explicit_experimental_negative_in_intact_export"
                ),
                "license": "EMBL_EBI_open_data_terms", "retrieved_at": retrieved_at,
            }
            add_observation(
                observations, quarantine, evidence, raw_a, raw_b, a, b, reason_a, reason_b,
                negative_class=negative_class,
            )

    negatome_version = negatome_manifest["source_version"]
    with negatome_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            raw_a, raw_b, publication, method = row
            a, reason_a = resolve_endpoint(resolution, "Negatome 2.0", negatome_version, "negative_evidence", "UniProtKB_AC-ID", raw_a)
            b, reason_b = resolve_endpoint(resolution, "Negatome 2.0", negatome_version, "negative_evidence", "UniProtKB_AC-ID", raw_b)
            source_record = f"Negatome2-manual-stringent-line:{line_number}"
            evidence = {
                "evidence_id": "", "pair_id": "", "source_database": "Negatome 2.0",
                "source_version": negatome_version, "source_record_id": source_record,
                "publication_id": f"PMID:{publication}" if publication.isdigit() else publication,
                "interaction_type_mi": "", "detection_method_mi": _one_mi(method),
                "evidence_polarity": "negative",
                "negative_definition": "manually_curated_stringent_noninteraction",
                "license": "terms_not_stated_internal_use_no_redistribution",
                "retrieved_at": retrieved_at,
            }
            add_observation(
                observations, quarantine, evidence, raw_a, raw_b, a, b, reason_a, reason_b,
                negative_class="curated_negative",
            )

    if luck_metadata is not None:
        assert luck2020_candidates_path is not None
        assert luck2020_source_quarantine_path is not None
        luck_version = str(luck_metadata["source_version"])
        with luck2020_candidates_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw_a = row["ensembl_gene_a"].strip()
                raw_b = row["ensembl_gene_b"].strip()
                a, reason_a = resolve_endpoint(
                    resolution, "Luck2020", luck_version, "negative_evidence", "Ensembl", raw_a
                )
                b, reason_b = resolve_endpoint(
                    resolution, "Luck2020", luck_version, "negative_evidence", "Ensembl", raw_b
                )
                evidence = {
                    "evidence_id": "", "pair_id": "", "source_database": "Luck2020",
                    "source_version": luck_version, "source_record_id": row["source_record_id"].strip(),
                    "publication_id": row["publication_id"].strip(), "interaction_type_mi": "",
                    "detection_method_mi": row["detection_method_mi"].strip(),
                    "evidence_polarity": "negative",
                    "negative_definition": row["negative_definition"].strip(),
                    "license": "publisher_supplement_review_required_underlying_HuRI_CC_BY_4.0",
                    "retrieved_at": retrieved_at,
                }
                add_observation(
                    observations, quarantine, evidence, raw_a, raw_b, a, b, reason_a, reason_b,
                    negative_class="screen_negative",
                )
        with luck2020_source_quarantine_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                quarantine.append(
                    {
                        "source_database": "Luck2020",
                        "source_version": luck_version,
                        "evidence_role": "negative_evidence",
                        "source_record_id": row["source_record_id"].strip(),
                        "raw_identifier_a": row["orf_id_a"].strip(),
                        "raw_identifier_b": row["orf_id_b"].strip(),
                        "endpoint_a_status": f"orf_mapping_count={row['mapping_count_a'].strip()}",
                        "endpoint_b_status": f"orf_mapping_count={row['mapping_count_b'].strip()}",
                        "reason": row["reason"].strip(),
                    }
                )

    observations = quarantine_ambiguous_source_records(observations, quarantine)
    pair_rows, evidence_rows = finalize_observations(observations)
    used_proteins = {endpoint for row in pair_rows for endpoint in (row["uniprot_a"], row["uniprot_b"])}
    protein_by_accession: dict[str, dict[str, str]] = {}
    protein_paths = [proteins_precluster_path]
    if luck2020_proteins_precluster_path is not None:
        protein_paths.append(luck2020_proteins_precluster_path)
    for protein_path in protein_paths:
        with protein_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                accession = row["uniprot_accession"].strip()
                prior = protein_by_accession.get(accession)
                if prior is not None and prior != row:
                    raise ValueError(f"conflicting pinned protein records for {accession}")
                protein_by_accession[accession] = row
    protein_rows = [protein_by_accession[key] for key in sorted(used_proteins) if key in protein_by_accession]
    if {row["uniprot_accession"] for row in protein_rows} != used_proteins:
        raise ValueError("normalized pairs reference a protein absent from the pinned sequence table")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("proteins.csv", list(protein_rows[0]) if protein_rows else [], protein_rows, ","),
        ("pairs.csv", PAIR_FIELDS, pair_rows, ","),
        ("evidence.csv", EVIDENCE_FIELDS, evidence_rows, ","),
        ("source_quarantine.tsv", QUARANTINE_FIELDS, quarantine, "\t"),
    )
    output_meta = {}
    for filename, fields, rows, delimiter in outputs:
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        output_meta[filename] = {"rows": len(rows), "sha256": sha256_file(path)}
    splits_path = output_dir / "splits.csv"
    splits_path.write_text("split_scheme,fold,pair_id,partition,pm_class\n", encoding="utf-8")
    output_meta["splits.csv"] = {"rows": 0, "sha256": sha256_file(splits_path)}

    pair_status = Counter(row["pair_status"] for row in pair_rows)
    pair_classes = Counter(row["evidence_class"] for row in pair_rows)
    metadata = {
        "schema_version": "1.0",
        "method": (
            "build_normalized_benchmark_v0.3"
            if luck_metadata is not None
            else "build_normalized_benchmark_v0.2"
        ),
        "status": "provisional_precluster_presplit",
        "inputs": {
            "raw_audit_sha256": sha256_file(raw_audit_path),
            "intact_metadata_sha256": sha256_file(intact_metadata_path),
            "identifier_resolution_sha256": sha256_file(resolution_path),
            "identifier_resolution_metadata_sha256": sha256_file(resolution_metadata_path),
            "proteins_precluster_sha256": sha256_file(proteins_precluster_path),
            "luck2020_candidates_sha256": (
                sha256_file(luck2020_candidates_path) if luck2020_candidates_path is not None else None
            ),
            "luck2020_metadata_sha256": (
                sha256_file(luck2020_metadata_path) if luck2020_metadata_path is not None else None
            ),
            "luck2020_resolution_sha256": (
                sha256_file(luck2020_resolution_path) if luck2020_resolution_path is not None else None
            ),
            "luck2020_resolution_metadata_sha256": (
                sha256_file(luck2020_resolution_metadata_path)
                if luck2020_resolution_metadata_path is not None
                else None
            ),
        },
        "counts": {
            "proteins": len(protein_rows),
            "pairs": len(pair_rows),
            "evidence": len(evidence_rows),
            "source_rows_quarantined_before_pairing": len(quarantine),
            "pair_status": dict(sorted(pair_status.items())),
            "pair_evidence_class": dict(sorted(pair_classes.items())),
        },
        "outputs": output_meta,
        "warnings": [
            "No unobserved pair was labeled negative.",
            "Negatome terms are not stated on the official download page; do not redistribute its rows.",
            "This dataset is not frozen until contradiction review, MMseqs2 clustering, and C3 split validation pass.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--intact-interim-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--resolution-metadata", type=Path, required=True)
    parser.add_argument("--proteins-precluster", type=Path, required=True)
    parser.add_argument("--luck2020-candidates", type=Path)
    parser.add_argument("--luck2020-metadata", type=Path)
    parser.add_argument("--luck2020-resolution", type=Path)
    parser.add_argument("--luck2020-resolution-metadata", type=Path)
    parser.add_argument("--luck2020-proteins-precluster", type=Path)
    parser.add_argument("--luck2020-source-quarantine", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = build_benchmark(
        args.raw_audit, args.intact_interim_dir, args.resolution,
        args.resolution_metadata, args.proteins_precluster, args.output_dir,
        args.luck2020_candidates,
        args.luck2020_metadata,
        args.luck2020_resolution,
        args.luck2020_resolution_metadata,
        args.luck2020_proteins_precluster,
        args.luck2020_source_quarantine,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
