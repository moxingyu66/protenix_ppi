#!/usr/bin/env python3
"""Validate normalized Protenix-PPI benchmark tables and leakage invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


ACCESSION = re.compile(r"^[A-Z0-9]{6,10}$")
MI_TERM = re.compile(r"^MI:\d{4}$")
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
PARTITIONS = {"train", "validation", "test"}
EVIDENCE_CLASSES = {
    "direct_positive",
    "curated_negative",
    "screen_negative",
    "compartment_negative",
    "unlabeled",
}
EXPECTED_LABEL = {
    "direct_positive": "1",
    "curated_negative": "0",
    "screen_negative": "0",
    "compartment_negative": "0",
    "unlabeled": "",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str], report: ValidationReport) -> bool:
    if rows:
        columns = set(rows[0])
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = set(next(reader, []))
    missing = sorted(required - columns)
    if missing:
        report.errors.append(f"{path.name}: missing columns: {', '.join(missing)}")
        return False
    return True


def validate_tables(
    proteins_path: Path,
    pairs_path: Path,
    evidence_path: Path,
    splits_path: Path,
) -> ValidationReport:
    report = ValidationReport()
    proteins = read_csv(proteins_path)
    pairs = read_csv(pairs_path)
    evidence = read_csv(evidence_path)
    splits = read_csv(splits_path)
    report.counts = {
        "proteins": len(proteins),
        "pairs": len(pairs),
        "evidence": len(evidence),
        "split_assignments": len(splits),
    }

    if not require_columns(
        proteins_path,
        proteins,
        {
            "uniprot_accession", "taxid", "sequence", "sequence_length", "sequence_sha256",
            "uniprot_release", "is_reviewed", "homology_cluster_30", "retrieved_at",
        },
        report,
    ):
        return report
    if not require_columns(
        pairs_path,
        pairs,
        {"pair_id", "uniprot_a", "uniprot_b", "label", "evidence_class", "complex_group_id", "pair_status"},
        report,
    ):
        return report
    if not require_columns(
        evidence_path,
        evidence,
        {
            "evidence_id", "pair_id", "source_database", "source_version", "source_record_id",
            "interaction_type_mi", "detection_method_mi", "evidence_polarity", "negative_definition",
            "license", "retrieved_at",
        },
        report,
    ):
        return report
    if not require_columns(
        splits_path,
        splits,
        {"split_scheme", "fold", "pair_id", "partition", "pm_class"},
        report,
    ):
        return report

    protein_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(proteins, start=2):
        accession = row["uniprot_accession"].strip()
        if accession in protein_by_id:
            report.errors.append(f"proteins.csv:{line}: duplicate accession {accession}")
            continue
        protein_by_id[accession] = row
        if not ACCESSION.fullmatch(accession):
            report.errors.append(f"proteins.csv:{line}: invalid canonical accession {accession!r}")
        if row["taxid"].strip() != "9606":
            report.errors.append(f"proteins.csv:{line}: primary benchmark taxid must be 9606")
        sequence = row["sequence"].strip().upper()
        if not sequence or set(sequence) - AA_ALPHABET:
            report.errors.append(f"proteins.csv:{line}: invalid amino-acid sequence for {accession}")
        try:
            declared_length = int(row["sequence_length"])
        except ValueError:
            declared_length = -1
        if declared_length != len(sequence):
            report.errors.append(f"proteins.csv:{line}: sequence length mismatch for {accession}")
        actual_hash = hashlib.sha256(sequence.encode("ascii", errors="ignore")).hexdigest()
        if row["sequence_sha256"].strip().lower() != actual_hash:
            report.errors.append(f"proteins.csv:{line}: sequence SHA-256 mismatch for {accession}")
        if set(sequence) & set("BXZJUO"):
            report.warnings.append(f"proteins.csv:{line}: ambiguous/nonstandard residue in {accession}")

    pair_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(pairs, start=2):
        pair_id = row["pair_id"].strip()
        a = row["uniprot_a"].strip()
        b = row["uniprot_b"].strip()
        expected_id = f"HUMAN_{a}_{b}"
        if pair_id in pair_by_id:
            report.errors.append(f"pairs.csv:{line}: duplicate pair_id {pair_id}")
            continue
        pair_by_id[pair_id] = row
        status = row["pair_status"].strip()
        if a > b:
            report.errors.append(f"pairs.csv:{line}: endpoints must be lexicographically ordered")
        if a == b and status != "quarantine":
            report.errors.append(
                f"pairs.csv:{line}: self-pairs must remain quarantined outside the primary heterotypic task"
            )
        if pair_id != expected_id:
            report.errors.append(f"pairs.csv:{line}: expected pair_id {expected_id}")
        for endpoint in (a, b):
            if endpoint not in protein_by_id:
                report.errors.append(f"pairs.csv:{line}: unknown protein {endpoint}")
        evidence_class = row["evidence_class"].strip()
        if evidence_class not in EVIDENCE_CLASSES:
            report.errors.append(f"pairs.csv:{line}: invalid evidence_class {evidence_class!r}")
        elif row["label"].strip() != EXPECTED_LABEL[evidence_class]:
            report.errors.append(
                f"pairs.csv:{line}: label {row['label']!r} conflicts with evidence_class {evidence_class}"
            )
        if status not in {"eligible", "quarantine", "excluded"}:
            report.errors.append(f"pairs.csv:{line}: invalid pair_status {status!r}")

    evidence_ids: set[str] = set()
    evidence_by_pair: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    source_records: dict[tuple[str, str, str], tuple[str, str]] = {}
    for line, row in enumerate(evidence, start=2):
        evidence_id = row["evidence_id"].strip()
        pair_id = row["pair_id"].strip()
        if not evidence_id or evidence_id in evidence_ids:
            report.errors.append(f"evidence.csv:{line}: missing or duplicate evidence_id {evidence_id!r}")
        evidence_ids.add(evidence_id)
        if pair_id not in pair_by_id:
            report.errors.append(f"evidence.csv:{line}: unknown pair_id {pair_id}")
        evidence_by_pair[pair_id].append(row)
        for field_name in ("interaction_type_mi", "detection_method_mi"):
            value = row[field_name].strip()
            if value and not MI_TERM.fullmatch(value):
                report.errors.append(f"evidence.csv:{line}: invalid {field_name} {value!r}")
        polarity = row["evidence_polarity"].strip()
        if polarity not in {"positive", "negative", "context_only"}:
            report.errors.append(f"evidence.csv:{line}: invalid evidence_polarity {polarity!r}")
        source_key = (
            row["source_database"].strip(),
            row["source_version"].strip(),
            row["source_record_id"].strip(),
        )
        prior = source_records.get(source_key)
        current = (pair_id, polarity)
        if prior is not None and prior != current:
            report.errors.append(f"evidence.csv:{line}: source record {source_key} has contradictory mappings")
        source_records[source_key] = current

    for pair_id, row in pair_by_id.items():
        if row["pair_status"].strip() == "eligible" and not evidence_by_pair.get(pair_id):
            report.errors.append(f"pairs.csv: eligible pair lacks evidence: {pair_id}")
        if row["evidence_class"].strip() == "direct_positive":
            positive_rows = [item for item in evidence_by_pair.get(pair_id, []) if item["evidence_polarity"].strip() == "positive"]
            if not positive_rows:
                report.errors.append(f"pairs.csv: direct_positive lacks positive evidence: {pair_id}")
            if positive_rows and not any(item["interaction_type_mi"].strip() == "MI:0407" for item in positive_rows):
                report.warnings.append(
                    f"pairs.csv: direct_positive has no MI:0407 interaction type yet; ontology descendant resolution required: {pair_id}"
                )
        if row["evidence_class"].strip().endswith("negative"):
            negative_rows = [item for item in evidence_by_pair.get(pair_id, []) if item["evidence_polarity"].strip() == "negative"]
            if not negative_rows or not all(item["negative_definition"].strip() for item in negative_rows):
                report.errors.append(f"pairs.csv: labeled negative lacks explicit negative evidence definition: {pair_id}")

    assignments: dict[tuple[str, str, str], str] = {}
    nodes_by_partition: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    clusters_by_partition: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    complex_partition: dict[tuple[str, str, str], str] = {}

    for line, row in enumerate(splits, start=2):
        scheme = row["split_scheme"].strip()
        fold = row["fold"].strip()
        pair_id = row["pair_id"].strip()
        partition = row["partition"].strip()
        if pair_id not in pair_by_id:
            report.errors.append(f"splits.csv:{line}: unknown pair_id {pair_id}")
            continue
        if partition not in PARTITIONS:
            report.errors.append(f"splits.csv:{line}: invalid partition {partition!r}")
            continue
        assignment_key = (scheme, fold, pair_id)
        if assignment_key in assignments:
            report.errors.append(f"splits.csv:{line}: duplicate assignment for {assignment_key}")
        assignments[assignment_key] = partition
        pair = pair_by_id[pair_id]
        for endpoint in (pair["uniprot_a"].strip(), pair["uniprot_b"].strip()):
            nodes_by_partition[(scheme, fold, partition)].add(endpoint)
            cluster = protein_by_id.get(endpoint, {}).get("homology_cluster_30", "").strip()
            if cluster:
                clusters_by_partition[(scheme, fold, partition)].add(cluster)
        group_id = pair["complex_group_id"].strip()
        if group_id:
            group_key = (scheme, fold, group_id)
            prior_partition = complex_partition.get(group_key)
            if prior_partition is not None and prior_partition != partition:
                report.errors.append(f"splits.csv:{line}: complex group {group_id} crosses partitions")
            complex_partition[group_key] = partition

    schemes_and_folds = {(row["split_scheme"].strip(), row["fold"].strip()) for row in splits}
    for scheme, fold in schemes_and_folds:
        if scheme == "c3_primary":
            _check_disjoint_sets(nodes_by_partition, scheme, fold, "protein", report)
            missing_clusters = [
                protein_id
                for protein_id, protein in protein_by_id.items()
                if not protein["homology_cluster_30"].strip()
            ]
            if missing_clusters:
                report.errors.append("c3_primary requires homology_cluster_30 for every protein")
            _check_disjoint_sets(clusters_by_partition, scheme, fold, "homology cluster", report)
            for row in splits:
                if row["split_scheme"].strip() == scheme and row["fold"].strip() == fold and row["partition"].strip() == "test":
                    if row["pm_class"].strip() != "C3":
                        report.errors.append(f"splits.csv: c3_primary test pair must be marked C3: {row['pair_id']}")
        if scheme == "homology_primary":
            missing_clusters = [
                protein_id for protein_id, protein in protein_by_id.items() if not protein["homology_cluster_30"].strip()
            ]
            if missing_clusters:
                report.errors.append("homology_primary requires homology_cluster_30 for every protein")
            _check_disjoint_sets(clusters_by_partition, scheme, fold, "homology cluster", report)

    return report


def _check_disjoint_sets(
    values: defaultdict[tuple[str, str, str], set[str]],
    scheme: str,
    fold: str,
    label: str,
    report: ValidationReport,
) -> None:
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = values[(scheme, fold, left)] & values[(scheme, fold, right)]
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            report.errors.append(
                f"{scheme} fold {fold}: {label} leakage between {left} and {right}: {preview}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = validate_tables(args.proteins, args.pairs, args.evidence, args.splits)
    rendered = json.dumps({**asdict(report), "passed": report.passed}, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
