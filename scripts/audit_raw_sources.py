#!/usr/bin/env python3
"""Validate acquired raw sources and report source-specific data risks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.fetch_raw_sources import load_registry, validate_download


UNIPROT_PREFIX = "uniprotkb:"
ESSENTIAL_MI_TERMS = {
    "MI:0407": "direct interaction",
    "MI:0915": "physical association",
    "MI:0914": "association",
    "MI:0208": "genetic interaction (sensu unexpected)",
    "MI:0018": "two hybrid",
    "MI:0004": "affinity chromatography technology",
    "MI:0006": "anti bait coimmunoprecipitation",
    "MI:0114": "x-ray crystallography",
}


def source_path(raw_root: Path, source: dict[str, Any]) -> Path:
    return raw_root / source["source_database"] / source["source_version"] / source["filename"]


def manifest_path(raw_root: Path, source: dict[str, Any]) -> Path:
    return (
        raw_root
        / source["source_database"]
        / source["source_version"]
        / f"{source['source_id']}.manifest.json"
    )


def audit_integrity(source: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    path = source_path(raw_root, source)
    manifest_file = manifest_path(raw_root, source)
    errors: list[str] = []
    if not path.is_file():
        return {"status": "not_acquired", "errors": [], "path": str(path)}
    validation = validate_download(path, source)
    errors.extend(validation["errors"])
    manifest: dict[str, Any] | None = None
    if not manifest_file.is_file():
        errors.append("retrieval manifest is missing")
    else:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"retrieval manifest is invalid: {exc}")
        if manifest is not None:
            if manifest.get("source_id") != source["source_id"]:
                errors.append("manifest source_id mismatch")
            if manifest.get("registry_url") != source["url"]:
                errors.append("manifest URL mismatch")
            observed = manifest.get("observed", {})
            if observed.get("sha256") != validation["sha256"]:
                errors.append("manifest SHA-256 does not match local file")
            if observed.get("bytes") != validation["bytes"]:
                errors.append("manifest byte count does not match local file")
            if not str(manifest.get("retrieved_at", "")).strip():
                errors.append("manifest retrieval time is missing")
            if not str(manifest.get("license_status", "")).strip():
                errors.append("manifest license status is missing")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "path": str(path.resolve()),
        "manifest": str(manifest_file.resolve()),
        "observed": validation,
        "license_status": source["license_status"],
    }


def profile_huri(path: Path) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    invalid_lines: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) != 2 or not all(re.fullmatch(r"ENSG\d{11}", value) for value in row):
                invalid_lines.append(line_number)
                continue
            rows.append((row[0], row[1]))
    canonical = [tuple(sorted(pair)) for pair in rows]
    return {
        "format": "headerless_two_column_ensembl_gene_pairs",
        "rows": len(rows),
        "invalid_rows": len(invalid_lines),
        "first_invalid_lines": invalid_lines[:10],
        "self_pairs_to_quarantine": sum(left == right for left, right in rows),
        "unique_directed_pairs": len(set(rows)),
        "unique_undirected_pairs": len(set(canonical)),
        "unique_ensembl_genes": len({item for pair in rows for item in pair}),
        "required_next_step": "Map GENCODE-v27 Ensembl genes to an explicitly pinned UniProt canonical release; never guess one-to-one mappings.",
    }


def profile_negatome(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.reader(handle, delimiter="\t"))
    rows = [row for row in raw_rows if len(row) == 4]
    pairs = [(row[0].strip(), row[1].strip()) for row in rows]
    canonical = [tuple(sorted(pair)) for pair in pairs]
    method_ids = [row[3].strip().split(maxsplit=1)[0].rstrip("-") for row in rows]
    return {
        "format": "four_columns_accession_accession_publication_detection_method",
        "rows": len(raw_rows),
        "valid_four_column_rows": len(rows),
        "invalid_column_count_rows": len(raw_rows) - len(rows),
        "self_pairs_to_quarantine": sum(left == right for left, right in pairs),
        "isoform_specific_rows_to_quarantine": sum("-" in left or "-" in right for left, right in pairs),
        "unique_directed_pairs": len(set(pairs)),
        "unique_undirected_pairs": len(set(canonical)),
        "duplicate_or_reciprocal_rows": len(pairs) - len(set(canonical)),
        "unique_accession_strings": len({item for pair in pairs for item in pair}),
        "unique_publication_identifiers": len({row[2].strip() for row in rows}),
        "top_detection_methods": Counter(method_ids).most_common(15),
        "required_next_step": "Resolve taxonomy and current accessions through a pinned UniProt release; preserve isoforms and self-pairs in quarantine, and do not redistribute until license terms are confirmed.",
    }


def profile_intact_negative(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {
        "#ID(s) interactor A", "ID(s) interactor B", "Taxid interactor A",
        "Taxid interactor B", "Interaction detection method(s)", "Interaction type(s)",
        "Type(s) interactor A", "Type(s) interactor B", "Interaction identifier(s)", "Negative",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"IntAct MITAB is missing required fields: {', '.join(missing)}")
    both_human = [
        row for row in rows
        if "taxid:9606" in row["Taxid interactor A"]
        and "taxid:9606" in row["Taxid interactor B"]
    ]
    both_human_protein = [
        row for row in both_human
        if "MI:0326" in row["Type(s) interactor A"]
        and "MI:0326" in row["Type(s) interactor B"]
    ]
    both_primary_uniprot = [
        row for row in both_human_protein
        if row["#ID(s) interactor A"].startswith(UNIPROT_PREFIX)
        and row["ID(s) interactor B"].startswith(UNIPROT_PREFIX)
    ]
    canonical = {
        tuple(sorted((row["#ID(s) interactor A"], row["ID(s) interactor B"])))
        for row in both_human_protein
    }
    canonical_primary_uniprot = {
        tuple(sorted((row["#ID(s) interactor A"], row["ID(s) interactor B"])))
        for row in both_primary_uniprot
    }
    return {
        "format": "PSI_MITAB_2.7_negative_export",
        "columns": len(fields),
        "rows": len(rows),
        "negative_flag_counts": dict(sorted(Counter(row["Negative"] for row in rows).items())),
        "both_interactors_human_rows": len(both_human),
        "both_human_protein_rows": len(both_human_protein),
        "both_human_protein_primary_uniprot_rows": len(both_primary_uniprot),
        "unique_human_protein_endpoint_pairs": len(canonical),
        "unique_human_protein_primary_uniprot_pairs": len(canonical_primary_uniprot),
        "unique_interaction_identifiers": len(
            {row["Interaction identifier(s)"] for row in both_human_protein}
        ),
        "top_interaction_types_all_rows": Counter(
            row["Interaction type(s)"] for row in rows
        ).most_common(10),
        "top_detection_methods_all_rows": Counter(
            row["Interaction detection method(s)"] for row in rows
        ).most_common(10),
        "required_next_step": "Require taxid 9606 and protein type for both endpoints, resolve non-UniProt IDs explicitly, and retain this source as a distinct negative stratum.",
    }


def parse_obo_terms(path: Path) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            if current and "id" in current:
                terms[current["id"]] = current
            current = {"is_a": []}
            continue
        if line.startswith("["):
            if current and "id" in current:
                terms[current["id"]] = current
            current = None
            continue
        if current is None or not line or line.startswith("!"):
            continue
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("name: "):
            current["name"] = line[6:]
        elif line.startswith("is_a: "):
            current["is_a"].append(line[6:].split(" ! ", 1)[0])
        elif line.startswith("is_obsolete: "):
            current["is_obsolete"] = line[13:].lower() == "true"
        elif line.startswith("replaced_by: "):
            current["replaced_by"] = line[13:]
    if current and "id" in current:
        terms[current["id"]] = current
    return terms


def profile_psi_mi(path: Path) -> dict[str, Any]:
    terms = parse_obo_terms(path)
    essential = {
        term_id: {
            "expected_name": expected_name,
            "observed_name": terms.get(term_id, {}).get("name"),
            "parents": terms.get(term_id, {}).get("is_a", []),
            "present": term_id in terms,
        }
        for term_id, expected_name in ESSENTIAL_MI_TERMS.items()
    }
    mismatches = [
        term_id for term_id, item in essential.items()
        if not item["present"] or item["observed_name"] != item["expected_name"]
    ]
    return {
        "term_count": len(terms),
        "essential_terms": essential,
        "essential_term_mismatches": mismatches,
        "required_next_step": "Generate and freeze interaction-type and detection-method descendant sets separately before filtering IntAct.",
    }


def audit(registry_path: Path, raw_root: Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    integrity = {
        source["source_id"]: audit_integrity(source, raw_root)
        for source in registry["sources"]
    }
    failures = [source_id for source_id, result in integrity.items() if result["status"] == "fail"]
    acquired = [source_id for source_id, result in integrity.items() if result["status"] == "pass"]
    by_id = {source["source_id"]: source for source in registry["sources"]}
    profiles: dict[str, Any] = {}
    if integrity.get("huri_pairs", {}).get("status") == "pass":
        profiles["huri_pairs"] = profile_huri(source_path(raw_root, by_id["huri_pairs"]))
    if integrity.get("negatome2_manual_stringent", {}).get("status") == "pass":
        profiles["negatome2_manual_stringent"] = profile_negatome(
            source_path(raw_root, by_id["negatome2_manual_stringent"])
        )
    if integrity.get("intact_human_negative", {}).get("status") == "pass":
        profiles["intact_human_negative"] = profile_intact_negative(
            source_path(raw_root, by_id["intact_human_negative"])
        )
    if integrity.get("psi_mi_obo", {}).get("status") == "pass":
        profiles["psi_mi_obo"] = profile_psi_mi(source_path(raw_root, by_id["psi_mi_obo"]))
    return {
        "schema_version": "1.0",
        "registry_id": registry.get("registry_id"),
        "status": "pass" if not failures else "fail",
        "acquired_source_ids": acquired,
        "not_acquired_source_ids": [
            source_id for source_id, result in integrity.items()
            if result["status"] == "not_acquired"
        ],
        "failed_source_ids": failures,
        "integrity": integrity,
        "profiles": profiles,
        "interpretation_warning": "Acquisition and format integrity do not establish biological eligibility. Identifier mapping, taxonomy, ontology filtering, contradiction resolution, and leakage control remain mandatory.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Raw PPI source audit",
        "",
        f"- Registry: `{report['registry_id']}`",
        f"- Status: **{report['status'].upper()}**",
        f"- Acquired sources: {len(report['acquired_source_ids'])}",
        f"- Not acquired: {', '.join(report['not_acquired_source_ids']) or 'none'}",
        f"- Integrity failures: {', '.join(report['failed_source_ids']) or 'none'}",
        "",
        "This is an acquisition and format audit, not a frozen biological benchmark.",
        "",
        "## Observed profiles",
        "",
    ]
    for source_id, profile in report["profiles"].items():
        lines.extend([f"### {source_id}", "", "```json", json.dumps(profile, indent=2, ensure_ascii=False), "```", ""])
    lines.extend(["## Warning", "", report["interpretation_warning"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.registry, args.raw_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
