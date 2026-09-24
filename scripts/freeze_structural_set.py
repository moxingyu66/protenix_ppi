#!/usr/bin/env python3
"""Freeze a manually reviewed subset of RCSB structural candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.fetch_structural_candidates import CANDIDATE_FIELDS
from protenix_ppi.scripts.evaluate_structure_preservation import STRUCTURAL_SET_SELECTION_RULE


SCHEMA_VERSION = "1.0"
MIN_COMPLEXES = 30
REVIEW_FIELDS = {
    "complex_id",
    "include",
    "biological_assembly_confirmed",
    "chain_mapping_confirmed",
    "human_heteromer_confirmed",
    "b4_b5_outputs_consulted",
    "reviewer_role",
    "reviewed_at",
    "rationale",
}
OUTPUT_FIELDS = [
    "complex_id",
    "pdb_id",
    "biological_assembly_id",
    "protein_chain_mapping_json",
    "reference_structure",
    "reference_structure_sha256",
    "source_url",
    "retrieved_at",
    "pdb_release_date",
    "selection_rule_version",
    "bootstrap_group",
    "post_cutoff",
    "homology_audit_passed",
    "ppi_train_overlap",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def parse_bool(value: str, source: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{source}: expected true or false")
    return normalized == "true"


def parse_timestamp(value: str, source: str) -> None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{source}: invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: timestamp must include timezone")


def freeze(candidates_path: Path, review_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("structural-set output directory already exists")
    candidates = read_csv(candidates_path, set(CANDIDATE_FIELDS))
    reviews = read_csv(review_path, REVIEW_FIELDS)
    candidates_by_id: dict[str, dict[str, str]] = {}
    for line, row in enumerate(candidates, start=2):
        complex_id = row["complex_id"].strip()
        if not complex_id or complex_id in candidates_by_id:
            raise ValueError(f"{candidates_path.name}:{line}: missing or duplicate complex_id")
        candidates_by_id[complex_id] = row
    selected: list[dict[str, str]] = []
    review_records: list[dict[str, str]] = []
    seen_reviews: set[str] = set()
    for line, review in enumerate(reviews, start=2):
        complex_id = review["complex_id"].strip()
        if not complex_id or complex_id in seen_reviews:
            raise ValueError(f"{review_path.name}:{line}: missing or duplicate complex_id")
        seen_reviews.add(complex_id)
        if complex_id not in candidates_by_id:
            raise ValueError(f"{review_path.name}:{line}: complex_id is not in candidate inventory")
        include = parse_bool(review["include"], f"{review_path.name}:{line}:include")
        for field in (
            "biological_assembly_confirmed",
            "chain_mapping_confirmed",
            "human_heteromer_confirmed",
            "b4_b5_outputs_consulted",
        ):
            parse_bool(review[field], f"{review_path.name}:{line}:{field}")
        parse_timestamp(review["reviewed_at"], f"{review_path.name}:{line}:reviewed_at")
        if not review["reviewer_role"].strip() or not review["rationale"].strip():
            raise ValueError(f"{review_path.name}:{line}: reviewer_role and rationale are required")
        candidate = candidates_by_id[complex_id]
        if include:
            if candidate["automated_eligibility"].strip().lower() != "true":
                raise ValueError(f"{complex_id}: cannot include an automatically ineligible candidate")
            for field in (
                "biological_assembly_confirmed",
                "chain_mapping_confirmed",
                "human_heteromer_confirmed",
            ):
                if not parse_bool(review[field], f"{review_path.name}:{line}:{field}"):
                    raise ValueError(f"{complex_id}: included candidate lacks {field}")
            if parse_bool(review["b4_b5_outputs_consulted"], f"{review_path.name}:{line}:b4_b5_outputs_consulted"):
                raise ValueError(f"{complex_id}: selection was informed by B4/B5 output")
            selected.append(candidate)
        review_records.append({**review, "include": str(include).lower()})
    if len(selected) < MIN_COMPLEXES:
        raise ValueError(f"manual review selected {len(selected)} complexes; at least {MIN_COMPLEXES} are required")
    selected.sort(key=lambda row: row["complex_id"])
    output_dir.mkdir(parents=True)
    output_set = output_dir / "structural_set.csv"
    output_rows: list[dict[str, str]] = []
    for row in selected:
        output_rows.append(
            {
                "complex_id": row["complex_id"],
                "pdb_id": row["pdb_id"],
                "biological_assembly_id": row["biological_assembly_id"],
                "protein_chain_mapping_json": row["protein_chain_mapping_json"],
                "reference_structure": str((candidates_path.parent / row["reference_structure"]).resolve()),
                "reference_structure_sha256": row["reference_structure_sha256"],
                "source_url": row["source_url"],
                "retrieved_at": row["retrieved_at"],
                "pdb_release_date": row["pdb_release_date"],
                "selection_rule_version": row["selection_rule_version"],
                "bootstrap_group": row["complex_id"],
                "post_cutoff": "true",
                "homology_audit_passed": "false",
                "ppi_train_overlap": "false",
            }
        )
    with output_set.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    review_output = output_dir / "manual_review_records.csv"
    with review_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(review_records)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "freeze_structural_set_v0.1",
        "status": "frozen_for_homology_audit_only",
        "selection_rule_version": STRUCTURAL_SET_SELECTION_RULE,
        "b4_b5_outputs_consulted": False,
        "inputs": {
            "candidate_inventory": {
                "path": str(candidates_path.resolve()),
                "sha256": sha256_file(candidates_path),
                "rows": len(candidates),
            },
            "manual_review": {
                "path": str(review_path.resolve()),
                "sha256": sha256_file(review_path),
                "rows": len(reviews),
            },
        },
        "counts": {"reviewed": len(reviews), "included": len(selected), "excluded": len(reviews) - len(selected)},
        "outputs": {
            "structural_set.csv": {"sha256": sha256_file(output_set), "rows": len(output_rows)},
            "manual_review_records.csv": {"sha256": sha256_file(review_output), "rows": len(review_records)},
        },
        "next_required_step": "independent Linux homology audit against frozen PPI training partition",
    }
    metadata_path = output_dir / "freeze_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.candidates, args.review, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
