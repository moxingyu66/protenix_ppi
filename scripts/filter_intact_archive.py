#!/usr/bin/env python3
"""Stream-filter a pinned IntAct human MITAB archive without extracting it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


MI_RE = re.compile(r"MI:\d{4}")
OUTPUT_FIELDS = [
    "source_record_id",
    "primary_id_a",
    "primary_id_b",
    "alt_ids_a",
    "alt_ids_b",
    "taxid_a",
    "taxid_b",
    "interaction_type_mi",
    "detection_method_mi",
    "publication_ids",
    "interaction_type_raw",
    "detection_method_raw",
    "expansion_method",
    "source_database",
    "negative",
    "decision",
    "decision_reason",
]
REQUIRED_MITAB_FIELDS = {
    "#ID(s) interactor A",
    "ID(s) interactor B",
    "Alt. ID(s) interactor A",
    "Alt. ID(s) interactor B",
    "Interaction detection method(s)",
    "Publication Identifier(s)",
    "Taxid interactor A",
    "Taxid interactor B",
    "Interaction type(s)",
    "Source database(s)",
    "Interaction identifier(s)",
    "Expansion method(s)",
    "Type(s) interactor A",
    "Type(s) interactor B",
    "Negative",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def term_ids(raw: str) -> set[str]:
    return set(MI_RE.findall(raw or ""))


def compiled_term_ids(compiled: dict[str, Any], section: str, field: str) -> set[str]:
    values = compiled.get(section, {}).get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"compiled policy {section}.{field} must be a list")
    result = {str(item.get("id")) for item in values if isinstance(item, dict)}
    if len(result) != len(values) or any(not MI_RE.fullmatch(item) for item in result):
        raise ValueError(f"compiled policy {section}.{field} contains invalid entries")
    return result


def load_and_validate_manifest(path: Path, archive_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != "intact_human_archive":
        raise ValueError("source manifest is not for intact_human_archive")
    observed = manifest.get("observed", {})
    actual_bytes = archive_path.stat().st_size
    if observed.get("bytes") != actual_bytes:
        raise ValueError("IntAct archive byte count does not match source manifest")
    actual_hash = sha256_file(archive_path)
    if observed.get("sha256") != actual_hash:
        raise ValueError("IntAct archive SHA-256 does not match source manifest")
    if not observed.get("valid"):
        raise ValueError("IntAct source manifest does not mark the archive valid")
    return manifest


def classify_row(
    row: dict[str, str],
    interaction_accept: set[str],
    interaction_manual_descendants: set[str],
    interaction_never: set[str],
    detection_accept: set[str],
    detection_manual: set[str],
    detection_never: set[str],
) -> tuple[str, str, set[str], set[str]]:
    if "taxid:9606" not in row["Taxid interactor A"] or "taxid:9606" not in row["Taxid interactor B"]:
        return "excluded", "not_two_human_endpoints", set(), set()
    if "MI:0326" not in row["Type(s) interactor A"] or "MI:0326" not in row["Type(s) interactor B"]:
        return "excluded", "not_two_protein_endpoints", set(), set()
    if row["Negative"].strip().lower() == "true":
        return "excluded", "negative_record_in_positive_archive", set(), set()
    types = term_ids(row["Interaction type(s)"])
    methods = term_ids(row["Interaction detection method(s)"])
    if len(types) != 1:
        return "review", "missing_or_ambiguous_interaction_type", types, methods
    interaction_type = next(iter(types))
    if interaction_type in interaction_never:
        return "excluded", "generic_or_nonphysical_interaction_type", types, methods
    if interaction_type not in interaction_accept:
        reason = (
            "direct_interaction_descendant_requires_review"
            if interaction_type in interaction_manual_descendants
            else "not_direct_interaction_type"
        )
        return ("review" if interaction_type in interaction_manual_descendants else "excluded"), reason, types, methods
    if len(methods) != 1:
        return "review", "missing_or_ambiguous_detection_method", types, methods
    method = next(iter(methods))
    if method in detection_manual or method in detection_never:
        return "review", "detection_method_requires_review", types, methods
    if method not in detection_accept:
        return "review", "unlisted_detection_method", types, methods
    expansion = row["Expansion method(s)"].strip()
    if expansion not in {"", "-"}:
        return "review", "participant_expansion_requires_review", types, methods
    primary_a = row["#ID(s) interactor A"].strip()
    primary_b = row["ID(s) interactor B"].strip()
    if not primary_a.startswith("uniprotkb:") or not primary_b.startswith("uniprotkb:"):
        return "review", "non_uniprot_primary_identifier", types, methods
    if primary_a == primary_b:
        return "review", "self_pair_outside_primary_scope", types, methods
    if not row["Interaction identifier(s)"].strip() or row["Interaction identifier(s)"].strip() == "-":
        return "review", "missing_source_record_identifier", types, methods
    return "candidate", "auto_direct_candidate", types, methods


def output_row(
    row: dict[str, str], decision: str, reason: str, types: set[str], methods: set[str]
) -> dict[str, str]:
    return {
        "source_record_id": row["Interaction identifier(s)"].strip(),
        "primary_id_a": row["#ID(s) interactor A"].strip(),
        "primary_id_b": row["ID(s) interactor B"].strip(),
        "alt_ids_a": row["Alt. ID(s) interactor A"].strip(),
        "alt_ids_b": row["Alt. ID(s) interactor B"].strip(),
        "taxid_a": row["Taxid interactor A"].strip(),
        "taxid_b": row["Taxid interactor B"].strip(),
        "interaction_type_mi": "|".join(sorted(types)),
        "detection_method_mi": "|".join(sorted(methods)),
        "publication_ids": row["Publication Identifier(s)"].strip(),
        "interaction_type_raw": row["Interaction type(s)"].strip(),
        "detection_method_raw": row["Interaction detection method(s)"].strip(),
        "expansion_method": row["Expansion method(s)"].strip(),
        "source_database": row["Source database(s)"].strip(),
        "negative": row["Negative"].strip(),
        "decision": decision,
        "decision_reason": reason,
    }


def filter_archive(
    archive_path: Path,
    source_manifest_path: Path,
    compiled_policy_path: Path,
    output_dir: Path,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    source_manifest = load_and_validate_manifest(source_manifest_path, archive_path)
    compiled = json.loads(compiled_policy_path.read_text(encoding="utf-8"))
    interaction_accept = compiled_term_ids(compiled, "interaction_type", "auto_accept_exact")
    interaction_manual = compiled_term_ids(
        compiled, "interaction_type", "direct_descendants_manual_review"
    )
    interaction_never = compiled_term_ids(
        compiled, "interaction_type", "never_auto_accept_exact"
    )
    detection_accept = compiled_term_ids(
        compiled, "detection_method", "auto_accept_non_obsolete_closure"
    )
    detection_manual = compiled_term_ids(
        compiled,
        "detection_method",
        "manual_review_members_removed_from_auto_closure",
    )
    detection_never = compiled_term_ids(
        compiled, "detection_method", "never_auto_accept_closure"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "intact_direct_candidates.tsv"
    review_path = output_dir / "intact_manual_review.tsv"
    candidate_partial = candidate_path.with_suffix(candidate_path.suffix + ".partial")
    review_partial = review_path.with_suffix(review_path.suffix + ".partial")
    for partial in (candidate_partial, review_partial):
        if partial.exists():
            raise ValueError(f"partial output already exists: {partial}")

    counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        matching = [info for info in archive.infolist() if info.filename == "human.txt"]
        if len(matching) != 1:
            raise ValueError("IntAct archive must contain exactly one root-level human.txt")
        info = matching[0]
        csv.field_size_limit(min(math.floor(2**31 - 1), 1_000_000_000))
        try:
            with (
                archive.open(info, "r") as binary,
                io.TextIOWrapper(binary, encoding="utf-8", newline="") as text,
                candidate_partial.open("x", encoding="utf-8", newline="") as candidate_handle,
                review_partial.open("x", encoding="utf-8", newline="") as review_handle,
            ):
                reader = csv.DictReader(text, delimiter="\t")
                missing = REQUIRED_MITAB_FIELDS - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(
                        "IntAct human.txt is missing MITAB fields: " + ", ".join(sorted(missing))
                    )
                candidate_writer = csv.DictWriter(
                    candidate_handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
                )
                review_writer = csv.DictWriter(
                    review_handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
                )
                candidate_writer.writeheader()
                review_writer.writeheader()
                for row_number, row in enumerate(reader, start=1):
                    counts["rows_total"] += 1
                    if None in row:
                        raise ValueError(f"MITAB row {row_number + 1} has excess columns")
                    decision, reason, types, methods = classify_row(
                        row,
                        interaction_accept,
                        interaction_manual,
                        interaction_never,
                        detection_accept,
                        detection_manual,
                        detection_never,
                    )
                    counts[f"decision_{decision}"] += 1
                    counts[f"reason_{reason}"] += 1
                    if decision == "candidate":
                        candidate_writer.writerow(output_row(row, decision, reason, types, methods))
                    elif decision == "review":
                        review_writer.writerow(output_row(row, decision, reason, types, methods))
                    if progress_every > 0 and row_number % progress_every == 0:
                        print(
                            f"processed={row_number} candidates={counts['decision_candidate']} "
                            f"review={counts['decision_review']}",
                            flush=True,
                        )
        except Exception:
            for partial in (candidate_partial, review_partial):
                if partial.exists():
                    invalid = partial.with_suffix(partial.suffix + ".invalid")
                    os.replace(partial, invalid)
            raise
    os.replace(candidate_partial, candidate_path)
    os.replace(review_partial, review_path)
    metadata = {
        "schema_version": "1.0",
        "method": "stream_filter_intact_mitab_v0.1",
        "source_version": source_manifest.get("source_version"),
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": source_manifest["observed"]["sha256"],
        "zip_member": "human.txt",
        "zip_member_uncompressed_bytes": info.file_size,
        "compiled_policy": str(compiled_policy_path.resolve()),
        "compiled_policy_sha256": sha256_file(compiled_policy_path),
        "policy_id": compiled.get("policy_id"),
        "ontology_commit": compiled.get("ontology_commit"),
        "ontology_sha256": compiled.get("ontology_sha256"),
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "intact_direct_candidates.tsv": {
                "sha256": sha256_file(candidate_path),
                "bytes": candidate_path.stat().st_size,
            },
            "intact_manual_review.tsv": {
                "sha256": sha256_file(review_path),
                "bytes": review_path.stat().st_size,
            },
        },
        "interpretation_warning": "Candidate rows are evidence candidates, not normalized PPI labels. UniProt mapping, isoform policy, contradiction resolution, pair canonicalization, and group/leakage audits remain required.",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--compiled-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    metadata = filter_archive(
        args.archive,
        args.source_manifest,
        args.compiled_policy,
        args.output_dir,
        args.progress_every,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
