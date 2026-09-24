#!/usr/bin/env python3
"""Extract explicitly assayed RRS negatives from Luck et al. 2020 tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from protenix_ppi.scripts.build_identifier_inventory import INVENTORY_FIELDS, sha256_file


SOURCE_DATABASE = "Luck2020"
SOURCE_VERSION = "Nature_supplement_2020-04-08"
ENSG_VERSIONED = re.compile(r"^(ENSG\d{11})(?:\.\d+)?$")
CANDIDATE_FIELDS = [
    "source_table", "source_record_id", "orf_id_a", "orf_id_b", "ensembl_gene_a",
    "ensembl_gene_b", "assay", "detection_method_mi", "publication_id",
    "negative_definition",
]
QUARANTINE_FIELDS = [
    "source_table", "source_record_id", "orf_id_a", "orf_id_b",
    "mapping_count_a", "mapping_count_b", "reason",
]


def validate_archive(path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != "luck2020_supplementary_tables":
        raise ValueError("source manifest is not for Luck2020 supplementary tables")
    observed = manifest.get("observed", {})
    if path.stat().st_size != observed.get("bytes") or sha256_file(path) != observed.get("sha256"):
        raise ValueError("Luck2020 supplementary archive does not match source manifest")
    if not observed.get("valid"):
        raise ValueError("Luck2020 source manifest does not mark the archive valid")
    return manifest


def read_table(archive: zipfile.ZipFile, number: int) -> list[dict[str, str]]:
    member = f"Supplementary_Tables_new/Supplementary Table {number}.txt"
    matching = [item for item in archive.infolist() if item.filename == member]
    if len(matching) != 1:
        raise ValueError(f"supplementary archive must contain exactly one {member}")
    text = archive.read(matching[0]).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def build_orf_mapping(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    mapping: defaultdict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        orf = row.get("orf_id", "").strip()
        raw_gene = row.get("ensembl_gene_id", "").strip()
        match = ENSG_VERSIONED.fullmatch(raw_gene)
        if not orf or not match:
            raise ValueError(f"invalid ORF/Ensembl mapping at supplementary table 2 line {line_number}")
        mapping[orf].add(match.group(1))
    return dict(mapping)


def extract_negatives(archive_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    manifest = validate_archive(archive_path, manifest_path)
    with zipfile.ZipFile(archive_path) as archive:
        orf_mapping = build_orf_mapping(read_table(archive, 2))
        table5 = read_table(archive, 5)
        table6 = read_table(archive, 6)

    candidates = []
    quarantine = []
    counts: Counter[str] = Counter()

    def consider(table_number: int, line_number: int, row: dict[str, str]) -> None:
        if table_number == 5:
            is_rrs = row["source"].strip() == "RRS_V1"
            result = row["score"].strip()
            orf_a, orf_b = row["ad_orf_id"].strip(), row["db_orf_id"].strip()
            assay = f"Y2H_v{row['assay_version'].strip()}"
            detection_method = "MI:0018"
            definition = "RRS_pair_explicitly_negative_in_Y2H_pairwise_test"
        else:
            is_rrs = row["source"].strip() == "RRS"
            result = row["result"].strip()
            orf_a, orf_b = row["orf_id_a"].strip(), row["orf_id_b"].strip()
            assay = f"MAPPIT_{row['experiment'].strip()}"
            detection_method = "MI:0231"
            definition = "RRS_pair_explicitly_negative_in_MAPPIT_pairwise_test"
        if not is_rrs:
            counts[f"table{table_number}_excluded_non_rrs"] += 1
            return
        negative_token = "0" if table_number == 5 else "0.0"
        if result != negative_token:
            reason = {
                "1": "positive_result",
                "1.0": "positive_result",
                "NA": "invalid_test",
                "AA": "autoactivator",
                "": "missing_result",
            }.get(result, "unrecognized_result")
            counts[f"table{table_number}_excluded_{reason}"] += 1
            return
        genes_a = orf_mapping.get(orf_a, set())
        genes_b = orf_mapping.get(orf_b, set())
        source_record_id = f"Luck2020-ST{table_number}-line:{line_number}"
        if len(genes_a) != 1 or len(genes_b) != 1:
            quarantine.append(
                {
                    "source_table": str(table_number),
                    "source_record_id": source_record_id,
                    "orf_id_a": orf_a,
                    "orf_id_b": orf_b,
                    "mapping_count_a": str(len(genes_a)),
                    "mapping_count_b": str(len(genes_b)),
                    "reason": "orf_to_ensembl_not_one_to_one",
                }
            )
            counts[f"table{table_number}_quarantine_ambiguous_orf_mapping"] += 1
            return
        gene_a, gene_b = next(iter(genes_a)), next(iter(genes_b))
        candidates.append(
            {
                "source_table": str(table_number),
                "source_record_id": source_record_id,
                "orf_id_a": orf_a,
                "orf_id_b": orf_b,
                "ensembl_gene_a": gene_a,
                "ensembl_gene_b": gene_b,
                "assay": assay,
                "detection_method_mi": detection_method,
                "publication_id": "PMID:32296183",
                "negative_definition": definition,
            }
        )
        counts[f"table{table_number}_candidate_negative"] += 1

    for line_number, row in enumerate(table5, start=2):
        consider(5, line_number, row)
    for line_number, row in enumerate(table6, start=2):
        consider(6, line_number, row)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "screen_negative_candidates.tsv"
    quarantine_path = output_dir / "source_quarantine.tsv"
    for path, fields, rows in (
        (candidate_path, CANDIDATE_FIELDS, candidates),
        (quarantine_path, QUARANTINE_FIELDS, quarantine),
    ):
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    inventory_counts: Counter[str] = Counter()
    for row in candidates:
        inventory_counts[row["ensembl_gene_a"]] += 1
        inventory_counts[row["ensembl_gene_b"]] += 1
    inventory_path = output_dir / "identifier_inventory.tsv"
    with inventory_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for identifier in sorted(inventory_counts):
            writer.writerow(
                {
                    "source_database": SOURCE_DATABASE,
                    "source_version": SOURCE_VERSION,
                    "evidence_role": "negative_evidence",
                    "namespace": "Ensembl",
                    "raw_identifier": identifier,
                    "mapping_query_id": identifier,
                    "isoform_specific": "false",
                    "mapping_status": "mapping_candidate",
                    "occurrence_count": inventory_counts[identifier],
                }
            )
    request_path = output_dir / "ensembl_gene_ids.txt"
    request_path.write_text("\n".join(sorted(inventory_counts)) + "\n", encoding="utf-8")

    metadata = {
        "schema_version": "1.0",
        "method": "extract_luck2020_rrs_screen_negatives_v0.1",
        "source_version": manifest["source_version"],
        "source_archive_sha256": manifest["observed"]["sha256"],
        "counts": {
            **dict(sorted(counts.items())),
            "candidate_rows": len(candidates),
            "candidate_unique_ensembl_pairs": len(
                {tuple(sorted((row["ensembl_gene_a"], row["ensembl_gene_b"]))) for row in candidates}
            ),
            "unique_ensembl_genes_for_mapping": len(inventory_counts),
            "quarantine_rows": len(quarantine),
        },
        "outputs": {
            "screen_negative_candidates.tsv": {"sha256": sha256_file(candidate_path), "rows": len(candidates)},
            "source_quarantine.tsv": {"sha256": sha256_file(quarantine_path), "rows": len(quarantine)},
            "identifier_inventory.tsv": {"sha256": sha256_file(inventory_path), "rows": len(inventory_counts)},
            "ensembl_gene_ids.txt": {"sha256": sha256_file(request_path), "rows": len(inventory_counts)},
        },
        "interpretation_warning": (
            "Candidates are explicitly negative assay observations for randomly selected RRS pairs. "
            "They are a screen-negative stratum, not universal evidence of non-binding."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = extract_negatives(args.archive, args.source_manifest, args.output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
