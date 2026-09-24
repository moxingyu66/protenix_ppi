#!/usr/bin/env python3
"""Fetch traceable post-cutoff human heteromer candidates from RCSB PDB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from protenix_ppi.scripts.evaluate_structure_preservation import (
    PRIMARY_CUTOFF,
    STRUCTURAL_SET_SELECTION_RULE,
    parse_mmcif_protein_chains,
)


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_ROOT = "https://data.rcsb.org/rest/v1/core"
FILES_ROOT = "https://files.rcsb.org/download"
REQUEST_USER_AGENT = "Protenix-PPI-structural-candidate-audit/0.1"
SCHEMA_VERSION = "1.0"
SELECTION_SEED = 20260915
MAX_RESOLUTION_ANGSTROM = 4.0
MIN_INTERFACE_RESIDUES = 20
MIN_BURIED_SURFACE_AREA = 500.0
ANTIBODY_PATTERN = re.compile(
    r"\b(antibody|immunoglobulin|fab|scfv|nanobody|heavy chain|light chain|vhh)\b",
    re.IGNORECASE,
)
CANDIDATE_FIELDS = [
    "complex_id",
    "pdb_id",
    "biological_assembly_id",
    "pdb_release_date",
    "experimental_method",
    "resolution_angstrom",
    "assembly_details",
    "assembly_method_details",
    "protein_chain_mapping_json",
    "protein_chain_entity_mapping_json",
    "uniprot_accessions",
    "entity_descriptions_json",
    "entity_lengths_json",
    "total_residues",
    "interface_residues",
    "buried_surface_area",
    "reference_structure",
    "reference_structure_sha256",
    "source_url",
    "retrieved_at",
    "selection_rule_version",
    "automated_eligibility",
    "manual_review_status",
    "exclusion_reasons",
]


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_candidate_key(identifier: str, seed: int = SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def build_search_query(start: int, rows: int) -> dict[str, Any]:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater",
                        "value": "2021-09-30T00:00:00Z",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_assembly_info.polymer_entity_instance_count_protein",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_assembly_info.polymer_composition",
                        "operator": "exact_match",
                        "value": "heteromeric protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                        "operator": "exact_match",
                        "value": "Homo sapiens",
                    },
                },
            ],
        },
        "return_type": "assembly",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }


class RcsbClient:
    def __init__(self, timeout: float = 30.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries

    def request_bytes(self, url: str, payload: dict[str, Any] | None = None) -> bytes:
        data = canonical_json(payload) if payload is not None else None
        headers = {"User-Agent": REQUEST_USER_AGENT, "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise ValueError(f"RCSB request failed after {self.retries} attempts: {url}: {last_error}")

    def request_json(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = self.request_bytes(url, payload)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"RCSB returned invalid JSON: {url}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"RCSB JSON response is not an object: {url}")
        return value


def _as_number(value: Any) -> float | None:
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    return int(number) if number is not None and number.is_integer() else None


def _source_taxonomy_ids(entity: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for item in entity.get("rcsb_entity_source_organism") or []:
        try:
            ids.add(int(item.get("ncbi_taxonomy_id")))
        except (TypeError, ValueError):
            pass
    for item in entity.get("entity_src_gen") or []:
        try:
            ids.add(int(item.get("pdbx_gene_src_ncbi_taxonomy_id")))
        except (TypeError, ValueError):
            pass
    return ids


def _entity_record(entity: dict[str, Any]) -> dict[str, Any]:
    identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    entity_id = str(identifiers.get("entity_id", "")).strip()
    uniprot = sorted({str(item).strip() for item in identifiers.get("uniprot_ids") or [] if str(item).strip()})
    asym_ids = sorted({str(item).strip() for item in identifiers.get("asym_ids") or [] if str(item).strip()})
    description = str((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description", "")).strip()
    length = _as_number((entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length"))
    polymer_type = str((entity.get("entity_poly") or {}).get("rcsb_entity_polymer_type", "")).strip()
    return {
        "entity_id": entity_id,
        "uniprot_ids": uniprot,
        "asym_ids": asym_ids,
        "description": description,
        "length": int(length) if length is not None and length.is_integer() else length,
        "polymer_type": polymer_type,
        "taxonomy_ids": sorted(_source_taxonomy_ids(entity)),
    }


def assess_metadata(
    entry: dict[str, Any], assembly: dict[str, Any], entities: list[dict[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    accession = entry.get("rcsb_accession_info") or {}
    entry_info = entry.get("rcsb_entry_info") or {}
    assembly_info = assembly.get("rcsb_assembly_info") or {}
    assembly_definition = assembly.get("pdbx_struct_assembly") or {}
    release_text = str(accession.get("initial_release_date", "")).strip()
    try:
        release_date = datetime.fromisoformat(release_text.replace("Z", "+00:00")).date()
    except ValueError:
        release_date = None
        reasons.append("missing_or_invalid_release_date")
    if release_date is not None and release_date <= PRIMARY_CUTOFF:
        reasons.append("not_post_cutoff")
    if str(entry_info.get("structure_determination_methodology", "")).lower() != "experimental":
        reasons.append("not_experimental")
    if str(assembly_info.get("polymer_composition", "")).lower() != "heteromeric protein":
        reasons.append("assembly_not_heteromeric_protein")
    if _as_int(assembly_info.get("polymer_entity_instance_count_protein")) != 2:
        reasons.append("assembly_not_exactly_two_protein_instances")
    if _as_int(assembly_info.get("polymer_entity_instance_count_nucleic_acid")) != 0:
        reasons.append("assembly_contains_nucleic_acid")
    if assembly_definition.get("rcsb_candidate_assembly") != "Y":
        reasons.append("not_rcsb_candidate_biological_assembly")
    resolution = _as_number(entry_info.get("resolution_combined"))
    if resolution is None:
        reasons.append("missing_resolution")
    elif resolution > MAX_RESOLUTION_ANGSTROM:
        reasons.append("resolution_above_4_angstrom")
    interface_residues = _as_number(assembly_info.get("total_number_interface_residues"))
    if interface_residues is None or interface_residues < MIN_INTERFACE_RESIDUES:
        reasons.append("interface_has_fewer_than_20_residues")
    buried_area = _as_number(assembly_info.get("total_assembly_buried_surface_area"))
    if buried_area is None or buried_area < MIN_BURIED_SURFACE_AREA:
        reasons.append("buried_surface_area_below_500")

    parsed_entities = [_entity_record(entity) for entity in entities]
    if len(parsed_entities) != 2:
        reasons.append("assembly_does_not_resolve_to_two_polymer_entities")
    for item in parsed_entities:
        entity_id = item["entity_id"] or "unknown"
        if item["polymer_type"] != "Protein":
            reasons.append(f"entity_{entity_id}_not_protein")
        if item["taxonomy_ids"] != [9606]:
            reasons.append(f"entity_{entity_id}_not_unambiguously_human")
        if len(item["uniprot_ids"]) != 1:
            reasons.append(f"entity_{entity_id}_does_not_have_one_uniprot")
        if not item["length"] or item["length"] < 30:
            reasons.append(f"entity_{entity_id}_shorter_than_30_residues")
        if ANTIBODY_PATTERN.search(item["description"]):
            reasons.append(f"entity_{entity_id}_antibody_like")
    accessions = [item["uniprot_ids"][0] for item in parsed_entities if len(item["uniprot_ids"]) == 1]
    if len(accessions) == 2 and len(set(accessions)) != 2:
        reasons.append("uniprot_pair_is_homomeric")
    return sorted(set(reasons)), {
        "release_date": release_date.isoformat() if release_date else "",
        "resolution": resolution,
        "experimental_method": str(entry_info.get("experimental_method", "")).strip(),
        "assembly_details": str(assembly_definition.get("details", "")).strip(),
        "assembly_method_details": str(assembly_definition.get("method_details", "")).strip(),
        "interface_residues": interface_residues,
        "buried_surface_area": buried_area,
        "entities": parsed_entities,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _assembly_entity_ids(
    assembly: dict[str, Any], entity_by_asym: dict[str, str]
) -> tuple[set[str], list[str]]:
    assembly_asym: set[str] = set()
    for item in assembly.get("pdbx_struct_assembly_gen") or []:
        assembly_asym.update(str(value).strip() for value in item.get("asym_id_list") or [])
    entity_ids = {entity_by_asym[asym] for asym in assembly_asym if asym in entity_by_asym}
    return entity_ids, sorted(assembly_asym)


def fetch_candidate(
    identifier: str,
    client: RcsbClient,
    output_dir: Path,
    retrieved_at: str,
) -> dict[str, str]:
    try:
        pdb_id, assembly_id = identifier.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(f"invalid RCSB assembly identifier: {identifier}") from exc
    pdb_id = pdb_id.upper()
    raw_dir = output_dir / "raw" / "candidates" / f"{pdb_id}-{assembly_id}"
    entry = client.request_json(f"{DATA_ROOT}/entry/{pdb_id}")
    assembly = client.request_json(f"{DATA_ROOT}/assembly/{pdb_id}/{assembly_id}")
    _write_json(raw_dir / "entry.json", entry)
    _write_json(raw_dir / "assembly.json", assembly)
    polymer_ids = [
        str(item) for item in (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
    ]
    all_entities: list[dict[str, Any]] = []
    entity_by_asym: dict[str, str] = {}
    for entity_id in polymer_ids:
        entity = client.request_json(f"{DATA_ROOT}/polymer_entity/{pdb_id}/{entity_id}")
        all_entities.append(entity)
        _write_json(raw_dir / f"polymer_entity_{entity_id}.json", entity)
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        for asym in identifiers.get("asym_ids") or []:
            entity_by_asym[str(asym)] = entity_id
    assembly_entity_ids, _ = _assembly_entity_ids(assembly, entity_by_asym)
    entities = [
        entity
        for entity in all_entities
        if str((entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("entity_id"))
        in assembly_entity_ids
    ]
    reasons, details = assess_metadata(entry, assembly, entities)
    source_url = f"{FILES_ROOT}/{pdb_id}-assembly{assembly_id}.cif"
    reference_relative = ""
    reference_sha256 = ""
    chain_mapping: dict[str, str] = {}
    chain_entities: dict[str, str] = {}
    if not reasons:
        reference_bytes = client.request_bytes(source_url)
        reference_path = output_dir / "references" / f"{pdb_id}-assembly{assembly_id}.cif"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(reference_bytes)
        reference_relative = str(reference_path.relative_to(output_dir)).replace("\\", "/")
        reference_sha256 = sha256_bytes(reference_bytes)
        try:
            chain_entities = parse_mmcif_protein_chains(reference_path)
        except ValueError as exc:
            reasons.append("assembly_mmcif_chain_parse_failed")
            details["mmcif_error"] = str(exc)
        entity_to_uniprot = {
            item["entity_id"]: item["uniprot_ids"][0]
            for item in details["entities"]
            if len(item["uniprot_ids"]) == 1
        }
        if chain_entities:
            unknown = sorted(set(chain_entities.values()) - set(entity_to_uniprot))
            if unknown:
                reasons.append("assembly_chain_entity_missing_uniprot")
            else:
                chain_mapping = {
                    chain: entity_to_uniprot[entity_id]
                    for chain, entity_id in sorted(chain_entities.items())
                }
                if len(chain_mapping) != 2:
                    reasons.append("downloaded_assembly_not_exactly_two_protein_chains")
                if len(set(chain_mapping.values())) != 2:
                    reasons.append("downloaded_assembly_not_heteromeric")

    parsed_entities = details["entities"]
    accessions = sorted(
        item["uniprot_ids"][0] for item in parsed_entities if len(item["uniprot_ids"]) == 1
    )
    lengths = {item["entity_id"]: item["length"] for item in parsed_entities}
    descriptions = {item["entity_id"]: item["description"] for item in parsed_entities}
    total_residues = sum(value for value in lengths.values() if isinstance(value, int))
    reasons = sorted(set(reasons))
    return {
        "complex_id": f"{pdb_id}_assembly_{assembly_id}",
        "pdb_id": pdb_id,
        "biological_assembly_id": assembly_id,
        "pdb_release_date": details["release_date"],
        "experimental_method": details["experimental_method"],
        "resolution_angstrom": "" if details["resolution"] is None else str(details["resolution"]),
        "assembly_details": details["assembly_details"],
        "assembly_method_details": details["assembly_method_details"],
        "protein_chain_mapping_json": json.dumps(chain_mapping, sort_keys=True, separators=(",", ":")),
        "protein_chain_entity_mapping_json": json.dumps(
            chain_entities, sort_keys=True, separators=(",", ":")
        ),
        "uniprot_accessions": ";".join(accessions),
        "entity_descriptions_json": json.dumps(descriptions, sort_keys=True, separators=(",", ":")),
        "entity_lengths_json": json.dumps(lengths, sort_keys=True, separators=(",", ":")),
        "total_residues": str(total_residues),
        "interface_residues": "" if details["interface_residues"] is None else str(details["interface_residues"]),
        "buried_surface_area": "" if details["buried_surface_area"] is None else str(details["buried_surface_area"]),
        "reference_structure": reference_relative,
        "reference_structure_sha256": reference_sha256,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "selection_rule_version": STRUCTURAL_SET_SELECTION_RULE,
        "automated_eligibility": "true" if not reasons else "false",
        "manual_review_status": "pending" if not reasons else "not_applicable",
        "exclusion_reasons": ";".join(reasons),
    }


def fetch_search_universe(
    client: RcsbClient,
    output_dir: Path,
    *,
    universe_limit: int | None = None,
    page_size: int = 500,
) -> tuple[list[str], int, list[dict[str, Any]]]:
    identifiers: list[str] = []
    pages: list[dict[str, Any]] = []
    total_count = 0
    start = 0
    while universe_limit is None or len(identifiers) < universe_limit:
        rows = page_size if universe_limit is None else min(page_size, universe_limit - len(identifiers))
        query = build_search_query(start, rows)
        response = client.request_json(SEARCH_URL, query)
        pages.append(response)
        if len(pages) == 1:
            total_count = int(response.get("total_count", 0))
        result_set = response.get("result_set") or []
        page_ids = [str(item.get("identifier", "")).strip() for item in result_set]
        page_ids = [item for item in page_ids if item]
        if not page_ids:
            break
        identifiers.extend(page_ids)
        start += len(page_ids)
        if start >= total_count:
            break
    for index, response in enumerate(pages):
        _write_json(output_dir / "raw" / "search" / f"page_{index:04d}.json", response)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RCSB search universe contains duplicate assembly identifiers")
    return identifiers, total_count, pages


def fetch_all(
    output_dir: Path,
    *,
    search_limit: int = 250,
    selection_seed: int = SELECTION_SEED,
    universe_limit: int | None = None,
    workers: int = 8,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("structural candidate output directory already exists")
    output_dir.mkdir(parents=True)
    client = RcsbClient(timeout=timeout, retries=retries)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    query_template = build_search_query(0, 500)
    _write_json(output_dir / "search_query.json", query_template)
    universe, total_count, pages = fetch_search_universe(
        client, output_dir, universe_limit=universe_limit
    )
    identifiers = sorted(
        universe, key=lambda item: (stable_candidate_key(item, selection_seed), item)
    )[:search_limit]
    _write_json(
        output_dir / "selected_assembly_ids.json",
        {
            "selection_seed": selection_seed,
            "universe_count": len(universe),
            "selected_count": len(identifiers),
            "identifiers": identifiers,
        },
    )
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_candidate, identifier, client, output_dir, retrieved_at): identifier
            for identifier in identifiers
        }
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # preserve network/schema failures as auditable rows
                failures.append({"identifier": identifier, "error": str(exc)})
    rows.sort(key=lambda row: (row["pdb_id"], int(row["biological_assembly_id"])))
    inventory_path = output_dir / "structural_candidates.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output_dir / "fetch_failures.json", failures)
    eligible = [row for row in rows if row["automated_eligibility"] == "true"]
    exclusion_counts: dict[str, int] = {}
    for row in rows:
        for reason in filter(None, row["exclusion_reasons"].split(";")):
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": STRUCTURAL_SET_SELECTION_RULE,
        "status": "provisional_candidates_requires_manual_review_and_homology_audit",
        "retrieved_at": retrieved_at,
        "sources": {
            "search_api": SEARCH_URL,
            "data_api": DATA_ROOT,
            "files": FILES_ROOT,
        },
        "selection_thresholds": {
            "primary_model_cutoff": PRIMARY_CUTOFF.isoformat(),
            "maximum_resolution_angstrom": MAX_RESOLUTION_ANGSTROM,
            "minimum_interface_residues": MIN_INTERFACE_RESIDUES,
            "minimum_buried_surface_area": MIN_BURIED_SURFACE_AREA,
            "protein_instances": 2,
            "human_taxonomy_id": 9606,
            "unique_uniprot_ids": 2,
            "antibody_like_entities_excluded": True,
        },
        "request": {
            "search_limit": search_limit,
            "selection_seed": selection_seed,
            "universe_limit": universe_limit,
            "workers": workers,
            "timeout_seconds": timeout,
            "retries": retries,
            "query_sha256": sha256_file(output_dir / "search_query.json"),
            "search_pages": len(pages),
            "reported_total_count": total_count,
            "scanned_universe_count": len(universe),
        },
        "counts": {
            "requested_identifiers": len(identifiers),
            "completed_records": len(rows),
            "request_or_schema_failures": len(failures),
            "automated_eligible": len(eligible),
            "excluded": len(rows) - len(eligible),
        },
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "outputs": {
            "selected_assembly_ids.json": {
                "rows": len(identifiers),
                "sha256": sha256_file(output_dir / "selected_assembly_ids.json"),
            },
            "structural_candidates.csv": {
                "rows": len(rows),
                "sha256": sha256_file(inventory_path),
            },
            "fetch_failures.json": {
                "rows": len(failures),
                "sha256": sha256_file(output_dir / "fetch_failures.json"),
            },
        },
        "warnings": [
            "Automated eligibility is not final inclusion.",
            "Every candidate requires biological-assembly manual review and Linux homology audit.",
            "No B4/B5 output may be consulted during selection.",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-limit", type=int, default=250)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument(
        "--universe-limit",
        type=int,
        help="Debug-only cap on the RCSB search universe; omit for a final acquisition",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.search_limit <= 0:
        raise ValueError("search-limit must be positive")
    if args.universe_limit is not None and args.universe_limit < args.search_limit:
        raise ValueError("universe-limit cannot be smaller than search-limit")
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if args.timeout <= 0 or args.retries <= 0:
        raise ValueError("timeout and retries must be positive")
    result = fetch_all(
        args.output_dir,
        search_limit=args.search_limit,
        selection_seed=args.selection_seed,
        universe_limit=args.universe_limit,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["counts"]["request_or_schema_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
