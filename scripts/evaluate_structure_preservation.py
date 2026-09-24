#!/usr/bin/env python3
"""Evaluate paired structural preservation and the B4/B5 non-inferiority gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shlex
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

from protenix_ppi.scripts.evaluate_predictions import percentile


PRIMARY_CUTOFF = date(2021, 9, 30)
MIN_COMPLEXES = 30
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260915
DOCKQ_NONINFERIORITY_MARGIN = -0.05
ACCEPTABLE_FRACTION_MARGIN = -0.10
ACCEPTABLE_DOCKQ = 0.23
HEX = set("0123456789abcdef")
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Z0-9]{3}$")
ASSEMBLY_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
STRUCTURAL_SET_SELECTION_RULE = "rcsb_human_heteromer_v0.1"
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
}

SET_FIELDS = {
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
}
METRIC_FIELDS = {
    "complex_id",
    "baseline_structure",
    "baseline_structure_sha256",
    "candidate_structure",
    "candidate_structure_sha256",
    "baseline_dockq",
    "candidate_dockq",
    "baseline_irmsd",
    "candidate_irmsd",
    "baseline_lrmsd",
    "candidate_lrmsd",
    "baseline_interface_contact_precision",
    "candidate_interface_contact_precision",
    "baseline_interface_contact_recall",
    "candidate_interface_contact_recall",
    "baseline_chain_collapse",
    "candidate_chain_collapse",
    "baseline_atomic_overlap",
    "candidate_atomic_overlap",
    "baseline_within_chain_geometry_failure",
    "candidate_within_chain_geometry_failure",
}


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


def parse_bool(value: str, location: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{location}: expected true or false")
    return normalized == "true"


def parse_number(value: str, location: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{location}: invalid number") from exc
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{location}: number outside [{minimum}, {maximum}]")
    return number


def resolve_hashed_file(value: str, digest: str, table_path: Path, location: str) -> Path:
    path = Path(value.strip())
    if not path.is_absolute():
        path = table_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{location}: structure file does not exist: {path}")
    normalized_digest = digest.strip().lower()
    if len(normalized_digest) != 64 or any(char not in HEX for char in normalized_digest):
        raise ValueError(f"{location}: invalid SHA-256")
    if sha256_file(path) != normalized_digest:
        raise ValueError(f"{location}: structure SHA-256 mismatch")
    return path


def parse_mmcif_protein_chains(path: Path) -> dict[str, str]:
    """Return label_asym_id -> label_entity_id from the assembly ATOM loop."""
    headers: list[str] = []
    collecting_loop = False
    atom_loop_seen = False
    parsed: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            if text == "loop_":
                headers = []
                collecting_loop = True
                atom_loop_seen = False
                continue
            if collecting_loop and text.startswith("_"):
                header = text.split()[0]
                headers.append(header)
                atom_loop_seen = atom_loop_seen or header.startswith("_atom_site.")
                continue
            if not collecting_loop or not atom_loop_seen:
                continue
            if text.startswith("#") or text.startswith("_") or text == "loop_":
                if parsed:
                    break
                headers = []
                collecting_loop = False
                atom_loop_seen = False
                continue
            required = {
                "_atom_site.group_PDB",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
            }
            if not required.issubset(headers):
                raise ValueError(f"{path.name}: atom_site loop lacks chain/entity fields")
            try:
                tokens = shlex.split(text, comments=False, posix=True)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: malformed atom_site row") from exc
            if len(tokens) != len(headers):
                raise ValueError(f"{path.name}:{line_number}: atom_site column count mismatch")
            row = dict(zip(headers, tokens))
            if row["_atom_site.group_PDB"].upper() != "ATOM":
                continue
            chain_id = row["_atom_site.label_asym_id"].strip()
            entity_id = row["_atom_site.label_entity_id"].strip()
            if not chain_id or chain_id in {".", "?"} or not entity_id or entity_id in {".", "?"}:
                raise ValueError(f"{path.name}:{line_number}: missing protein chain/entity ID")
            previous = parsed.setdefault(chain_id, entity_id)
            if previous != entity_id:
                raise ValueError(f"{path.name}: one protein chain maps to multiple entities")
    if len(parsed) < 2:
        raise ValueError(f"{path.name}: biological assembly contains fewer than two protein chains")
    return parsed


def parse_mmcif_chain_sequences(path: Path) -> dict[str, str]:
    """Extract one-letter protein sequences from the first ATOM model."""
    headers: list[str] = []
    collecting_loop = False
    atom_loop_seen = False
    residue_by_chain: dict[str, dict[tuple[int, str], str]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            if text == "loop_":
                headers = []
                collecting_loop = True
                atom_loop_seen = False
                continue
            if collecting_loop and text.startswith("_"):
                header = text.split()[0]
                headers.append(header)
                atom_loop_seen = atom_loop_seen or header.startswith("_atom_site.")
                continue
            if not collecting_loop or not atom_loop_seen:
                continue
            if text.startswith("#") or text.startswith("_") or text == "loop_":
                if residue_by_chain:
                    break
                headers = []
                collecting_loop = False
                atom_loop_seen = False
                continue
            required = {
                "_atom_site.group_PDB",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.label_comp_id",
            }
            if not required.issubset(headers):
                raise ValueError(f"{path.name}: atom_site loop lacks sequence fields")
            try:
                tokens = shlex.split(text, comments=False, posix=True)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: malformed atom_site row") from exc
            if len(tokens) != len(headers):
                raise ValueError(f"{path.name}:{line_number}: atom_site column count mismatch")
            row = dict(zip(headers, tokens))
            group = row["_atom_site.group_PDB"].upper()
            chain = row["_atom_site.label_asym_id"].strip()
            seq_id = row["_atom_site.label_seq_id"].strip()
            residue = row["_atom_site.label_comp_id"].strip().upper()
            if group != "ATOM" and residue not in {"MSE", "SEC"}:
                continue
            if not chain or chain in {".", "?"} or not seq_id or seq_id in {".", "?"}:
                continue
            try:
                numeric_seq = int(seq_id)
            except ValueError:
                raise ValueError(f"{path.name}:{line_number}: non-integer label_seq_id")
            residue_by_chain[chain].setdefault((numeric_seq, residue), AA3_TO_AA1.get(residue, "X"))
    sequences: dict[str, str] = {}
    for chain, residues in sorted(residue_by_chain.items()):
        by_position: dict[int, str] = {}
        for (position, _residue_name), amino_acid in sorted(residues.items()):
            by_position.setdefault(position, amino_acid)
        sequence = "".join(by_position[position] for position in sorted(by_position))
        if sequence:
            sequences[chain] = sequence
    if len(sequences) < 2:
        raise ValueError(f"{path.name}: fewer than two ATOM protein chains have sequences")
    return sequences


def validate_structural_set_provenance(
    row: dict[str, str], table_path: Path, line: int, reference_path: Path
) -> dict[str, object]:
    """Validate biological-assembly and chain provenance for one set row."""
    location = f"{table_path.name}:{line}"
    pdb_id = row["pdb_id"].strip().upper()
    if not PDB_ID_PATTERN.fullmatch(pdb_id):
        raise ValueError(f"{location}: invalid pdb_id")
    assembly_id = row["biological_assembly_id"].strip()
    if not ASSEMBLY_ID_PATTERN.fullmatch(assembly_id):
        raise ValueError(f"{location}: invalid biological_assembly_id")
    try:
        mapping = json.loads(row["protein_chain_mapping_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: invalid protein_chain_mapping_json") from exc
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError(f"{location}: chain mapping must contain at least two protein chains")
    normalized_mapping: dict[str, str] = {}
    for chain, accession in mapping.items():
        chain_id = str(chain).strip()
        protein_id = str(accession).strip()
        if not chain_id or not protein_id or "REPLACE" in protein_id.upper():
            raise ValueError(f"{location}: chain mapping contains an empty or placeholder value")
        if chain_id in normalized_mapping:
            raise ValueError(f"{location}: duplicate chain in mapping")
        normalized_mapping[chain_id] = protein_id
    if len(set(normalized_mapping.values())) < 2:
        raise ValueError(f"{location}: reference is not a heteromeric protein complex")
    structure_chains = parse_mmcif_protein_chains(reference_path)
    if set(normalized_mapping) != set(structure_chains):
        raise ValueError(f"{location}: chain mapping does not exactly cover assembly ATOM chains")

    source_url = row["source_url"].strip()
    parsed = urlparse(source_url)
    expected_name = f"{pdb_id}-assembly{assembly_id}.cif".lower()
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "files.rcsb.org"
        or Path(parsed.path).name.lower() != expected_name
    ):
        raise ValueError(f"{location}: source_url is not the matching RCSB biological assembly")
    retrieved_text = row["retrieved_at"].strip()
    normalized_timestamp = retrieved_text[:-1] + "+00:00" if retrieved_text.endswith("Z") else retrieved_text
    try:
        retrieved_at = datetime.fromisoformat(normalized_timestamp)
    except ValueError as exc:
        raise ValueError(f"{location}: invalid retrieved_at timestamp") from exc
    if retrieved_at.tzinfo is None:
        raise ValueError(f"{location}: retrieved_at must include a timezone")
    if row["selection_rule_version"].strip() != STRUCTURAL_SET_SELECTION_RULE:
        raise ValueError(f"{location}: unexpected selection_rule_version")
    return {
        "pdb_id": pdb_id,
        "biological_assembly_id": assembly_id,
        "protein_chain_mapping": normalized_mapping,
        "protein_chain_entity_mapping": structure_chains,
        "source_url": source_url,
        "retrieved_at": retrieved_text,
    }


def bootstrap_delta(rows: list[dict], value_fn, replicates: int, seed: int) -> dict:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["bootstrap_group"]].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        sample_rows = [row for group in sampled for row in groups[group]]
        values.append(float(value_fn(sample_rows)))
    return {
        "replicates": replicates,
        "seed": seed,
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
    }


def mean_delta(rows: list[dict], field: str) -> float:
    return statistics.fmean(row[f"candidate_{field}"] - row[f"baseline_{field}"] for row in rows)


def acceptable_delta(rows: list[dict]) -> float:
    baseline = statistics.fmean(row["baseline_dockq"] >= ACCEPTABLE_DOCKQ for row in rows)
    candidate = statistics.fmean(row["candidate_dockq"] >= ACCEPTABLE_DOCKQ for row in rows)
    return candidate - baseline


def evaluate(
    structural_set_path: Path,
    metrics_path: Path,
    min_complexes: int = MIN_COMPLEXES,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    set_rows = read_csv(structural_set_path, SET_FIELDS)
    metric_rows = read_csv(metrics_path, METRIC_FIELDS)
    if len(set_rows) < min_complexes:
        raise ValueError(f"structural preservation set requires at least {min_complexes} complexes")
    set_by_id: dict[str, dict] = {}
    for line, row in enumerate(set_rows, start=2):
        complex_id = row["complex_id"].strip()
        if not complex_id or complex_id in set_by_id:
            raise ValueError(f"{structural_set_path.name}:{line}: missing or duplicate complex_id")
        try:
            release_date = date.fromisoformat(row["pdb_release_date"].strip())
        except ValueError as exc:
            raise ValueError(f"{structural_set_path.name}:{line}: invalid pdb_release_date") from exc
        post_cutoff = parse_bool(row["post_cutoff"], f"{structural_set_path.name}:{line}:post_cutoff")
        homology_passed = parse_bool(
            row["homology_audit_passed"], f"{structural_set_path.name}:{line}:homology_audit_passed"
        )
        training_overlap = parse_bool(
            row["ppi_train_overlap"], f"{structural_set_path.name}:{line}:ppi_train_overlap"
        )
        if release_date <= PRIMARY_CUTOFF or not post_cutoff:
            raise ValueError(f"{complex_id}: structural reference is not post-cutoff")
        if not homology_passed or training_overlap:
            raise ValueError(f"{complex_id}: structural set independence audit failed")
        reference_path = resolve_hashed_file(
            row["reference_structure"],
            row["reference_structure_sha256"],
            structural_set_path,
            f"{structural_set_path.name}:{line}:reference",
        )
        provenance = validate_structural_set_provenance(
            row, structural_set_path, line, reference_path
        )
        set_by_id[complex_id] = {
            "bootstrap_group": row["bootstrap_group"].strip() or complex_id,
            "release_date": release_date.isoformat(),
            **provenance,
        }

    parsed: list[dict] = []
    seen: set[str] = set()
    unit_interval_fields = ("dockq", "interface_contact_precision", "interface_contact_recall")
    rmsd_fields = ("irmsd", "lrmsd")
    failure_fields = ("chain_collapse", "atomic_overlap", "within_chain_geometry_failure")
    for line, row in enumerate(metric_rows, start=2):
        complex_id = row["complex_id"].strip()
        if not complex_id or complex_id in seen or complex_id not in set_by_id:
            raise ValueError(f"{metrics_path.name}:{line}: duplicate or unexpected complex_id {complex_id!r}")
        seen.add(complex_id)
        resolve_hashed_file(
            row["baseline_structure"], row["baseline_structure_sha256"], metrics_path, f"{complex_id}:baseline"
        )
        resolve_hashed_file(
            row["candidate_structure"], row["candidate_structure_sha256"], metrics_path, f"{complex_id}:candidate"
        )
        item = {"complex_id": complex_id, "bootstrap_group": set_by_id[complex_id]["bootstrap_group"]}
        for field in unit_interval_fields:
            item[f"baseline_{field}"] = parse_number(row[f"baseline_{field}"], f"{complex_id}:baseline_{field}", 0, 1)
            item[f"candidate_{field}"] = parse_number(row[f"candidate_{field}"], f"{complex_id}:candidate_{field}", 0, 1)
        for field in rmsd_fields:
            item[f"baseline_{field}"] = parse_number(row[f"baseline_{field}"], f"{complex_id}:baseline_{field}")
            item[f"candidate_{field}"] = parse_number(row[f"candidate_{field}"], f"{complex_id}:candidate_{field}")
        for field in failure_fields:
            item[f"baseline_{field}"] = parse_bool(row[f"baseline_{field}"], f"{complex_id}:baseline_{field}")
            item[f"candidate_{field}"] = parse_bool(row[f"candidate_{field}"], f"{complex_id}:candidate_{field}")
        parsed.append(item)
    if seen != set(set_by_id):
        missing = sorted(set(set_by_id) - seen)[:5]
        raise ValueError(f"metrics do not cover the complete structural set; missing {missing}")

    continuous = {}
    for field in unit_interval_fields + rmsd_fields:
        delta = mean_delta(parsed, field)
        continuous[field] = {
            "baseline_mean": statistics.fmean(row[f"baseline_{field}"] for row in parsed),
            "candidate_mean": statistics.fmean(row[f"candidate_{field}"] for row in parsed),
            "candidate_minus_baseline_mean": delta,
            "paired_group_bootstrap_delta": bootstrap_delta(
                parsed, lambda values, key=field: mean_delta(values, key), bootstrap_replicates, bootstrap_seed
            ),
        }
    acceptable = {
        "threshold": ACCEPTABLE_DOCKQ,
        "baseline_fraction": statistics.fmean(row["baseline_dockq"] >= ACCEPTABLE_DOCKQ for row in parsed),
        "candidate_fraction": statistics.fmean(row["candidate_dockq"] >= ACCEPTABLE_DOCKQ for row in parsed),
        "candidate_minus_baseline_fraction": acceptable_delta(parsed),
        "paired_group_bootstrap_delta": bootstrap_delta(
            parsed, acceptable_delta, bootstrap_replicates, bootstrap_seed
        ),
    }
    new_failures = {
        field: sum(row[f"candidate_{field}"] and not row[f"baseline_{field}"] for row in parsed)
        for field in failure_fields
    }
    dockq_ci = continuous["dockq"]["paired_group_bootstrap_delta"]
    acceptable_ci = acceptable["paired_group_bootstrap_delta"]
    gate_checks = {
        "dockq_point_delta_at_least_minus_0_05": continuous["dockq"]["candidate_minus_baseline_mean"] >= DOCKQ_NONINFERIORITY_MARGIN,
        "dockq_lower_95_at_least_minus_0_05": dockq_ci["lower_95"] >= DOCKQ_NONINFERIORITY_MARGIN,
        "acceptable_fraction_point_delta_at_least_minus_0_10": acceptable["candidate_minus_baseline_fraction"] >= ACCEPTABLE_FRACTION_MARGIN,
        "acceptable_fraction_lower_95_at_least_minus_0_10": acceptable_ci["lower_95"] >= ACCEPTABLE_FRACTION_MARGIN,
        "no_new_severe_geometry_failures": sum(new_failures.values()) == 0,
    }
    return {
        "schema_version": "1.0",
        "complex_count": len(parsed),
        "input_sha256": {
            "structural_set": sha256_file(structural_set_path),
            "metrics": sha256_file(metrics_path),
        },
        "thresholds": {
            "dockq_noninferiority_margin": DOCKQ_NONINFERIORITY_MARGIN,
            "acceptable_dockq": ACCEPTABLE_DOCKQ,
            "acceptable_fraction_margin": ACCEPTABLE_FRACTION_MARGIN,
            "minimum_complexes": min_complexes,
        },
        "metrics": continuous,
        "acceptable_complexes": acceptable,
        "new_severe_geometry_failures": new_failures,
        "gate_checks": gate_checks,
        "structural_preservation_passed": all(gate_checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-set", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.structural_set, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["structural_preservation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
