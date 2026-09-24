#!/usr/bin/env python3
"""Convert pinned Protenix B1 outputs into traceable test predictions.

This adapter deliberately does not import Protenix.  It validates immutable
inputs and consumes the JSON/mmCIF files produced by a pinned server run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.validate_dataset import validate_tables


PRIMARY_MODEL = "protenix_base_default_v1.0.0"
PRIMARY_CUTOFF = "2021-09-30"
PRIMARY_SCORE = "iptm"
COMPARATOR_SCORE = "ranking_score"
PREDICTION_FIELDS = [
    "pair_id",
    "label",
    "score",
    "partition",
    "evidence_class",
    "bootstrap_group",
]
RUN_FIELDS = {
    "pair_id",
    "sample_name",
    "input_json",
    "input_sha256",
    "predictions_dir",
}
LOCK_HEX = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _require_bool(mapping: dict[str, Any], key: str, expected: bool) -> None:
    if mapping.get(key) is not expected:
        raise ValueError(f"condition lock requires {key}={str(expected).lower()}")


def validate_backbone_lock(path: Path) -> dict[str, Any]:
    lock = load_json(path)
    if not isinstance(lock, dict):
        raise ValueError("backbone lock must be a JSON object")
    if lock.get("repository_url") != "https://github.com/bytedance/Protenix.git":
        raise ValueError("backbone lock must reference the official Protenix repository")
    if lock.get("model_name") != PRIMARY_MODEL:
        raise ValueError(f"B1 primary model must be {PRIMARY_MODEL}")
    if lock.get("declared_training_cutoff") != PRIMARY_CUTOFF:
        raise ValueError(f"B1 primary cutoff must be {PRIMARY_CUTOFF}")
    if not LOCK_HEX.fullmatch(str(lock.get("source_commit", ""))):
        raise ValueError("backbone lock source_commit must be a 40-character Git commit")
    if not SHA256_HEX.fullmatch(str(lock.get("checkpoint_sha256", ""))):
        raise ValueError("backbone lock checkpoint_sha256 must be a SHA-256 digest")
    if lock.get("g0_route") not in {"A", "B", "C"}:
        raise ValueError("scientific B1 requires a G0 route of A, B, or C")
    for key in (
        "package_version",
        "python_version",
        "torch_version",
        "cuda_runtime",
        "nvidia_driver",
        "triangle_attention_kernel",
        "triangle_multiplicative_kernel",
    ):
        value = str(lock.get(key, "")).strip()
        if not value or value.startswith("REPLACE"):
            raise ValueError(f"backbone lock field is missing: {key}")
    return lock


def validate_condition_lock(path: Path, backbone_lock: dict[str, Any]) -> dict[str, Any]:
    condition = load_json(path)
    if not isinstance(condition, dict):
        raise ValueError("condition lock must be a JSON object")
    if condition.get("model_name") != backbone_lock["model_name"]:
        raise ValueError("condition/backbone model_name mismatch")
    if not isinstance(condition.get("seed"), int):
        raise ValueError("condition lock seed must be an integer")
    expected_numbers = {"n_cycle": 10, "n_step": 200, "n_sample": 5}
    for key, expected in expected_numbers.items():
        if condition.get(key) != expected:
            raise ValueError(f"B1 primary condition requires {key}={expected}")
    _require_bool(condition, "use_msa", True)
    _require_bool(condition, "use_template", False)
    _require_bool(condition, "msa_pair_as_unpair", True)
    _require_bool(condition, "sorted_by_ranking_score", True)
    _require_bool(condition, "use_default_params", True)
    _require_bool(condition, "enable_tf32", True)
    _require_bool(condition, "enable_efficient_fusion", True)
    _require_bool(condition, "enable_diffusion_shared_vars_cache", True)
    if condition.get("dtype") != "bf16":
        raise ValueError("B1 primary condition requires dtype=bf16")
    if condition.get("sample_selection") != "native_rank_0":
        raise ValueError("B1 sample_selection must be native_rank_0")
    if condition.get("primary_score_field") != PRIMARY_SCORE:
        raise ValueError(f"B1 primary_score_field must be {PRIMARY_SCORE}")
    if condition.get("comparator_score_fields") != [COMPARATOR_SCORE]:
        raise ValueError(f"B1 comparator_score_fields must be [{COMPARATOR_SCORE!r}]")
    if condition.get("label_fitted_combination") is not False:
        raise ValueError("B1 forbids label-fitted score combinations")
    return condition


def select_input_sample(input_path: Path, sample_name: str) -> dict[str, Any]:
    payload = load_json(input_path)
    samples = payload if isinstance(payload, list) else [payload]
    matches = [item for item in samples if isinstance(item, dict) and item.get("name") == sample_name]
    if len(matches) != 1:
        raise ValueError(f"{input_path}: expected exactly one sample named {sample_name!r}")
    return matches[0]


def validate_two_protein_chains(
    sample: dict[str, Any], expected_sequences: tuple[str, str], input_path: Path
) -> None:
    entities = sample.get("sequences")
    if not isinstance(entities, list):
        raise ValueError(f"{input_path}: sample sequences must be a list")
    observed: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {"proteinChain"}:
            raise ValueError(f"{input_path}: B1 primary input may contain proteinChain entities only")
        chain = entity["proteinChain"]
        if not isinstance(chain, dict):
            raise ValueError(f"{input_path}: proteinChain must be an object")
        count = chain.get("count", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{input_path}: proteinChain count must be a positive integer")
        sequence = chain.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"{input_path}: proteinChain sequence is missing")
        observed.extend([sequence.strip().upper()] * count)
    if len(observed) != 2:
        raise ValueError(f"{input_path}: B1 primary input must contain exactly two protein chains")
    if sorted(observed) != sorted(sequence.strip().upper() for sequence in expected_sequences):
        raise ValueError(f"{input_path}: protein-chain sequences do not match the frozen benchmark pair")


def _finite_number(summary: dict[str, Any], key: str, path: Path) -> float:
    value = summary.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{path}: {key} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {key} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: {key} must be finite")
    return number


def _validate_two_chain_confidence(summary: dict[str, Any], path: Path) -> None:
    matrix = summary.get("chain_pair_iptm")
    if not (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        raise ValueError(f"{path}: chain_pair_iptm must be a 2x2 matrix")
    chain_scores = summary.get("chain_iptm")
    if not isinstance(chain_scores, list) or len(chain_scores) != 2:
        raise ValueError(f"{path}: chain_iptm must contain two chain scores")


def load_ranked_summaries(
    predictions_dir: Path, sample_name: str, expected_count: int
) -> list[tuple[int, Path, dict[str, Any]]]:
    pattern = re.compile(
        rf"^{re.escape(sample_name)}_summary_confidence_sample_(\d+)\.json$"
    )
    found: dict[int, tuple[Path, dict[str, Any]]] = {}
    if not predictions_dir.is_dir():
        raise ValueError(f"predictions directory does not exist: {predictions_dir}")
    for path in predictions_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        rank = int(match.group(1))
        if rank in found:
            raise ValueError(f"{predictions_dir}: duplicate confidence rank {rank}")
        summary = load_json(path)
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: summary confidence must be a JSON object")
        iptm = _finite_number(summary, PRIMARY_SCORE, path)
        _finite_number(summary, COMPARATOR_SCORE, path)
        _finite_number(summary, "ptm", path)
        _finite_number(summary, "plddt", path)
        if not 0.0 <= iptm <= 1.0:
            raise ValueError(f"{path}: iptm must lie in [0,1]")
        _validate_two_chain_confidence(summary, path)
        found[rank] = (path, summary)
    expected_ranks = set(range(expected_count))
    if set(found) != expected_ranks:
        raise ValueError(
            f"{predictions_dir}: expected confidence ranks {sorted(expected_ranks)}, "
            f"observed {sorted(found)}"
        )
    ranked = [(rank, *found[rank]) for rank in sorted(found)]
    rank_zero_score = _finite_number(ranked[0][2], COMPARATOR_SCORE, ranked[0][1])
    best_score = max(_finite_number(summary, COMPARATOR_SCORE, path) for _, path, summary in ranked)
    if rank_zero_score != best_score:
        raise ValueError(f"{predictions_dir}: rank 0 is not the highest native ranking_score")
    return ranked


def _write_predictions(path: Path, rows: list[dict[str, str]], score_key: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in PREDICTION_FIELDS}
            output["score"] = row[score_key]
            writer.writerow(output)


def build_predictions(
    proteins_path: Path,
    pairs_path: Path,
    evidence_path: Path,
    splits_path: Path,
    runs_path: Path,
    backbone_lock_path: Path,
    condition_lock_path: Path,
    output_dir: Path,
    split_scheme: str,
    fold: str,
) -> dict[str, Any]:
    validation = validate_tables(proteins_path, pairs_path, evidence_path, splits_path)
    if validation.errors:
        raise ValueError("dataset validation failed:\n- " + "\n- ".join(validation.errors))
    backbone = validate_backbone_lock(backbone_lock_path)
    condition = validate_condition_lock(condition_lock_path, backbone)

    proteins = {row["uniprot_accession"].strip(): row for row in read_csv(proteins_path)}
    pairs = {row["pair_id"].strip(): row for row in read_csv(pairs_path)}
    test_ids = {
        row["pair_id"].strip()
        for row in read_csv(splits_path)
        if row["split_scheme"].strip() == split_scheme
        and row["fold"].strip() == fold
        and row["partition"].strip() == "test"
    }
    if not test_ids:
        raise ValueError(f"no test pairs found for {split_scheme!r} fold {fold!r}")

    with runs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = RUN_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"run manifest is missing columns: {', '.join(sorted(missing))}")
        run_rows = list(reader)
    run_ids = [row["pair_id"].strip() for row in run_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run manifest contains duplicate pair_id values")
    if set(run_ids) != test_ids:
        raise ValueError(
            "run manifest must cover the frozen test set exactly; "
            f"missing={sorted(test_ids - set(run_ids))[:5]}, extra={sorted(set(run_ids) - test_ids)[:5]}"
        )

    prediction_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    backbone_digest = sha256_file(backbone_lock_path)
    condition_digest = sha256_file(condition_lock_path)
    for run in sorted(run_rows, key=lambda item: item["pair_id"].strip()):
        pair_id = run["pair_id"].strip()
        pair = pairs[pair_id]
        if pair["pair_status"].strip() != "eligible" or pair["label"].strip() not in {"0", "1"}:
            raise ValueError(f"test pair {pair_id} is not eligible and labeled")
        input_path = resolve_manifest_path(run["input_json"], runs_path)
        predictions_dir = resolve_manifest_path(run["predictions_dir"], runs_path)
        expected_input_hash = run["input_sha256"].strip().lower()
        if not SHA256_HEX.fullmatch(expected_input_hash):
            raise ValueError(f"run manifest {pair_id}: input_sha256 is invalid")
        actual_input_hash = sha256_file(input_path)
        if actual_input_hash != expected_input_hash:
            raise ValueError(f"run manifest {pair_id}: input SHA-256 mismatch")
        sample_name = run["sample_name"].strip()
        if sample_name != pair_id:
            raise ValueError(f"run manifest {pair_id}: sample_name must equal pair_id")
        sample = select_input_sample(input_path, sample_name)
        expected_sequences = (
            proteins[pair["uniprot_a"].strip()]["sequence"],
            proteins[pair["uniprot_b"].strip()]["sequence"],
        )
        validate_two_protein_chains(sample, expected_sequences, input_path)
        ranked = load_ranked_summaries(predictions_dir, sample_name, condition["n_sample"])
        _, summary_path, summary = ranked[0]
        structure_path = predictions_dir / f"{sample_name}_sample_0.cif"
        if not structure_path.is_file() or structure_path.stat().st_size == 0:
            raise ValueError(f"missing or empty rank-0 structure: {structure_path}")
        iptm = _finite_number(summary, PRIMARY_SCORE, summary_path)
        ranking_score = _finite_number(summary, COMPARATOR_SCORE, summary_path)
        ptm = _finite_number(summary, "ptm", summary_path)
        plddt = _finite_number(summary, "plddt", summary_path)
        base_row = {
            "pair_id": pair_id,
            "label": pair["label"].strip(),
            "score": "",
            "partition": "test",
            "evidence_class": pair["evidence_class"].strip(),
            "bootstrap_group": pair["complex_group_id"].strip() or pair_id,
            PRIMARY_SCORE: format(iptm, ".17g"),
            COMPARATOR_SCORE: format(ranking_score, ".17g"),
        }
        prediction_rows.append(base_row)
        provenance_rows.append(
            {
                "pair_id": pair_id,
                "sample_name": sample_name,
                "seed": str(condition["seed"]),
                "selected_rank": "0",
                "iptm": base_row[PRIMARY_SCORE],
                "ranking_score": base_row[COMPARATOR_SCORE],
                "ptm": format(ptm, ".17g"),
                "plddt": format(plddt, ".17g"),
                "has_clash": str(summary.get("has_clash", "")),
                "num_recycles": str(summary.get("num_recycles", "")),
                "input_path": str(input_path),
                "input_sha256": actual_input_hash,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path),
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": sha256_file(structure_path),
                "backbone_lock_sha256": backbone_digest,
                "condition_lock_sha256": condition_digest,
                "source_commit": str(backbone["source_commit"]),
                "checkpoint_sha256": str(backbone["checkpoint_sha256"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = output_dir / "predictions.csv"
    comparator_path = output_dir / "predictions_ranking_score.csv"
    _write_predictions(primary_path, prediction_rows, PRIMARY_SCORE)
    _write_predictions(comparator_path, prediction_rows, COMPARATOR_SCORE)
    provenance_path = output_dir / "source_manifest.csv"
    provenance_fields = list(provenance_rows[0])
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=provenance_fields)
        writer.writeheader()
        writer.writerows(provenance_rows)

    metadata = {
        "method": "B1_Protenix_zero_shot",
        "protocol_version": "b1_zero_shot_protocol_v0.1",
        "model_name": backbone["model_name"],
        "declared_training_cutoff": backbone["declared_training_cutoff"],
        "source_commit": backbone["source_commit"],
        "checkpoint_sha256": backbone["checkpoint_sha256"],
        "seed": condition["seed"],
        "split_scheme": split_scheme,
        "fold": fold,
        "test_pair_count": len(prediction_rows),
        "primary_score": PRIMARY_SCORE,
        "comparator_score": COMPARATOR_SCORE,
        "sample_selection": "native_rank_0",
        "label_fitted_combination": False,
        "condition": condition,
        "input_sha256": {
            "proteins": sha256_file(proteins_path),
            "pairs": sha256_file(pairs_path),
            "evidence": sha256_file(evidence_path),
            "splits": sha256_file(splits_path),
            "runs": sha256_file(runs_path),
            "backbone_lock": backbone_digest,
            "condition_lock": condition_digest,
        },
        "output_sha256": {
            "predictions": sha256_file(primary_path),
            "predictions_ranking_score": sha256_file(comparator_path),
            "source_manifest": sha256_file(provenance_path),
        },
        "interpretation_warning": (
            "B1 is an uncalibrated native-confidence baseline, not a fitted probability model. "
            "No biological performance is established until real frozen-test outputs are evaluated."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--backbone-lock", type=Path, required=True)
    parser.add_argument("--condition-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-scheme", default="c3_primary")
    parser.add_argument("--fold", default="0")
    args = parser.parse_args()
    metadata = build_predictions(
        args.proteins,
        args.pairs,
        args.evidence,
        args.splits,
        args.runs,
        args.backbone_lock,
        args.condition_lock,
        args.output_dir,
        args.split_scheme,
        args.fold,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
