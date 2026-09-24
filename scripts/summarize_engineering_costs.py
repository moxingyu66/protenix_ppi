#!/usr/bin/env python3
"""Validate and summarize per-pair and per-run engineering cost ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from protenix_ppi.scripts.evaluate_predictions import percentile
from protenix_ppi.scripts.pair_feature_bundle import sha256_file


REGISTERED_SEEDS = {42, 123, 999}
PAIR_FIELDS = {
    "method",
    "seed",
    "stage",
    "pair_id",
    "partition",
    "status",
    "gpu_count",
    "gpu_model",
    "peak_gpu_memory_mib",
    "wall_time_seconds",
    "preprocessing_seconds",
    "output_bytes",
    "input_residues",
    "input_sha256",
    "command_sha256",
    "stdout_sha256",
}
RUN_FIELDS = {
    "method",
    "seed",
    "stage",
    "status",
    "gpu_count",
    "gpu_model",
    "peak_gpu_memory_mib",
    "wall_time_seconds",
    "output_bytes",
    "command_sha256",
    "stdout_sha256",
}
HEX = set("0123456789abcdef")


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


def parse_float(value: str, location: str, minimum: float = 0.0, allow_blank: bool = False) -> float | None:
    if allow_blank and not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{location}: invalid number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{location}: number must be finite and >= {minimum}")
    return result


def parse_int(value: str, location: str, minimum: int = 0) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{location}: invalid integer") from exc
    if result < minimum:
        raise ValueError(f"{location}: integer must be >= {minimum}")
    return result


def validate_digest(value: str, location: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in HEX for character in digest):
        raise ValueError(f"{location}: invalid SHA-256")
    return digest


def expected_pair_ids(splits_path: Path, partition: str) -> set[str]:
    rows = read_csv(splits_path, {"split_scheme", "fold", "pair_id", "partition"})
    expected = {
        row["pair_id"].strip()
        for row in rows
        if row["split_scheme"].strip() == "c3_primary"
        and row["fold"].strip() == "0"
        and row["partition"].strip() == partition
    }
    if not expected:
        raise ValueError(f"splits.csv contains no c3_primary fold-0 {partition} pairs")
    return expected


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0, "total": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "total": sum(values),
    }


def parse_pair_rows(path: Path, partition: str) -> list[dict]:
    rows = read_csv(path, PAIR_FIELDS)
    parsed = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        if row["partition"].strip() != partition:
            continue
        method = row["method"].strip()
        stage = row["stage"].strip()
        pair_id = row["pair_id"].strip()
        if not method or not stage or not pair_id:
            raise ValueError(f"{path.name}:{line}: method, stage, and pair_id are required")
        seed = parse_int(row["seed"], f"{path.name}:{line}:seed")
        key = (method, seed, stage, pair_id)
        if key in seen:
            raise ValueError(f"{path.name}:{line}: duplicate cost row {key}")
        seen.add(key)
        status = row["status"].strip().lower()
        if status not in {"success", "failure"}:
            raise ValueError(f"{path.name}:{line}: status must be success or failure")
        gpu_count = parse_int(row["gpu_count"], f"{path.name}:{line}:gpu_count")
        gpu_model = row["gpu_model"].strip()
        peak_memory = parse_float(
            row["peak_gpu_memory_mib"], f"{path.name}:{line}:peak_gpu_memory_mib", allow_blank=gpu_count == 0
        )
        if gpu_count > 0 and (not gpu_model or peak_memory is None or peak_memory <= 0):
            raise ValueError(f"{path.name}:{line}: GPU rows require model and positive peak memory")
        parsed.append(
            {
                "method": method,
                "seed": seed,
                "stage": stage,
                "pair_id": pair_id,
                "status": status,
                "gpu_count": gpu_count,
                "gpu_model": gpu_model,
                "peak_gpu_memory_mib": peak_memory,
                "wall_time_seconds": parse_float(row["wall_time_seconds"], f"{path.name}:{line}:wall_time_seconds", 0.000001),
                "preprocessing_seconds": parse_float(row["preprocessing_seconds"], f"{path.name}:{line}:preprocessing_seconds"),
                "output_bytes": parse_int(row["output_bytes"], f"{path.name}:{line}:output_bytes"),
                "input_residues": parse_int(row["input_residues"], f"{path.name}:{line}:input_residues", 1),
                "input_sha256": validate_digest(row["input_sha256"], f"{path.name}:{line}:input_sha256"),
                "command_sha256": validate_digest(row["command_sha256"], f"{path.name}:{line}:command_sha256"),
                "stdout_sha256": validate_digest(row["stdout_sha256"], f"{path.name}:{line}:stdout_sha256"),
            }
        )
    if not parsed:
        raise ValueError(f"{path.name} has no rows for partition {partition}")
    return parsed


def parse_run_rows(path: Path) -> list[dict]:
    rows = read_csv(path, RUN_FIELDS)
    parsed = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        method = row["method"].strip()
        stage = row["stage"].strip()
        seed_text = row["seed"].strip()
        if not method or not stage:
            raise ValueError(f"{path.name}:{line}: method and stage are required")
        seed = None if seed_text.upper() == "NA" else parse_int(seed_text, f"{path.name}:{line}:seed")
        key = (method, seed, stage)
        if key in seen:
            raise ValueError(f"{path.name}:{line}: duplicate run cost row {key}")
        seen.add(key)
        status = row["status"].strip().lower()
        if status not in {"success", "failure"}:
            raise ValueError(f"{path.name}:{line}: status must be success or failure")
        gpu_count = parse_int(row["gpu_count"], f"{path.name}:{line}:gpu_count")
        gpu_model = row["gpu_model"].strip()
        peak_memory = parse_float(
            row["peak_gpu_memory_mib"], f"{path.name}:{line}:peak_gpu_memory_mib", allow_blank=gpu_count == 0
        )
        if gpu_count > 0 and (not gpu_model or peak_memory is None or peak_memory <= 0):
            raise ValueError(f"{path.name}:{line}: GPU rows require model and positive peak memory")
        parsed.append(
            {
                "method": method,
                "seed": seed,
                "stage": stage,
                "status": status,
                "gpu_count": gpu_count,
                "gpu_model": gpu_model,
                "peak_gpu_memory_mib": peak_memory,
                "wall_time_seconds": parse_float(row["wall_time_seconds"], f"{path.name}:{line}:wall_time_seconds", 0.000001),
                "output_bytes": parse_int(row["output_bytes"], f"{path.name}:{line}:output_bytes"),
                "command_sha256": validate_digest(row["command_sha256"], f"{path.name}:{line}:command_sha256"),
                "stdout_sha256": validate_digest(row["stdout_sha256"], f"{path.name}:{line}:stdout_sha256"),
            }
        )
    return parsed


def summarize_pair_group(rows: list[dict], expected: set[str]) -> dict:
    attempted = {row["pair_id"] for row in rows}
    successful = {row["pair_id"] for row in rows if row["status"] == "success"}
    wall = [float(row["wall_time_seconds"]) for row in rows]
    preprocessing = [float(row["preprocessing_seconds"]) for row in rows]
    memory = [float(row["peak_gpu_memory_mib"]) for row in rows if row["peak_gpu_memory_mib"] is not None]
    total_gpu_seconds = sum(float(row["wall_time_seconds"]) * int(row["gpu_count"]) for row in rows)
    successful_wall = sum(float(row["wall_time_seconds"]) for row in rows if row["status"] == "success")
    return {
        "attempted_pairs": len(attempted),
        "successful_pairs": len(successful),
        "failed_pairs": len(attempted - successful),
        "expected_pairs": len(expected),
        "attempt_coverage_complete": attempted == expected,
        "successful_prediction_coverage_complete": successful == expected,
        "missing_attempt_pair_ids": sorted(expected - attempted)[:20],
        "failed_pair_ids": sorted(attempted - successful)[:20],
        "unexpected_pair_ids": sorted(attempted - expected)[:20],
        "wall_time_seconds": distribution(wall),
        "preprocessing_seconds": distribution(preprocessing),
        "peak_gpu_memory_mib": distribution(memory),
        "total_gpu_hours": total_gpu_seconds / 3600,
        "total_output_bytes": sum(int(row["output_bytes"]) for row in rows),
        "successful_pairs_per_wall_hour": len(successful) / (successful_wall / 3600) if successful_wall > 0 else None,
        "gpu_models": sorted({row["gpu_model"] for row in rows if row["gpu_model"]}),
        "input_residue_range": [min(row["input_residues"] for row in rows), max(row["input_residues"] for row in rows)],
    }


def summarize(pair_costs_path: Path, run_costs_path: Path, splits_path: Path, partition: str = "test") -> dict:
    expected = expected_pair_ids(splits_path, partition)
    pair_rows = parse_pair_rows(pair_costs_path, partition)
    run_rows = parse_run_rows(run_costs_path)
    pair_groups: defaultdict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in pair_rows:
        pair_groups[(row["method"], row["seed"], row["stage"])].append(row)
    method_stage_seeds: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    pair_summary = {}
    for (method, seed, stage), rows in sorted(pair_groups.items()):
        method_stage_seeds[(method, stage)].add(seed)
        pair_summary[f"{method}|seed={seed}|stage={stage}"] = summarize_pair_group(rows, expected)
    seed_coverage = {
        f"{method}|stage={stage}": {
            "observed": sorted(seeds),
            "registered_complete": seeds == REGISTERED_SEEDS,
        }
        for (method, stage), seeds in sorted(method_stage_seeds.items())
    }

    run_groups: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        run_groups[(row["method"], row["stage"])].append(row)
    run_summary = {}
    for (method, stage), rows in sorted(run_groups.items()):
        wall = [float(row["wall_time_seconds"]) for row in rows]
        memory = [float(row["peak_gpu_memory_mib"]) for row in rows if row["peak_gpu_memory_mib"] is not None]
        observed_run_seeds = {row["seed"] for row in rows if row["seed"] is not None}
        run_summary[f"{method}|stage={stage}"] = {
            "runs": len(rows),
            "successful_runs": sum(row["status"] == "success" for row in rows),
            "failed_runs": sum(row["status"] == "failure" for row in rows),
            "seeds": sorted(observed_run_seeds),
            "registered_seed_coverage_complete": not observed_run_seeds or observed_run_seeds == REGISTERED_SEEDS,
            "wall_time_seconds": distribution(wall),
            "peak_gpu_memory_mib": distribution(memory),
            "total_gpu_hours": sum(float(row["wall_time_seconds"]) * int(row["gpu_count"]) for row in rows) / 3600,
            "total_output_bytes": sum(int(row["output_bytes"]) for row in rows),
            "gpu_models": sorted({row["gpu_model"] for row in rows if row["gpu_model"]}),
        }
    coverage_complete = all(item["attempt_coverage_complete"] for item in pair_summary.values())
    execution_complete = coverage_complete and all(
        item["successful_prediction_coverage_complete"] for item in pair_summary.values()
    ) and all(item["failed_runs"] == 0 for item in run_summary.values())
    registered_seed_complete = all(item["registered_complete"] for item in seed_coverage.values()) and all(
        item["registered_seed_coverage_complete"] for item in run_summary.values()
    )
    return {
        "schema_version": "1.0",
        "partition": partition,
        "expected_c3_pair_count": len(expected),
        "input_sha256": {
            "pair_costs": sha256_file(pair_costs_path),
            "run_costs": sha256_file(run_costs_path),
            "splits": sha256_file(splits_path),
        },
        "pair_costs": pair_summary,
        "registered_seed_coverage": seed_coverage,
        "run_costs": run_summary,
        "attempt_coverage_complete": coverage_complete,
        "registered_seed_coverage_complete": registered_seed_complete,
        "experiment_execution_complete": execution_complete and registered_seed_complete,
        "interpretation_limits": [
            "GPU-hours are wall time multiplied by allocated GPU count; they are not monetary cost.",
            "Per-pair output_bytes must count newly attributable files and must not repeatedly count a shared cache.",
            "Linear throughput projections do not include scheduler queue time and are valid only for the recorded hardware and length distribution.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-costs", type=Path, required=True)
    parser.add_argument("--run-costs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.pair_costs, args.run_costs, args.splits, args.partition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["experiment_execution_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
