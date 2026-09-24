#!/usr/bin/env python3
"""Generate a deterministic homology-aware C3 split for labeled PPI pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


ALGORITHM_VERSION = "c3_hash_cluster_v0.2"
PARTITIONS = ("train", "validation", "test")
NODE_TARGETS = {"train": 0.60, "validation": 0.20, "test": 0.20}
PAIR_TARGETS = {"train": 0.80, "validation": 0.10, "test": 0.10}


@dataclass(frozen=True)
class CandidateStats:
    seed: int
    score: float
    retained_pairs: int
    quarantined_pairs: int
    retention_fraction: float
    pair_counts: dict[str, int]
    positive_counts: dict[str, int]
    negative_counts: dict[str, int]
    protein_counts: dict[str, int]
    cluster_counts: dict[str, int]
    prevalence: dict[str, float | None]
    quarantine_reasons: dict[str, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unit_interval(seed: int, cluster_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{cluster_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_clusters(cluster_ids: set[str], seed: int) -> dict[str, str]:
    assignment: dict[str, str] = {}
    train_cutoff = NODE_TARGETS["train"]
    validation_cutoff = train_cutoff + NODE_TARGETS["validation"]
    for cluster_id in sorted(cluster_ids):
        value = stable_unit_interval(seed, cluster_id)
        if value < train_cutoff:
            partition = "train"
        elif value < validation_cutoff:
            partition = "validation"
        else:
            partition = "test"
        assignment[cluster_id] = partition
    return assignment


def eligible_labeled_pairs(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in pairs
        if row.get("pair_status", "").strip() == "eligible" and row.get("label", "").strip() in {"0", "1"}
    ]


def pair_retention_decisions(
    pairs: list[dict[str, str]],
    protein_to_cluster: dict[str, str],
    cluster_assignment: dict[str, str],
) -> dict[str, tuple[str | None, str]]:
    """Assign retained pairs and quarantine cross-pool evidence groups atomically."""
    pair_pools: dict[str, tuple[str, str]] = {}
    group_pools: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        pair_id = pair["pair_id"].strip()
        pool_a = cluster_assignment[protein_to_cluster[pair["uniprot_a"].strip()]]
        pool_b = cluster_assignment[protein_to_cluster[pair["uniprot_b"].strip()]]
        pair_pools[pair_id] = (pool_a, pool_b)
        group_id = pair.get("complex_group_id", "").strip()
        if group_id:
            group_pools[group_id].update((pool_a, pool_b))

    decisions: dict[str, tuple[str | None, str]] = {}
    for pair in pairs:
        pair_id = pair["pair_id"].strip()
        pool_a, pool_b = pair_pools[pair_id]
        group_id = pair.get("complex_group_id", "").strip()
        if group_id and len(group_pools[group_id]) != 1:
            decisions[pair_id] = (None, "complex_group_cross_partition_quarantine")
        elif pool_a != pool_b:
            decisions[pair_id] = (None, "cross_partition_quarantine")
        else:
            decisions[pair_id] = (pool_a, "")
    return decisions


def evaluate_candidate(
    seed: int,
    proteins: list[dict[str, str]],
    pairs: list[dict[str, str]],
    protein_to_cluster: dict[str, str],
    min_per_class: int,
) -> tuple[CandidateStats | None, dict[str, str]]:
    cluster_assignment = assign_clusters(set(protein_to_cluster.values()), seed)
    decisions = pair_retention_decisions(pairs, protein_to_cluster, cluster_assignment)
    pair_counts = Counter({partition: 0 for partition in PARTITIONS})
    positive_counts = Counter({partition: 0 for partition in PARTITIONS})
    negative_counts = Counter({partition: 0 for partition in PARTITIONS})
    protein_counts = Counter({partition: 0 for partition in PARTITIONS})
    cluster_counts = Counter(cluster_assignment.values())

    for protein in proteins:
        accession = protein["uniprot_accession"].strip()
        cluster = protein_to_cluster[accession]
        protein_counts[cluster_assignment[cluster]] += 1

    quarantine_reasons: Counter[str] = Counter()
    for pair in pairs:
        partition, reason = decisions[pair["pair_id"].strip()]
        if partition is None:
            quarantine_reasons[reason] += 1
            continue
        pair_counts[partition] += 1
        if pair["label"].strip() == "1":
            positive_counts[partition] += 1
        else:
            negative_counts[partition] += 1

    if any(positive_counts[p] < min_per_class or negative_counts[p] < min_per_class for p in PARTITIONS):
        return None, cluster_assignment

    retained = sum(pair_counts.values())
    if retained == 0:
        return None, cluster_assignment
    prevalence = {
        p: positive_counts[p] / pair_counts[p] if pair_counts[p] else None for p in PARTITIONS
    }
    global_prevalence = sum(positive_counts.values()) / retained
    protein_total = len(proteins)
    labeled_total = len(pairs)

    pair_deviation = sum(abs(pair_counts[p] / retained - PAIR_TARGETS[p]) for p in PARTITIONS)
    prevalence_deviation = sum(abs(float(prevalence[p]) - global_prevalence) for p in PARTITIONS)
    node_deviation = sum(abs(protein_counts[p] / protein_total - NODE_TARGETS[p]) for p in PARTITIONS)
    retention_fraction = retained / labeled_total if labeled_total else 0.0
    score = pair_deviation + prevalence_deviation + 0.5 * node_deviation - 0.1 * retention_fraction

    return (
        CandidateStats(
            seed=seed,
            score=round(score, 12),
            retained_pairs=retained,
            quarantined_pairs=sum(quarantine_reasons.values()),
            retention_fraction=retention_fraction,
            pair_counts=dict(pair_counts),
            positive_counts=dict(positive_counts),
            negative_counts=dict(negative_counts),
            protein_counts=dict(protein_counts),
            cluster_counts=dict(cluster_counts),
            prevalence=prevalence,
            quarantine_reasons=dict(sorted(quarantine_reasons.items())),
        ),
        cluster_assignment,
    )


def select_candidate(
    proteins: list[dict[str, str]],
    pairs: list[dict[str, str]],
    protein_to_cluster: dict[str, str],
    seed_start: int,
    seed_end: int,
    min_per_class: int,
) -> tuple[CandidateStats, dict[str, str]]:
    candidates: list[tuple[CandidateStats, dict[str, str]]] = []
    for seed in range(seed_start, seed_end + 1):
        stats, assignment = evaluate_candidate(seed, proteins, pairs, protein_to_cluster, min_per_class)
        if stats is not None:
            candidates.append((stats, assignment))
    if not candidates:
        raise ValueError(
            "No candidate split satisfies the per-partition class minimum; expand the dataset or explicitly revise the protocol."
        )
    return min(candidates, key=lambda item: (item[0].score, item[0].seed))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(
    proteins_path: Path,
    pairs_path: Path,
    output_dir: Path,
    seed_start: int = 0,
    seed_end: int = 999,
    min_per_class: int = 1,
) -> CandidateStats:
    proteins = read_csv(proteins_path)
    all_pairs = read_csv(pairs_path)
    pairs = eligible_labeled_pairs(all_pairs)
    if not proteins:
        raise ValueError("proteins.csv is empty")
    if not pairs:
        raise ValueError("pairs.csv has no eligible labeled pairs")

    protein_to_cluster: dict[str, str] = {}
    for row in proteins:
        accession = row.get("uniprot_accession", "").strip()
        cluster = row.get("homology_cluster_30", "").strip()
        if not accession or not cluster:
            raise ValueError("Every protein requires uniprot_accession and homology_cluster_30")
        if accession in protein_to_cluster:
            raise ValueError(f"Duplicate protein accession: {accession}")
        protein_to_cluster[accession] = cluster
    for pair in pairs:
        for endpoint in (pair["uniprot_a"].strip(), pair["uniprot_b"].strip()):
            if endpoint not in protein_to_cluster:
                raise ValueError(f"Pair references missing protein: {endpoint}")

    stats, assignment = select_candidate(
        proteins, pairs, protein_to_cluster, seed_start, seed_end, min_per_class
    )
    decisions = pair_retention_decisions(pairs, protein_to_cluster, assignment)
    split_rows: list[dict[str, str]] = []
    quarantine_rows: list[dict[str, str]] = []
    for pair in sorted(pairs, key=lambda row: row["pair_id"]):
        cluster_a = protein_to_cluster[pair["uniprot_a"].strip()]
        cluster_b = protein_to_cluster[pair["uniprot_b"].strip()]
        partition_a = assignment[cluster_a]
        partition_b = assignment[cluster_b]
        partition, reason = decisions[pair["pair_id"].strip()]
        if partition is not None:
            split_rows.append(
                {
                    "split_scheme": "c3_primary",
                    "fold": "0",
                    "pair_id": pair["pair_id"],
                    "partition": partition,
                    "pm_class": "" if partition == "train" else "C3",
                }
            )
        else:
            quarantine_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "reason": reason,
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "assigned_partition_a": partition_a,
                    "assigned_partition_b": partition_b,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "splits.csv",
        ["split_scheme", "fold", "pair_id", "partition", "pm_class"],
        split_rows,
    )
    write_csv(
        output_dir / "cross_partition_pairs.csv",
        ["pair_id", "reason", "cluster_a", "cluster_b", "assigned_partition_a", "assigned_partition_b"],
        quarantine_rows,
    )
    protein_assignment_rows = [
        {
            "uniprot_accession": row["uniprot_accession"].strip(),
            "homology_cluster_30": protein_to_cluster[row["uniprot_accession"].strip()],
            "partition": assignment[protein_to_cluster[row["uniprot_accession"].strip()]],
        }
        for row in sorted(proteins, key=lambda item: item["uniprot_accession"])
    ]
    write_csv(
        output_dir / "protein_pool_assignments.csv",
        ["uniprot_accession", "homology_cluster_30", "partition"],
        protein_assignment_rows,
    )
    metadata = {
        "algorithm_version": ALGORITHM_VERSION,
        "proteins_sha256": sha256_file(proteins_path),
        "pairs_sha256": sha256_file(pairs_path),
        "seed_search": {"start": seed_start, "end": seed_end, "selected": stats.seed},
        "node_targets": NODE_TARGETS,
        "pair_targets": PAIR_TARGETS,
        "minimum_per_class_per_partition": min_per_class,
        "stats": asdict(stats),
        "note": (
            "Cross-partition pairs are quarantined, not relabeled or reassigned. "
            "If any member of a nonempty complex_group_id spans protein pools, every pair "
            "in that evidence group is quarantined atomically."
        ),
    }
    (output_dir / "c3_split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--min-per-class", type=int, default=1)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise SystemExit("--seed-end must be >= --seed-start")
    if args.min_per_class < 1:
        raise SystemExit("--min-per-class must be >= 1")
    stats = generate(
        args.proteins,
        args.pairs,
        args.output_dir,
        args.seed_start,
        args.seed_end,
        args.min_per_class,
    )
    print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
