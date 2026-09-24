#!/usr/bin/env python3
"""Evaluate PPI ranking predictions with tie-aware metrics and paired bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Prediction:
    pair_id: str
    label: int
    score: float
    partition: str
    evidence_class: str
    bootstrap_group: str


@dataclass(frozen=True)
class CohortMember:
    pair_id: str
    label: int
    evidence_class: str
    bootstrap_group: str


def read_predictions(path: Path, partition: str) -> list[Prediction]:
    required = {"pair_id", "label", "score", "partition", "evidence_class", "bootstrap_group"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        predictions: list[Prediction] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            if row["partition"].strip() != partition:
                continue
            pair_id = row["pair_id"].strip()
            if not pair_id or pair_id in seen:
                raise ValueError(f"{path.name}:{line}: missing or duplicate pair_id {pair_id!r}")
            seen.add(pair_id)
            try:
                label = int(row["label"])
                score = float(row["score"])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line}: invalid label or score") from exc
            if label not in {0, 1}:
                raise ValueError(f"{path.name}:{line}: label must be 0 or 1")
            if not math.isfinite(score):
                raise ValueError(f"{path.name}:{line}: score must be finite")
            group = row["bootstrap_group"].strip() or pair_id
            predictions.append(
                Prediction(
                    pair_id=pair_id,
                    label=label,
                    score=score,
                    partition=partition,
                    evidence_class=row["evidence_class"].strip(),
                    bootstrap_group=group,
                )
            )
    if not predictions:
        raise ValueError(f"{path.name} has no rows for partition {partition!r}")
    return predictions


def read_cohort_membership(path: Path, cohort: str, partition: str) -> dict[str, CohortMember]:
    required = {"cohort", "partition", "pair_id", "label", "evidence_class", "bootstrap_group"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        members: dict[str, CohortMember] = {}
        for line, row in enumerate(reader, start=2):
            if row["cohort"].strip() != cohort or row["partition"].strip() != partition:
                continue
            pair_id = row["pair_id"].strip()
            if not pair_id or pair_id in members:
                raise ValueError(f"{path.name}:{line}: missing or duplicate cohort pair_id {pair_id!r}")
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line}: invalid cohort label") from exc
            if label not in {0, 1}:
                raise ValueError(f"{path.name}:{line}: cohort label must be 0 or 1")
            members[pair_id] = CohortMember(
                pair_id=pair_id,
                label=label,
                evidence_class=row["evidence_class"].strip(),
                bootstrap_group=row["bootstrap_group"].strip() or pair_id,
            )
    if not members:
        raise ValueError(f"{path.name} has no rows for cohort={cohort!r}, partition={partition!r}")
    return members


def select_cohort(predictions: list[Prediction], members: dict[str, CohortMember]) -> list[Prediction]:
    predictions_by_id = {item.pair_id: item for item in predictions}
    missing = sorted(set(members) - set(predictions_by_id))
    if missing:
        raise ValueError(f"prediction file is missing cohort pairs: {missing[:5]}")
    selected: list[Prediction] = []
    for pair_id in sorted(members):
        prediction = predictions_by_id[pair_id]
        member = members[pair_id]
        if prediction.label != member.label:
            raise ValueError(f"cohort label mismatch for {pair_id}")
        if prediction.evidence_class != member.evidence_class:
            raise ValueError(f"cohort evidence_class mismatch for {pair_id}")
        if prediction.bootstrap_group != member.bootstrap_group:
            raise ValueError(f"cohort bootstrap_group mismatch for {pair_id}")
        selected.append(prediction)
    return selected


def _score_groups(predictions: list[Prediction], descending: bool = True) -> list[list[Prediction]]:
    ordered = sorted(predictions, key=lambda item: item.score, reverse=descending)
    groups: list[list[Prediction]] = []
    for item in ordered:
        if not groups or item.score != groups[-1][0].score:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def average_precision(predictions: list[Prediction]) -> float | None:
    positives = sum(item.label for item in predictions)
    if positives == 0:
        return None
    true_positives = 0
    seen = 0
    result = 0.0
    for group in _score_groups(predictions):
        group_positives = sum(item.label for item in group)
        true_positives += group_positives
        seen += len(group)
        result += (group_positives / positives) * (true_positives / seen)
    return result


def pr_auc_trapezoid(predictions: list[Prediction]) -> float | None:
    positives = sum(item.label for item in predictions)
    if positives == 0:
        return None
    points = [(0.0, 1.0)]
    true_positives = 0
    seen = 0
    for group in _score_groups(predictions):
        true_positives += sum(item.label for item in group)
        seen += len(group)
        points.append((true_positives / positives, true_positives / seen))
    area = 0.0
    for (recall_a, precision_a), (recall_b, precision_b) in zip(points, points[1:]):
        area += (recall_b - recall_a) * (precision_a + precision_b) / 2
    return area


def auroc(predictions: list[Prediction]) -> float | None:
    positives = sum(item.label for item in predictions)
    negatives = len(predictions) - positives
    if positives == 0 or negatives == 0:
        return None
    negatives_before = 0
    favorable_pairs = 0.0
    for group in _score_groups(predictions, descending=False):
        group_positives = sum(item.label for item in group)
        group_negatives = len(group) - group_positives
        favorable_pairs += group_positives * negatives_before
        favorable_pairs += 0.5 * group_positives * group_negatives
        negatives_before += group_negatives
    return favorable_pairs / (positives * negatives)


def expected_hits_at_k(predictions: list[Prediction], k: int) -> float:
    if k <= 0 or not predictions:
        return 0.0
    remaining = min(k, len(predictions))
    expected_hits = 0.0
    for group in _score_groups(predictions):
        group_positives = sum(item.label for item in group)
        if remaining >= len(group):
            expected_hits += group_positives
            remaining -= len(group)
        else:
            expected_hits += remaining * group_positives / len(group)
            break
        if remaining == 0:
            break
    return expected_hits


def precision_at_k(predictions: list[Prediction], k: int) -> float:
    denominator = min(max(k, 0), len(predictions))
    return expected_hits_at_k(predictions, k) / denominator if denominator else 0.0


def recall_at_k(predictions: list[Prediction], k: int) -> float | None:
    positives = sum(item.label for item in predictions)
    return expected_hits_at_k(predictions, k) / positives if positives else None


def enrichment_factor(predictions: list[Prediction], fraction: float = 0.01) -> float | None:
    if not predictions:
        return None
    prevalence = sum(item.label for item in predictions) / len(predictions)
    if prevalence == 0:
        return None
    k = max(1, math.ceil(len(predictions) * fraction))
    return precision_at_k(predictions, k) / prevalence


def bedroc(predictions: list[Prediction], alpha: float = 20.0) -> float | None:
    """Return tie-aware BEDROC using expected exponential rank weights.

    Active members of an equal-score group receive their expected share of the
    exponential weights occupied by that whole group. This makes the result
    independent of CSV row order while retaining the Truchon--Bayly/RDKit
    normalization for rankings without ties.
    """
    if not predictions:
        return None
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("BEDROC alpha must be finite and greater than zero")
    total = len(predictions)
    positives = sum(item.label for item in predictions)
    if positives == 0 or positives == total:
        return None

    weighted_active_sum = 0.0
    rank_start = 1
    for group in _score_groups(predictions):
        active_fraction = sum(item.label for item in group) / len(group)
        rank_stop = rank_start + len(group)
        group_weight = sum(
            math.exp(-alpha * rank / total) for rank in range(rank_start, rank_stop)
        )
        weighted_active_sum += active_fraction * group_weight
        rank_start = rank_stop

    observed_mean = weighted_active_sum / positives
    maximum_mean = sum(
        math.exp(-alpha * rank / total) for rank in range(1, positives + 1)
    ) / positives
    minimum_mean = sum(
        math.exp(-alpha * rank / total)
        for rank in range(total - positives + 1, total + 1)
    ) / positives
    denominator = maximum_mean - minimum_mean
    if denominator <= 0:
        return None
    value = (observed_mean - minimum_mean) / denominator
    # Protect the documented [0, 1] range from floating-point roundoff only.
    return min(1.0, max(0.0, value))


def brier_score(predictions: list[Prediction]) -> float | None:
    if not predictions or any(item.score < 0 or item.score > 1 for item in predictions):
        return None
    return sum((item.score - item.label) ** 2 for item in predictions) / len(predictions)


def expected_calibration_error(predictions: list[Prediction], bins: int = 10) -> float | None:
    if not predictions or bins <= 0 or any(item.score < 0 or item.score > 1 for item in predictions):
        return None
    bucketed: list[list[Prediction]] = [[] for _ in range(bins)]
    for item in predictions:
        index = min(int(item.score * bins), bins - 1)
        bucketed[index].append(item)
    total = len(predictions)
    error = 0.0
    for bucket in bucketed:
        if not bucket:
            continue
        confidence = sum(item.score for item in bucket) / len(bucket)
        accuracy = sum(item.label for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - confidence)
    return error


def compute_metrics(predictions: list[Prediction]) -> dict[str, float | int | None]:
    positives = sum(item.label for item in predictions)
    k_one_percent = max(1, math.ceil(len(predictions) * 0.01))
    return {
        "n": len(predictions),
        "positives": positives,
        "negatives": len(predictions) - positives,
        "prevalence": positives / len(predictions) if predictions else None,
        "average_precision": average_precision(predictions),
        "pr_auc_trapezoid": pr_auc_trapezoid(predictions),
        "auroc": auroc(predictions),
        "precision_at_50": precision_at_k(predictions, 50),
        "precision_at_100": precision_at_k(predictions, 100),
        "recall_at_50": recall_at_k(predictions, 50),
        "recall_at_100": recall_at_k(predictions, 100),
        "ef_at_1_percent": enrichment_factor(predictions, 0.01),
        "bedroc_alpha_20": bedroc(predictions, 20.0),
        "k_at_1_percent": k_one_percent,
        "brier_score": brier_score(predictions),
        "ece_10_equal_width": expected_calibration_error(predictions, 10),
    }


def align_predictions(candidate: list[Prediction], baseline: list[Prediction]) -> tuple[list[Prediction], list[Prediction]]:
    candidate_by_id = {item.pair_id: item for item in candidate}
    baseline_by_id = {item.pair_id: item for item in baseline}
    if set(candidate_by_id) != set(baseline_by_id):
        missing_candidate = sorted(set(baseline_by_id) - set(candidate_by_id))[:5]
        missing_baseline = sorted(set(candidate_by_id) - set(baseline_by_id))[:5]
        raise ValueError(
            f"candidate/baseline pair sets differ; missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    ordered_ids = sorted(candidate_by_id)
    aligned_candidate = [candidate_by_id[pair_id] for pair_id in ordered_ids]
    aligned_baseline = [baseline_by_id[pair_id] for pair_id in ordered_ids]
    for left, right in zip(aligned_candidate, aligned_baseline):
        if left.label != right.label:
            raise ValueError(f"label mismatch for {left.pair_id}")
        if left.bootstrap_group != right.bootstrap_group:
            raise ValueError(f"bootstrap_group mismatch for {left.pair_id}")
    return aligned_candidate, aligned_baseline


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_group_bootstrap(
    candidate: list[Prediction],
    baseline: list[Prediction] | None,
    metric: Callable[[list[Prediction]], float | None],
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    if replicates <= 0:
        return {"replicates_requested": replicates, "replicates_valid": 0, "lower_95": None, "upper_95": None}
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(candidate):
        groups[item.bootstrap_group].append(index)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled_groups = [rng.choice(group_ids) for _ in group_ids]
        indices = [index for group_id in sampled_groups for index in groups[group_id]]
        candidate_value = metric([candidate[index] for index in indices])
        if candidate_value is None:
            continue
        if baseline is None:
            values.append(candidate_value)
        else:
            baseline_value = metric([baseline[index] for index in indices])
            if baseline_value is not None:
                values.append(candidate_value - baseline_value)
    if not values:
        return {"replicates_requested": replicates, "replicates_valid": 0, "lower_95": None, "upper_95": None}
    return {
        "replicates_requested": replicates,
        "replicates_valid": len(values),
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
    }


def evaluate(
    candidate: list[Prediction],
    baseline: list[Prediction] | None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    if baseline is not None:
        candidate, baseline = align_predictions(candidate, baseline)
    overall = compute_metrics(candidate)
    result: dict = {
        "metric_definition": "average_precision_with_score_ties_grouped",
        "candidate": overall,
        "strata": {},
        "bootstrap_average_precision": paired_group_bootstrap(
            candidate, None, average_precision, bootstrap_replicates, bootstrap_seed
        ),
    }
    negative_classes = sorted({item.evidence_class for item in candidate if item.label == 0})
    for negative_class in negative_classes:
        subset = [item for item in candidate if item.label == 1 or item.evidence_class == negative_class]
        result["strata"][f"positives_vs_{negative_class}"] = compute_metrics(subset)
    if baseline is not None:
        candidate_ap = average_precision(candidate)
        baseline_ap = average_precision(baseline)
        absolute_delta = None if candidate_ap is None or baseline_ap is None else candidate_ap - baseline_ap
        relative_delta = None if absolute_delta is None or baseline_ap == 0 else absolute_delta / baseline_ap
        result["baseline"] = compute_metrics(baseline)
        result["comparison"] = {
            "average_precision_absolute_delta": absolute_delta,
            "average_precision_relative_delta": relative_delta,
            "paired_bootstrap_absolute_delta": paired_group_bootstrap(
                candidate, baseline, average_precision, bootstrap_replicates, bootstrap_seed
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--cohort-membership", type=Path)
    parser.add_argument("--cohort")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.cohort_membership) != bool(args.cohort):
        raise SystemExit("--cohort-membership and --cohort must be supplied together")
    candidate = read_predictions(args.predictions, args.partition)
    baseline = read_predictions(args.baseline, args.partition) if args.baseline else None
    full_partition_rows = len(candidate)
    if args.cohort_membership:
        members = read_cohort_membership(args.cohort_membership, args.cohort, args.partition)
        candidate = select_cohort(candidate, members)
        if baseline is not None:
            baseline = select_cohort(baseline, members)
    result = evaluate(candidate, baseline, args.bootstrap_replicates, args.bootstrap_seed)
    result["inputs"] = {
        "predictions": str(args.predictions.resolve()),
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "partition": args.partition,
        "full_partition_rows": full_partition_rows,
        "cohort_membership": str(args.cohort_membership.resolve()) if args.cohort_membership else None,
        "cohort": args.cohort,
        "bootstrap_seed": args.bootstrap_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
