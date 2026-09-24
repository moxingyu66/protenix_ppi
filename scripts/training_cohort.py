"""Shared validation-cohort selection checks for B0--B3 training."""

from __future__ import annotations

from pathlib import Path

from protenix_ppi.scripts.evaluate_predictions import read_cohort_membership


PRIMARY_SELECTION_COHORT = "balanced_explicit_1to1"


def select_examples_by_cohort(examples, membership_path: Path, cohort: str, partition: str = "validation"):
    members = read_cohort_membership(membership_path, cohort, partition)
    examples_by_id = {pair["pair_id"].strip(): (pair, features) for pair, features in examples}
    missing = sorted(set(members) - set(examples_by_id))
    if missing:
        raise ValueError(f"training examples are missing validation-cohort pairs: {missing[:5]}")
    selected = []
    for pair_id in sorted(members):
        pair, features = examples_by_id[pair_id]
        member = members[pair_id]
        if int(pair["label"]) != member.label:
            raise ValueError(f"validation-cohort label mismatch for {pair_id}")
        if pair["evidence_class"].strip() != member.evidence_class:
            raise ValueError(f"validation-cohort evidence_class mismatch for {pair_id}")
        bootstrap_group = pair["complex_group_id"].strip() or pair_id
        if bootstrap_group != member.bootstrap_group:
            raise ValueError(f"validation-cohort bootstrap_group mismatch for {pair_id}")
        selected.append((pair, features))
    labels = {int(pair["label"]) for pair, _ in selected}
    if labels != {0, 1}:
        raise ValueError(f"validation cohort must contain both labels; observed {sorted(labels)}")
    return selected
