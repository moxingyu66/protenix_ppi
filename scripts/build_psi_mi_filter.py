#!/usr/bin/env python3
"""Compile a pinned PSI-MI filtering policy into immutable term sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from protenix_ppi.scripts.audit_raw_sources import parse_obo_terms


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descendants(terms: dict[str, dict[str, Any]], roots: set[str]) -> set[str]:
    result = set(roots)
    changed = True
    while changed:
        changed = False
        for term_id, term in terms.items():
            if term_id not in result and any(parent in result for parent in term.get("is_a", [])):
                result.add(term_id)
                changed = True
    return result


def _require_terms(
    terms: dict[str, dict[str, Any]], term_ids: set[str], label: str
) -> None:
    missing = sorted(term_ids - set(terms))
    if missing:
        raise ValueError(f"{label} contains unknown PSI-MI terms: {', '.join(missing)}")
    obsolete = sorted(term_id for term_id in term_ids if terms[term_id].get("is_obsolete"))
    if obsolete:
        raise ValueError(f"{label} contains obsolete PSI-MI roots: {', '.join(obsolete)}")


def named_terms(terms: dict[str, dict[str, Any]], term_ids: set[str]) -> list[dict[str, str]]:
    return [
        {"id": term_id, "name": str(terms[term_id].get("name", ""))}
        for term_id in sorted(term_ids)
    ]


def compile_policy(ontology_path: Path, policy_path: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("PSI-MI filter policy must be a JSON object")
    ontology_hash = sha256_file(ontology_path)
    if ontology_hash != policy.get("ontology_sha256"):
        raise ValueError(
            f"ontology SHA-256 mismatch: expected {policy.get('ontology_sha256')}, observed {ontology_hash}"
        )
    terms = parse_obo_terms(ontology_path)
    interaction = policy.get("interaction_type_policy", {})
    detection = policy.get("detection_method_policy", {})
    interaction_accept = set(interaction.get("auto_accept_exact", []))
    interaction_never = set(interaction.get("never_auto_accept_exact", []))
    detection_roots = set(detection.get("auto_accept_roots_with_descendants", []))
    detection_never_roots = set(detection.get("never_auto_accept_roots_with_descendants", []))
    detection_manual_exact = set(detection.get("manual_review_exact_even_if_under_auto_root", []))
    _require_terms(terms, interaction_accept, "interaction auto-accept")
    _require_terms(terms, interaction_never, "interaction never-auto-accept")
    _require_terms(terms, detection_roots, "detection auto-accept roots")
    _require_terms(terms, detection_never_roots, "detection never-auto-accept roots")
    _require_terms(terms, detection_manual_exact, "detection explicit manual-review terms")

    direct_descendants = descendants(terms, interaction_accept) - interaction_accept
    detection_accept_all = descendants(terms, detection_roots)
    detection_never_all = descendants(terms, detection_never_roots)
    overlap = detection_accept_all & detection_never_all
    if overlap:
        raise ValueError(
            "detection auto-accept and never-auto-accept closures overlap: "
            + ", ".join(sorted(overlap))
        )
    obsolete_auto_terms = {
        term_id for term_id in detection_accept_all if terms[term_id].get("is_obsolete")
    }
    manual_not_in_closure = detection_manual_exact - detection_accept_all
    if manual_not_in_closure:
        raise ValueError(
            "detection explicit manual-review terms are outside the auto closure: "
            + ", ".join(sorted(manual_not_in_closure))
        )
    detection_accept = detection_accept_all - obsolete_auto_terms - detection_manual_exact
    return {
        "schema_version": "1.0",
        "policy_id": policy.get("policy_id"),
        "ontology_commit": policy.get("ontology_commit"),
        "ontology_sha256": ontology_hash,
        "policy_sha256": sha256_file(policy_path),
        "interaction_type": {
            "auto_accept_exact": named_terms(terms, interaction_accept),
            "direct_descendants_manual_review": named_terms(terms, direct_descendants),
            "never_auto_accept_exact": named_terms(terms, interaction_never),
        },
        "detection_method": {
            "auto_accept_roots": named_terms(terms, detection_roots),
            "auto_accept_non_obsolete_closure": named_terms(terms, detection_accept),
            "excluded_obsolete_members_of_auto_closure": named_terms(terms, obsolete_auto_terms),
            "manual_review_members_removed_from_auto_closure": named_terms(
                terms, detection_manual_exact
            ),
            "never_auto_accept_roots": named_terms(terms, detection_never_roots),
            "never_auto_accept_closure": named_terms(terms, detection_never_all),
        },
        "decision_rule": policy.get("decision_rule"),
        "scientific_scope": policy.get("scientific_scope"),
        "counts": {
            "ontology_terms": len(terms),
            "interaction_auto_accept": len(interaction_accept),
            "interaction_descendants_manual_review": len(direct_descendants),
            "detection_auto_accept_non_obsolete": len(detection_accept),
            "detection_auto_closure_obsolete_excluded": len(obsolete_auto_terms),
            "detection_auto_closure_manual_review_excluded": len(detection_manual_exact),
            "detection_never_auto_accept": len(detection_never_all),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_policy(args.ontology, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
