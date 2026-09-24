# Evaluation protocol v0.2

Status: frozen design prepared before B0/B1 scoring; cohort membership awaits the final MMseqs2/C3 split

## Why v0.2 is required

The provisional v0.3 evidence pool contains 51,446 eligible positives and 1,518 eligible explicit negatives, approximately 97.1% positives. Average precision for a random ranking equals the evaluated cohort prevalence, so pooled AP near 0.97 would not by itself demonstrate useful PPI discrimination.

No unobserved pair may be added as a negative to repair this imbalance. Instead, evaluation uses named, hashed cohorts constructed only from explicit labels before any model score is inspected.

## Primary endpoint

The primary screening comparison is the paired difference in average precision between a candidate method and its designated baseline on the `balanced_explicit_1to1` C3 test cohort.

That cohort contains:

- every retained C3 test pair with an explicit negative label;
- an equal-sized deterministic SHA-256-ranked sample of retained direct positives;
- no random or merely unobserved protein pairs.

The case-control prevalence is exactly 0.5 by design. It supports controlled discrimination comparisons, not an estimate of proteome-wide PPI prevalence or positive predictive value.

## Required secondary cohorts

`build_evaluation_cohorts.py` also freezes:

- `full_evidence_pool`: every eligible labeled pair retained by C3, with observed evidence-pool prevalence reported;
- `balanced_curated_1to1`: every curated negative plus an equal deterministic positive sample;
- `balanced_screen_1to1`: every screen negative plus an equal deterministic positive sample.

All model comparisons must use identical cohort membership and pair IDs. The full pool and both negative strata are mandatory secondary results.

## Cohort construction

After the final C3 split is generated, run:

```bash
python3 -m protenix_ppi.scripts.build_evaluation_cohorts \
  --pairs protenix_ppi/data/processed/human_ppi_2026_03_v0.3/pairs.csv \
  --splits protenix_ppi/data/processed/human_ppi_2026_03_v0.3/splits/c3_primary/splits.csv \
  --output-dir protenix_ppi/data/processed/human_ppi_2026_03_v0.3/evaluation_cohorts \
  --seed 20260915
```

The generated `cohort_membership.csv` and `cohort_metadata.json` must be hashed and frozen before B0/B1 scoring. Validation and test cohorts are generated independently. Cohorts must never be regenerated in response to model performance.

## Metric definitions

- Primary AUPRC is tie-aware average precision: `sum((Recall_n - Recall_{n-1}) * Precision_n)`.
- Trapezoidal PR-AUC remains a separately named secondary metric.
- AUROC uses the Mann–Whitney definition with 0.5 credit for score ties.
- Precision@50/100 and EF@1% are reported only with the cohort name and prevalence.
- BEDROC uses `alpha = 20`. With score ties, active members receive their
  expected share of the exponential rank weights occupied by the entire tie
  group, so CSV row order cannot change the result. BEDROC is undefined for a
  single-class cohort and is reported only with the named cohort/prevalence.
- Brier score and ECE are computed only for scores in `[0,1]`. Native `iptm` is a confidence score, not automatically a calibrated interaction probability; calibration claims require a validation-only calibration transform.

Precision@K and EF from a balanced case-control cohort do not represent a real operational screen. Proteome-wide or disease-panel claims require a separately frozen candidate pool and an explicit prevalence assumption.

## Evaluation command

For the primary cohort:

```bash
python3 -m protenix_ppi.scripts.evaluate_predictions \
  --predictions <METHOD_PREDICTIONS.csv> \
  --baseline <BASELINE_PREDICTIONS.csv> \
  --partition test \
  --cohort-membership protenix_ppi/data/processed/human_ppi_2026_03_v0.3/evaluation_cohorts/cohort_membership.csv \
  --cohort balanced_explicit_1to1 \
  --output <RESULTS_DIR>/evaluation_balanced_explicit_1to1.json
```

Repeat with the three secondary cohort names. Prediction files may contain the entire test partition; the evaluator verifies labels, evidence strata, bootstrap groups, and complete membership before selecting a cohort.

## Uncertainty and seeds

- Training seeds are 42, 123, and 999.
- Report each seed, mean, and standard deviation; never select the best test seed.
- Use 2,000 paired bootstrap replicates with seed `20260915`.
- Resample `bootstrap_group`, using `complex_group_id` when available and pair ID otherwise.
- Report point estimates, 95% percentile intervals, paired absolute AP delta, and relative delta.

Create a five-column manifest from `configs/multiseed_comparison_manifest.template.csv` and aggregate only after all three registered runs exist:

```bash
python3 -m protenix_ppi.scripts.aggregate_multiseed \
  --manifest <RESULTS_DIR>/multiseed_comparison_manifest.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --cohort balanced_explicit_1to1 \
  --output <RESULTS_DIR>/multiseed_balanced_explicit_1to1.json
```

The aggregator rejects missing/extra seeds, metadata-seed mismatches, cohort coverage drift, and pair/label/bootstrap-group differences between methods or seeds.

## Decision comparisons

- B1 versus the exact constant-score reference: native zero-shot signal. Every
  null score is 0.5, so tie-aware AP equals cohort prevalence without noise
  from an arbitrary random permutation.
- B2 versus B0b on the same cohort: incremental frozen structural-representation value beyond sequence embeddings.
- B3 versus B2: value of task-specific head training.
- B4/B5 versus B3: structural-update value, only after the structural-tuning gate opens.

For the registered trained-method comparisons B0b versus B0a, B2 versus B0b,
and B3 versus B2, a conservative claim is supported only when all three
conditions hold on `balanced_explicit_1to1` test data:

1. the across-seed mean absolute AP delta is positive;
2. the AP delta is positive for each of seeds 42, 123, and 999;
3. the paired group-bootstrap 95% lower bound for the absolute AP delta is
   positive for each seed.

Failure of this rule means that the registered incremental-value claim is not
established; it does not prove equivalence. Regardless of the primary outcome,
the full evidence pool and curated/screen-negative cohorts must still be
reported. Run the complete bound comparison with:

```bash
python3 -m protenix_ppi.scripts.evaluate_comparison_suite \
  --manifest <RESULTS_DIR>/multiseed_comparison_manifest.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --benchmark-freeze-manifest <DATASET_DIR>/benchmark_freeze_manifest.json \
  --comparison b2_vs_b0b \
  --output-dir <RESULTS_DIR>/comparisons/b2_vs_b0b
```

Valid comparison identifiers are `b1_vs_constant`, `b0b_vs_b0a`,
`b2_vs_b0b`, `b2_vs_b1`, `b3_vs_b2`, and `b3_vs_b1`.
The suite checks exact method roles, frozen data hashes, all three seeds,
prediction hashes, and—in the B3/B2 comparison—the identical frozen feature
bundle before producing the four mandatory cohort reports.

This protocol does not authorize structural tuning and does not change the requirement for structural-preservation, cost, and three-seed validation.
