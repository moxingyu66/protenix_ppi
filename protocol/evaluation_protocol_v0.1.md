# Evaluation protocol v0.1

Status: superseded before model scoring by `evaluation_protocol_v0.2.md`; the v0.3 evidence pool has approximately 97.1% positives before C3 retention, so pooled AP alone is not an informative primary discrimination endpoint

## Primary endpoint

The primary endpoint is **average precision (AP)** on the frozen `c3_primary` test partition. In project figures it may be labeled `AUPRC (average precision)`, but the report must state the exact definition.

AP is computed over descending score thresholds as:

```text
AP = sum_n (Recall_n - Recall_{n-1}) * Precision_n
```

All observations with exactly equal scores enter at the same threshold. Their internal file order cannot change AP.

Trapezoidal area under the precision-recall curve is reported separately as `pr_auc_trapezoid`; it must not silently replace AP because interpolation can give a different result.

## Other metrics

- AUROC uses the Mann–Whitney interpretation with a contribution of 0.5 for equal positive/negative scores.
- Precision@K and Recall@K use an expected tie-aware cutoff. If K falls inside an equal-score group, the expected number of positives is proportional to that group's class composition.
- EF@1% is tie-aware Precision@K divided by the test-set positive prevalence, with `K = max(1, ceil(0.01 * N))`.
- Brier score is the mean squared probability error.
- ECE uses 10 fixed-width bins on `[0,1]`, weighted by bin occupancy.
- Calibration metrics are undefined for scores outside `[0,1]`; such scores must be calibrated on validation data before calibration claims are made.

## Required prediction table

One file per method and seed:

| Field | Meaning |
|---|---|
| `pair_id` | Frozen benchmark pair identifier |
| `label` | Ground-truth 0/1 label |
| `score` | Higher means more likely to interact |
| `partition` | Train/validation/test |
| `evidence_class` | Positive or negative evidence stratum |
| `bootstrap_group` | Dependence-safe resampling group; use pair ID if no stronger grouping exists |

Prediction tables must contain exactly one row per pair. Candidate and baseline comparisons require identical pair IDs and labels.

## Stratification

The overall test score is primary. Additional one-vs-positive evaluations are reported separately for:

- curated experimental negatives;
- explicit screen negatives;
- compartment-rule negatives;
- other predeclared difficult-negative strata.

This prevents easy compartment negatives from hiding failure against biologically plausible hard negatives.

## Uncertainty

- Use paired resampling for comparisons: candidate and baseline predictions from the same resampled observations.
- Resample `bootstrap_group`, not arbitrary rows, when multiple observations share a complex, publication batch, or other dependence unit.
- Default descriptive interval: 2,000 bootstrap replicates with seed `20260915`.
- Report the point estimate, 95% percentile interval, paired absolute delta, and relative delta where the baseline is nonzero.
- A confidence interval is descriptive evidence, not a substitute for the predeclared effect-size gate.

## Multi-seed reporting

Model training seeds are 42, 123, and 999. Preserve each prediction file and report:

1. each seed separately;
2. mean and standard deviation across seeds;
3. an explicitly named ensemble only if its construction was fixed using validation data;
4. paired bootstrap intervals for the primary comparison.

Never select the best test seed.

## Decision comparisons

- B1 vs prevalence/random reference: native zero-shot signal.
- B2 vs B0 on identical C3 pairs: incremental value of frozen Protenix structural features.
- B3 vs B2: value of head tuning beyond an external frozen classifier.
- B4/B5 vs B3: value of structural updates, conditional on structural-preservation results.
