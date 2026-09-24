# Protenix-PPI research protocol v0.1

Status: internal pre-model protocol; provisional labels have been constructed, so this file must not be described as an externally preregistered record  
Primary model: `protenix_base_default_v1.0.0`  
Declared pretraining cutoff: `2021-09-30`

No Protenix PPI score has been generated or inspected. Metric, model, and split rules may still be frozen before model evaluation, but any external preregistration claim requires a separately timestamped v0.2 record before B0/B1 model scoring.

## 1. Research question

For human direct physical protein-protein interaction screening, do Protenix structural representations improve ranking under a strict double-unseen-protein (C3) evaluation, and does task-specific lightweight tuning add value beyond a frozen representation without materially degrading complex structure prediction?

## 2. Testable hypotheses

- **H1 — structural signal:** zero-shot Protenix interface/confidence outputs outperform a prevalence-matched random ranking on C3.
- **H2 — frozen representation value:** a classifier trained on frozen Protenix representations outperforms a matched sequence-only baseline on C3.
- **H3 — tuning value:** a Protenix PPI head outperforms the frozen-representation classifier on C3.
- **H4 — structural-module value:** lightweight structural tuning outperforms head-only tuning while satisfying the structural-preservation bound.

H4 is optional and must not be tested until the G4 decision gate is passed.

## 3. Experimental ladder

| ID | System | Updated parameters | Purpose |
|---|---|---|---|
| B0 | Sequence statistics / ESM embedding classifier | External classifier | Detect sequence and topology shortcuts |
| B1 | Protenix zero-shot | None | Measure native structural prior |
| B2 | Frozen Protenix representation + classifier | External classifier | Measure accessible frozen structural signal |
| B3 | Protenix confidence/ranking head | Head only | Test task-specific calibration/ranking |
| B4 | PEFT on selected pair/interface projections | Adapter parameters only | Optional structural representation update |
| B5 | Selected late diffusion layers | Restricted late layers | Optional, high-risk mechanism test |

The executable B3 definition is the nonlinear PPI task head in `b3_ppi_head_protocol_v0.1.md`. It consumes the same frozen Protenix pair-feature bundle as B2 and updates no backbone parameter. The native structural-confidence training path is not treated as a binary PPI objective.

## 4. Primary estimand and metrics

- Primary estimand: paired difference in C3 AUPRC between the candidate method and its designated baseline on an identical frozen test set.
- Primary metric: AUPRC.
- Ranking metrics: Precision@50, Precision@100, EF@1%, BEDROC where the candidate-pool size supports it.
- Secondary discrimination metric: AUROC.
- Calibration metrics: Brier score and ECE.
- Structural metrics: DockQ, interface RMSD, ligand RMSD where applicable, and interface-contact precision/recall.
- Engineering metrics: peak GPU memory, wall-clock time per pair, preprocessing time, and disk footprint.

Collection and completeness rules for these quantities are specified in `engineering_cost_protocol_v0.1.md`; failed/OOM/timeout observations may not be dropped from the denominator.

## 5. Split policy

- Pair-random split is diagnostic only and cannot support the primary claim.
- C3 is the primary PPI evaluation: neither endpoint protein may occur in training.
- Homology clusters must be assigned atomically to one split. The working threshold is 30% sequence identity and at least 50% alignment coverage; the exact MMseqs2 command will be frozen before labels are inspected.
- Reciprocal pairs `(A,B)` and `(B,A)` are one undirected observation.
- Entries derived from the same stable complex or publication batch must receive a group identifier and cannot cross splits.
- Structural temporal holdout entries must postdate the primary model cutoff and be audited for sequence homology to pre-cutoff structures.
- No test result may be used for threshold selection, feature selection, early stopping, or hyperparameter tuning.

## 6. Label policy

- Positive means direct physical interaction supported by an accepted binary/direct detection method.
- Unobserved pairs are unlabeled, not automatically negative.
- Curated negatives are reserved primarily for validation/testing unless their evidence provenance is sufficient and the resulting sampling distribution is explicitly modeled.
- Compartment-separated negatives and screen-negative hard pairs must remain distinct strata.
- Every record must preserve source version, accession mapping, evidence identifier, detection method, retrieval date, and transformation history.

## 7. Repetition and uncertainty

- Training seeds: 42, 123, and 999.
- Report mean, individual-seed values, and a 95% bootstrap confidence interval computed over test pairs or leakage-safe groups.
- Statistical comparisons must use paired predictions on the same test observations.
- The decision rule must use an effect size and confidence interval, not a single best seed.

## 8. Predeclared gates

### G0 — environment gate

Pass when the real execution environment, per-GPU memory, usable storage, runtime isolation method, and scheduler limits are recorded.

### G1 — Protenix execution gate

Pass when a pinned source revision can:

1. complete one official two-chain inference;
2. emit a structure and confidence outputs;
3. repeat deterministically enough under a fixed seed for pipeline debugging;
4. execute one training/backward step with nonzero gradients in the intended trainable module;
5. record peak memory and elapsed time.

### G4 — permission to test structural tuning

Structural tuning is permitted only if:

- B0–B3 are complete on the frozen C3 test set;
- B2 demonstrates usable structural signal beyond B0;
- B3 leaves a scientifically meaningful, diagnosed limitation rather than only an engineering inconvenience;
- an independent structural preservation set is available;
- the machine can support the proposed backward graph.

The executable evidence contract and exact authorization checks are frozen in
`g4_structural_tuning_gate_v0.1.md`. In addition to the conditions above, B3
must pass the registered three-seed comparison against B2, all evidence must be
hash-bound to the same G2 freeze, and the limitation diagnosis must predate and
exclude B4/B5 outputs.

### G5 — structural tuning success

Working thresholds, to be finalized before B4/B5:

- C3 AUPRC improvement over B3 is at least 2.5% relative and its paired uncertainty interval is reported;
- mean DockQ degradation is no worse than 0.05;
- the fraction of acceptable complexes (`DockQ >= 0.23`) does not drop by more than 10 percentage points;
- no systematic chain collapse, atomic overlap, or within-chain geometry failure is observed.

The executable non-inferiority interpretation is frozen in `structural_preservation_protocol_v0.1.md`: the 95% paired-bootstrap lower bounds must also remain within the DockQ and acceptable-fraction margins, and no new severe geometry failure is permitted.

## 9. Model-version rule

The primary experiment must record both the installed package version and exact Git commit. Changing the commit, model checkpoint, default recycling count, diffusion steps, MSA policy, template policy, or kernel implementation creates a new experimental condition and cannot silently replace earlier results.

## 10. Important implementation warning

The official `finetune_demo.sh` fine-tunes on a PDB-ID subset with native structural losses. It is evidence that a backward graph exists, but it is not a PPI-label fine-tuning solution. PPI experiments will require a traceable pair-label dataset adapter, a task head, explicit negative-sample gradient handling, and structural-preservation evaluation.
