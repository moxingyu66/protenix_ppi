# B2 frozen Protenix representation protocol v0.1

Status: bundle validator and controlled classifier implemented; exact internal tensor hook intentionally remains open until the pinned G1 source commit is inspected on the real server

## Purpose

B2 tests whether label-blind frozen Protenix representations add PPI discrimination beyond the B0b frozen ESM-2 sequence baseline. B2 is not Protenix fine-tuning: the backbone receives no gradient and is not updated.

## Mandatory order

1. G0 identifies the real Linux GPU route.
2. G1 freezes the Protenix Git commit, package, checkpoint digest, kernels, and working inference path.
3. G2 freezes MMseqs2 clusters, the C3 split, and evaluation cohorts.
4. The exact feature hook and pooling rule are inspected against the pinned G1 source and written to `feature_extraction_lock.json`.
5. Frozen features are extracted once without consulting PPI labels or model performance.
6. The bundle validator must pass before B2 training.

The tensor hook is not named in advance because moving Protenix revisions may expose different module/tensor names. Inventing a hook before G1 would create a nominal protocol that might not correspond to the executed computation. The hook must nevertheless be frozen before feature extraction or B2 model fitting.

Source-level planning evidence for the currently audited commit identifies a
candidate, not a final lock: the third return value `z` from
`Protenix.get_pairformer_output(...)`, the final recycled Pairformer pair
representation with shape `[..., N_token, N_token, c_z]`. The proposed pooling
is cross-protein token-pair masking, direction symmetrization, and per-channel
mean/standard-deviation pooling. See
`artifacts/source/protenix_pair_feature_hook_audit_20260915.md`; G1 must verify
the exact path, configured dimension, cycle timing, dtype, and swap tolerance on
the executed installation before this candidate can enter the feature lock.

## Feature requirements

The primary B2 vector must contain a genuine frozen Protenix structural or pair representation, not only native `iptm`/`ranking_score` scalars. Its lock must record:

- exact source module and tensor name from the pinned commit;
- tensor shape before pooling;
- protein-chain masks and inter-chain mask construction;
- label-blind pooling and any normalization;
- diffusion sample/rank aggregation policy;
- final ordered feature names and dimension;
- numerical dtype;
- a chain-order swap-invariance audit.

The representation must be invariant to exchanging protein A and B, either by construction or by a predeclared symmetric aggregation. Canonical accession order alone is not a biological feature.

Native B1 scalar scores may be included only as separately named components. A scalar-only classifier is a B1 recalibration diagnostic, not the primary B2 representation experiment.

## Bundle contract

Each bundle contains:

```text
features.npy
feature_index.csv
feature_metadata.json
feature_extraction_lock.json
b2_hook_validation.json
```

`feature_index.csv` binds every vector row to the exact pair, C3 partition, Protenix input hash, and feature-vector hash. The bundle must cover every retained train/validation/test pair exactly once.

The validator rejects:

- a later-cutoff or different backbone;
- missing/placeholder tensor definitions;
- label-aware extraction;
- a non-symmetric feature policy;
- missing, duplicate, or unexpected pair IDs;
- partition drift;
- pairs/splits/backbone/extraction-lock/hook-validation hash drift;
- vector checksum mismatch;
- NaN, infinity, shape, dtype, or feature-name mismatch.

Validate with:

```bash
python3 -m protenix_ppi.scripts.pair_feature_bundle \
  --bundle-dir <B2_FEATURE_BUNDLE> \
  --pairs <DATASET_DIR>/pairs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json
```

Templates are provided at `configs/b2_feature_extraction_lock.template.json` and `configs/b2_feature_metadata.template.json`. Placeholders are deliberately rejected by the validator.

Before validating a feature bundle, validate the evidence-bound hook lock:

```bash
python3 -m protenix_ppi.scripts.validate_b2_hook_lock \
  --lock <B2_FEATURE_BUNDLE>/feature_extraction_lock.json \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json \
  --g1-manifest <G1_RUN_DIR>/manifest.json \
  --hook-evidence <G1_RUN_DIR>/backward/b2_hook_evidence.json \
  --output <B2_FEATURE_BUNDLE>/b2_hook_validation.json
```

The feature-bundle validator fails closed when this report is absent, is not
`passed`, or disagrees with the extraction lock, backbone, G1 manifest, hook
evidence, source commit, tensor shape, or registered feature component. B2 and
B3 therefore cannot train by accidentally skipping this command.

The hook validator requires the actual final-recycle `z` tensor, source-commit and
G1 hash binding, label-blind extraction, exact tensor shape, MSA/template policy,
and a passed A/B chain-swap audit. Its output must exist before B2 training; the
hook-evidence template is `configs/b2_hook_evidence.template.json`.

## Controlled classifier

B2 uses the same standardized, class-balanced logistic-regression family as B0a/B0b. The fixed regularization grid is `C = 0.01, 0.1, 1, 10, 100`; validation AP selects C, ties select the smaller value, and the final model is refit on train plus validation before one test prediction pass.

Run each registered seed:

```bash
python3 -m protenix_ppi.scripts.train_b2 \
  --proteins <DATASET_DIR>/mmseqs30/proteins.clustered.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --feature-bundle <B2_FEATURE_BUNDLE> \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json \
  --output-dir <RESULTS_DIR>/B2/seed_42 \
  --seed 42
```

Repeat with seeds 123 and 999. All seeds use the identical frozen feature bundle and C3 test pairs.

## Interpretation gate

The primary structural-representation claim compares B2 against B0b on the same `balanced_explicit_1to1` test cohort with paired uncertainty. B2 must also report the full evidence pool and curated/screen-negative cohorts.

A B2 improvement does not justify structural fine-tuning by itself. It establishes that useful frozen Protenix signal exists and provides the reference that B3 must exceed.
