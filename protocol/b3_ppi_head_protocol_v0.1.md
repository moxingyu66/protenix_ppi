# B3 Protenix PPI task-head protocol v0.1

Status: deterministic head trainer implemented and tested on fixtures; no biological Protenix feature bundle has been produced

## Definition

B3 is a label-aware nonlinear PPI ranking head trained on the exact frozen Protenix feature bundle used by B2. No Protenix backbone parameter enters the optimizer and no Protenix checkpoint is rewritten.

This definition makes the comparison controlled:

- B2: standardized class-balanced linear probe;
- B3: one-hidden-layer nonlinear task head;
- identical Protenix features, C3 split, validation cohort, and test cohort.

An improvement therefore measures the value of task-specific head capacity, not a hidden backbone update.

## Why the native confidence-training mode is not used directly

An audit of upstream Protenix commit `4c355be4553512f72453ecbfb65e69f4c35d1413` on 2026-09-15 found `ConfidenceHead` and `train_confidence_only`. That native path predicts structural confidence targets such as pLDDT, PAE, PDE, and resolved atoms from structure-labelled training data. It is not a binary PPI-label objective.

The same audit found that `get_optimizer(..., param_names=...)` uses substring groups only in the non-AdamW branch. In the AdamW branch it selects every parameter with `requires_grad=True`. Therefore `finetune_params_with_substring=["confidence_head"]` alone is not proof that only the head is trained.

The upstream commit above is audit evidence, not the final backbone lock. G1 must repeat this check on the actually executed commit.

## Frozen architecture and optimization

- input: the validated B2 pair-feature vector;
- standardization: fitted on train only during epoch selection, then refitted on train plus validation;
- hidden layers: one layer of 64 units;
- activation: ReLU;
- output: binary logistic probability;
- optimizer: Adam;
- learning rate: `1e-3`;
- L2 coefficient: `1e-4`;
- nominal batch size: 256;
- class handling: balanced sample weights;
- maximum epochs: 100;
- early-stopping patience: 10 epochs;
- epoch selection: AP on the frozen `balanced_explicit_1to1` validation cohort;
- final fit: exactly the selected number of epochs on all train plus validation examples.

The test partition is never used for epoch selection. The final prediction file still covers every C3 test pair so all frozen evaluation cohorts can be assessed.

## Run command

```bash
python3 -m protenix_ppi.scripts.train_b3 \
  --proteins <DATASET_DIR>/mmseqs30/proteins.clustered.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --feature-bundle <B2_FEATURE_BUNDLE> \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json \
  --output-dir <RESULTS_DIR>/B3/seed_42 \
  --seed 42
```

Repeat with seeds 123 and 999. Do not tune hidden size, learning rate, patience, or other architecture choices using test results.

## Required outputs

- `predictions.csv`: every untouched C3 test pair;
- `task_head.joblib`: scaler and task-head parameters only;
- `metadata.json`: architecture, trainable parameter count, validation curve, selected epoch, hashes, software, and explicit `backbone_updated=false` evidence.

## Interpretation and next gate

B3 is successful only if it improves over B2 on the same primary cohort with paired uncertainty and the result is consistent across the registered seeds. Because B3 cannot change coordinates, it inherits the frozen B1 structures; it must not claim improved structure prediction.

Structural-module tuning remains locked. B4/B5 may be designed only after B0--B3 results, a real backward-memory measurement, and an independent structural-preservation set are available.
