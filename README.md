# Protenix-PPI validation workspace

This directory turns the research proposal into a gated, reproducible validation project.

## Current status

- Compute track: **G0 remains open** until the actual Linux GPU server is audited.
- Data track: **G2 provisional normalization complete; MMseqs2/C3 freeze pending**.
- Primary backbone: `protenix_base_default_v1.0.0`
- Primary backbone training cutoff: `2021-09-30`
- No model installation, weight download, or PPI fine-tuning has been performed yet.
- All eight registered raw sources pass integrity checks.
- UniProt `2026_03` identifier/sequence snapshot is pinned.
- Current provisional normalized build: `data/processed/human_ppi_2026_03_v0.3`.
- The provisional build contains 53,486 pairs, including 1,518 eligible explicit negatives, and passes schema/source validation, but it has no MMseqs2 clusters or split assignments and is not a frozen benchmark.

## Why this model is pinned

The primary study uses the 368M-parameter `protenix_base_default_v1.0.0` because its published training cutoff matches AlphaFold 3 and permits a defensible temporal holdout. The newer `protenix_base_20250630_v1.0.0` is excluded from the primary study because its later cutoff would contaminate a post-2021 structural holdout. Protenix-v2 may be used later as a robustness check, not as the first implementation target.

## Project gates

1. **G0 Environment audit:** identify the real GPU, storage, runtime, and job constraints.
2. **G1 Backbone smoke test:** reproduce one official two-chain inference and one backward pass.
3. **G2 Benchmark freeze:** construct traceable labels and freeze Random/C3/homology/temporal splits.
4. **G3 Baselines:** run sequence, zero-shot Protenix, and frozen-representation baselines.
5. **G4 Head tuning:** train and evaluate a PPI confidence/ranking head.
6. **G5 Optional structural tuning:** proceed only if head tuning passes the predeclared gate.
7. **G6 Final validation:** structural preservation, three-seed statistics, calibration, and cost accounting.

## G2 data contract

The normalized benchmark schema is documented in `protocol/data_contract_v0.1.md`. Empty CSV templates live under `data/templates/`. Validate populated tables with:

Identifier normalization and its current counts are documented in `protocol/uniprot_mapping_protocol_v0.1.md`.

Raw-source acquisition is frozen separately in `protocol/source_acquisition_protocol_v0.1.md` and `configs/source_registry_v0.1.json`. Fetch the small core sources with:

```bash
python3 -m protenix_ppi.scripts.fetch_raw_sources \
  --registry protenix_ppi/configs/source_registry_v0.1.json \
  --output-root protenix_ppi/data/raw
```

Run `audit_raw_sources.py` before any biological normalization. A successful raw audit proves file integrity and provenance only; it does not prove that records are eligible labels.

```bash
python3 protenix_ppi/scripts/validate_dataset.py \
  --proteins <DATASET_DIR>/proteins.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits.csv
```

The validator checks sequence hashes, pair canonicalization, label/evidence consistency, source-record contradictions, C3 node separation, homology-cluster separation, and complex-group leakage.

Generate the frozen homology-aware C3 split after all proteins have MMseqs2 cluster IDs:

```bash
bash protenix_ppi/scripts/run_mmseqs2_clustering.sh \
  protenix_ppi/data/processed/human_ppi_2026_03_v0.3/proteins.csv \
  protenix_ppi/data/processed/human_ppi_2026_03_v0.3/mmseqs30 \
  8
```

Use `mmseqs30/proteins.clustered.csv` as the protein table for split generation.

```bash
python3 protenix_ppi/scripts/make_c3_split.py \
  --proteins <DATASET_DIR>/proteins.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --output-dir <DATASET_DIR>/splits/c3_primary
```

The predeclared algorithm is documented in `protocol/c3_split_protocol_v0.1.md`. It uses 60/20/20 homology-cluster pools to target an approximately 80/10/10 retained-pair split, searches a fixed seed range before model scoring, and explicitly quarantines cross-pool pairs.

## Evaluation engine

Metric definitions, prevalence controls, and uncertainty rules are frozen in `protocol/evaluation_protocol_v0.2.md`. For the primary v0.3 dataset, use the one-command route after MMseqs2 finishes. It validates the MMseqs2 version, fixed parameters, FASTA/cluster hashes, unchanged protein records, C3 leakage invariants, complex-group quarantine, and cohort membership before writing the final freeze manifest:

```bash
python3 -m protenix_ppi.scripts.finalize_c3_benchmark \
  --dataset-dir protenix_ppi/data/processed/human_ppi_2026_03_v0.3
```

The command refuses to run when final C3 outputs already exist. Do not remove that guard or regenerate membership after inspecting a model score. The manual cohort command below remains available for debugging:

```bash
python3 -m protenix_ppi.scripts.build_evaluation_cohorts \
  --pairs protenix_ppi/data/processed/human_ppi_2026_03_v0.3/pairs.csv \
  --splits protenix_ppi/data/processed/human_ppi_2026_03_v0.3/splits/c3_primary/splits.csv \
  --output-dir protenix_ppi/data/processed/human_ppi_2026_03_v0.3/evaluation_cohorts \
  --seed 20260915
```

Evaluate one prediction file, optionally against an aligned baseline, on the primary fixed cohort with:

```bash
python3 protenix_ppi/scripts/evaluate_predictions.py \
  --predictions <METHOD_PREDICTIONS.csv> \
  --baseline <BASELINE_PREDICTIONS.csv> \
  --partition test \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --cohort balanced_explicit_1to1 \
  --output <RESULTS_DIR>/evaluation.json
```

The evaluator reports tie-aware average precision, trapezoidal PR-AUC as a distinct metric, AUROC, Precision/Recall@K, EF@1%, Brier score, ECE, negative-stratum results, and paired group-bootstrap uncertainty. Full-pool, curated-negative, and screen-negative cohorts remain mandatory secondary results; balanced-cohort precision is not interpreted as proteome-wide positive predictive value.

Aggregate an exact 42/123/999 seed comparison with:

```bash
python3 -m protenix_ppi.scripts.aggregate_multiseed \
  --manifest <RESULTS_DIR>/multiseed_comparison_manifest.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --cohort balanced_explicit_1to1 \
  --output <RESULTS_DIR>/multiseed_balanced_explicit_1to1.json
```

Use `configs/multiseed_comparison_manifest.template.csv`. The aggregator requires all three registered seeds and matching prediction provenance; it never selects the best test seed.

For a final registered B0b/B2/B3 comparison, prefer the four-cohort suite. It
binds results to the G2 freeze manifest and applies the predeclared conservative
three-seed support rule:

```bash
python3 -m protenix_ppi.scripts.evaluate_comparison_suite \
  --manifest <RESULTS_DIR>/multiseed_comparison_manifest.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --benchmark-freeze-manifest <DATASET_DIR>/benchmark_freeze_manifest.json \
  --comparison b2_vs_b0b \
  --output-dir <RESULTS_DIR>/comparisons/b2_vs_b0b
```

## B1 Protenix zero-shot adapter

The locked zero-shot policy is specified in `protocol/b1_zero_shot_protocol_v0.1.md`. It uses native `iptm` from Protenix rank 0 as the primary score and reports native `ranking_score` separately. It never fits a score combination or selects a sample using PPI labels.

After a real pinned Protenix run, convert one complete seed's outputs with:

```bash
python3 -m protenix_ppi.scripts.make_b1_predictions \
  --proteins <DATASET_DIR>/proteins.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --runs <B1_SEED_DIR>/b1_runs.csv \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json \
  --condition-lock <B1_SEED_DIR>/condition_lock.json \
  --output-dir <RESULTS_DIR>/B1/seed_42
```

Templates are available at `configs/b1_condition_lock.template.json` and `configs/b1_runs.template.csv`. The adapter checks exact test-set coverage, two-protein-chain inputs, sequence identity, native ranks 0–4, rank-0 sorting, model/cutoff locks, and all source hashes before writing standard test-only predictions and a per-pair provenance manifest.

Create the exact tied null reference for seed 42 (repeat for 123 and 999), then
use comparison identifier `b1_vs_constant` in the four-cohort comparison suite:

```bash
python3 -m protenix_ppi.scripts.make_constant_reference \
  --pairs <DATASET_DIR>/pairs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --benchmark-freeze-manifest <DATASET_DIR>/benchmark_freeze_manifest.json \
  --output-dir <RESULTS_DIR>/C0/seed_42 \
  --seed 42
```

The reference assigns 0.5 to every test pair. With tie-aware AP it is exactly
the cohort-prevalence baseline and cannot gain or lose from arbitrary row order.

## B0a pipeline baseline

The symmetric amino-acid-composition sanity baseline is specified in `protocol/baseline_protocol_v0.1.md` and trained with:

```bash
python3 -m protenix_ppi.scripts.train_b0a \
  --proteins <DATASET_DIR>/mmseqs30/proteins.clustered.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --output-dir <RESULTS_DIR>/B0a/seed_42 \
  --seed 42
```

B0a validates the end-to-end data/split/model/prediction path. A frozen protein-language-model baseline (B0b) is still required before claiming that Protenix adds value beyond sequence information.

## B0b frozen protein-language-model baseline

The ESM-2 embedding policy is specified in `protocol/esm_baseline_protocol_v0.1.md`. Generate the primary frozen embedding bundle on a suitable GPU server (start with batch size 1 and float32 compute):

```bash
python3 -m protenix_ppi.scripts.extract_esm2_embeddings \
  --proteins <DATASET_DIR>/mmseqs30/proteins.clustered.csv \
  --output-dir <EMBEDDING_BUNDLE> \
  --model-id facebook/esm2_t33_650M_UR50D \
  --revision 08e4846e537177426273712802403f7ba8261b6c \
  --device cuda \
  --compute-dtype float32 \
  --batch-size 1
```

The resolved immutable model revision, checkpoint hashes, window policy, software versions, timing, and peak allocated GPU memory are written into the bundle metadata. No label or split assignment is read during extraction.

Validate the resulting offline bundle with:

```bash
python3 -m protenix_ppi.scripts.embedding_bundle \
  --bundle-dir <EMBEDDING_BUNDLE> \
  --proteins <DATASET_DIR>/proteins.csv
```

Train the controlled symmetric classifier with:

```bash
python3 -m protenix_ppi.scripts.train_b0b \
  --proteins <DATASET_DIR>/mmseqs30/proteins.clustered.csv \
  --pairs <DATASET_DIR>/pairs.csv \
  --evidence <DATASET_DIR>/evidence.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --embedding-bundle <EMBEDDING_BUNDLE> \
  --output-dir <RESULTS_DIR>/B0b/seed_42 \
  --seed 42
```

The bundle contract binds every vector to an exact accession and sequence hash and records the immutable encoder revision and checkpoint digests.

## B2 frozen Protenix representation baseline

The B2 contract is specified in `protocol/b2_frozen_protenix_protocol_v0.1.md`. The exact internal Protenix tensor hook must be filled only after G1 pins and inspects the executed source commit; the current templates reject placeholders and label-aware or non-symmetric feature policies.

Validate a server-produced feature bundle with:

```bash
python3 -m protenix_ppi.scripts.pair_feature_bundle \
  --bundle-dir <B2_FEATURE_BUNDLE> \
  --pairs <DATASET_DIR>/pairs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --backbone-lock <G1_RUN_DIR>/backbone_lock.json
```

Before that bundle validation, validate its evidence-bound hook lock with
`scripts/validate_b2_hook_lock.py` as specified in
`protocol/b2_frozen_protenix_protocol_v0.1.md`. This confirms that the captured
feature is the final-recycle Pairformer `z` tensor from the same G1 commit and
condition, rather than a placeholder or a native scalar score. The resulting
`b2_hook_validation.json` is a mandatory, hash-bound bundle member; both B2 and
B3 fail closed if it is missing or inconsistent.

Train the controlled frozen-feature classifier with:

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

B2 is compared against B0b on identical frozen C3 cohorts. It does not update any Protenix parameter and cannot by itself authorize structural fine-tuning.

## B3 nonlinear PPI task head

B3 is specified in `protocol/b3_ppi_head_protocol_v0.1.md`. It trains a fixed 64-unit nonlinear head on the same frozen feature bundle as B2; the optimizer contains no Protenix parameter.

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

The upstream native `train_confidence_only` path remains a structural-confidence training path, not a binary PPI-label solution. B3 must beat B2 before any structural-module experiment is considered.

## Structural preservation for optional B4/B5

The independent-set and non-inferiority rules are frozen in `protocol/structural_preservation_protocol_v0.1.md`. Populate the supplied set/metric templates and evaluate with:

```bash
python3 -m protenix_ppi.scripts.evaluate_structure_preservation \
  --structural-set <STRUCTURE_SET>/structural_set.csv \
  --metrics <RESULTS_DIR>/structural_preservation_metrics.csv \
  --output <RESULTS_DIR>/structural_preservation_evaluation.json
```

Structural tuning fails this gate if the paired DockQ or acceptable-complex confidence bound exceeds the allowed degradation, or if it introduces any severe geometry failure.

To build a traceable pool of post-cutoff structural candidates before manual
review, use the RCSB-only acquisition script. It scans the complete eligible
assembly universe, then selects identifiers by a fixed SHA-256 seed rather than
by release order or any Protenix score. The script records raw RCSB JSON,
biological-assembly mmCIF hashes, exclusion reasons, and the exact query:

```bash
python3 -m protenix_ppi.scripts.fetch_structural_candidates \
  --output-dir <STRUCTURE_CANDIDATES>/rcsb_postcutoff_human_v0.1_20260915 \
  --search-limit 400 \
  --selection-seed 20260915 \
  --workers 8
```

The resulting `structural_candidates.csv` is provisional. It is not the final
30-complex preservation set until biological-assembly review, UniProt/chain
mapping review, and Linux MMseqs2 homology audit pass. For a connectivity smoke
test only, `--universe-limit` may cap the RCSB search universe; that output must
not be used as the final set.

After manual review, freeze the selected set with:

```bash
python3 -m protenix_ppi.scripts.freeze_structural_set \
  --candidates <STRUCTURE_CANDIDATES>/structural_candidates.csv \
  --review <STRUCTURE_CANDIDATES>/structural_review.csv \
  --output-dir <STRUCTURE_SET>/frozen_v0.1
```

Start the review table from `configs/structural_review.template.csv`. The
freezer requires at least 30 included candidates and writes `structural_set.csv`
with `homology_audit_passed=false`; this is intentional. Run the independent
Linux homology audit afterward and only then populate the audit fields used by
G4.

On Linux, perform that audit against the actual C3 training protein pool with:

```bash
python3 -m protenix_ppi.scripts.audit_structural_homology \
  --structural-set <STRUCTURE_SET>/frozen_v0.1/structural_set.csv \
  --ppi-proteins protenix_ppi/data/processed/human_ppi_2026_03_v0.3/mmseqs30/proteins.clustered.csv \
  --protein-pool-assignments protenix_ppi/data/processed/human_ppi_2026_03_v0.3/splits/c3_primary/protein_pool_assignments.csv \
  --benchmark-freeze-manifest protenix_ppi/data/processed/human_ppi_2026_03_v0.3/benchmark_freeze_manifest.json \
  --output-dir <STRUCTURE_SET>/homology_audit_v0.1 \
  --threads 8
```

The audit uses the same 30% identity/50% coverage rule as C3, preserves exact
MMseqs2 command/version/log evidence, and writes
`structural_homology_audit.json` plus an immutable
`structural_set_audited.csv`. Use the latter as the `structural_set` input to
G4; its audit flags are populated from this exact run. Every
`independence_passed` flag must be true before G4 can open.

Before any B4/B5 run, evaluate the separate G4 authorization gate documented in
`protocol/g4_structural_tuning_gate_v0.1.md`. It requires complete B0--B3
three-seed suites, supported B2-over-B0b and B3-over-B2 comparisons, a reviewed
and hashed scientific limitation diagnosis, at least 30 independently audited
post-cutoff complexes, a passing real G1 bundle, and a successful capacity probe
for the exact proposed backward graph:

```bash
python3 -m protenix_ppi.scripts.evaluate_g4_gate \
  --decision-manifest <G4_DIR>/g4_decision_manifest.json \
  --output <G4_DIR>/g4_evaluation.json
```

Templates are `configs/g4_decision_manifest.template.json`,
`configs/structural_homology_audit.template.json`, and
`configs/g4_backward_capacity_report.template.json`. A passing result authorizes
only the named proposal; it is not a B4/B5 efficacy or preservation result.

## Engineering cost accounting

The measurement rules and ledger definitions are frozen in `protocol/engineering_cost_protocol_v0.1.md`. Populate `configs/pair_costs.template.csv` and `configs/run_costs.template.csv`, retaining failed pairs and runs, then execute:

```bash
python3 -m protenix_ppi.scripts.summarize_engineering_costs \
  --pair-costs <RESULTS_DIR>/pair_costs.csv \
  --run-costs <RESULTS_DIR>/run_costs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --partition test \
  --output <RESULTS_DIR>/engineering_cost_summary.json
```

The report verifies C3 pair coverage and registered seeds before reporting throughput, peak memory, GPU-hours, preprocessing time, disk output, and failures.

## Do this now

Run the environment collector on the machine that will actually execute Protenix.

Windows PowerShell:

```powershell
powershell -NoProfile -File .\protenix_ppi\scripts\collect_env.ps1
```

Linux or a Slurm login node:

```bash
bash protenix_ppi/scripts/collect_env.sh
```

The report is written under `protenix_ppi/artifacts/env/`. Review it and fill in any unknown values in `protocol/environment_card.md`. Do not post hostnames, usernames, IP addresses, mount credentials, or scheduler account names when sharing the report.

Classify a collected report with the local, dependency-free evaluator:

```powershell
python .\protenix_ppi\scripts\evaluate_env.py .\protenix_ppi\artifacts\env\environment_windows_YYYYMMDD_HHMMSS.txt
```

or on Linux:

```bash
python3 protenix_ppi/scripts/evaluate_env.py protenix_ppi/artifacts/env/environment_linux_YYYYMMDD_HHMMSS.txt
```

The A/B/C/D result is only a hardware route classification. G1 still requires real Protenix inference and backward-pass evidence.

The prepared G1 protocol and evidence templates are available at:

- `protocol/gate_1_plan.md`
- `configs/backbone_lock.template.json`
- `configs/g1_manifest.template.json`
- `configs/g1_gradient_summary.template.json`

After a real G1 run, validate its evidence bundle with:

```bash
python3 protenix_ppi/scripts/validate_g1_artifacts.py <G1_RUN_DIRECTORY>
```

## Source documents

- `../AF3微调PPI调研方案.md`
- `../S04_杨信_PPI_AF3类轻量微调_前期调研说明_学生版.md`
- Official Protenix repository: <https://github.com/bytedance/Protenix>
- Official training/inference guide: <https://github.com/bytedance/Protenix/blob/main/docs/training_inference_instructions.md>
