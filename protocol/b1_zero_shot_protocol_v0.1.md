# B1 Protenix zero-shot protocol v0.1

Status: adapter implemented and synthetically tested; no biological Protenix output evaluated

## 1. Purpose

B1 measures the native interaction-ranking signal of the pinned Protenix model without fitting any parameter, threshold, feature combination, or sample-selection rule to PPI labels. It is a baseline, not evidence that `ipTM` is a binding probability.

## 2. Locked backbone and primary condition

- Model: `protenix_base_default_v1.0.0` (368M)
- Declared training cutoff: `2021-09-30`
- Source commit and checkpoint SHA-256: inherited from the validated G1 `backbone_lock.json`
- Input: exactly two protein chains matching the frozen canonical benchmark sequences; no DNA, RNA, ligand, or extra chain
- MSA: enabled
- Template: disabled
- `msa_pair_as_unpair`: `true`
- Pairformer cycles: 10
- Diffusion steps: 200
- Samples per seed: 5
- Data type: BF16
- Official default parameters: enabled
- TF32, efficient fusion, and diffusion shared-variable cache: enabled
- Native sorting: `sorted_by_ranking_score=true`
- Seeds: 42, 123, and 999, processed and reported separately

The no-MSA condition is a G1 diagnostic or a separately named sensitivity analysis. It must not silently replace primary B1.

## 3. Frozen score and sample-selection rule

For each pair and seed:

1. Use the file with native rank `0` after Protenix sorting.
2. Use its native `iptm` as the primary B1 PPI score.
3. Report its native `ranking_score` as a separate comparator.
4. Do not select a diffusion sample using the PPI label or the largest `iptm`.
5. Do not fit or tune a combination of `iptm`, `ptm`, pLDDT, PAE, clash, or other fields in B1.

The current upstream native sorting score is:

```text
ranking_score = 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100 * has_clash
```

This formula is upstream model behavior, not a project-fitted PPI score. The adapter verifies that rank 0 has the maximum emitted `ranking_score` across the five summaries.

The upstream filename schema, fields, and ranking formula were checked against official Protenix commit `4c355be4553512f72453ecbfb65e69f4c35d1413` (retrieved 2026-09-15), specifically `runner/dumper.py`, `configs/configs_inference.py`, and `protenix/model/sample_confidence.py`. The actual experimental source commit remains the one recorded by G1 and need not equal this schema-audit commit.

## 4. Required upstream files

Each frozen test pair must have:

- the exact Protenix input JSON and its pre-recorded SHA-256;
- five summary files named `<sample_name>_summary_confidence_sample_<rank>.json` for ranks 0–4;
- a non-empty rank-0 structure named `<sample_name>_sample_0.cif`;
- a row in `b1_runs.csv` binding the pair ID, sample name, input file, input hash, and predictions directory;
- one condition lock and one validated backbone lock shared by the seed run.

Relative paths in `b1_runs.csv` are resolved relative to that CSV file.

## 5. Adapter and outputs

Run once per seed/condition directory:

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

The adapter writes:

- `predictions.csv`: test-only primary `iptm` predictions;
- `predictions_ranking_score.csv`: test-only native-ranking comparator;
- `source_manifest.csv`: per-pair input, summary, structure, checkpoint, commit, seed, rank, and SHA-256 provenance;
- `metadata.json`: condition, policy, dataset hashes, and output hashes.

The run fails rather than silently continuing if the test set is incomplete, an input sequence differs, a non-protein entity is present, the model/cutoff is wrong, a rank is missing, rank 0 is not the native best, or any required hash differs.

## 6. Evaluation and interpretation

Evaluate each seed on the identical frozen C3 test set with `evaluate_predictions.py`. The primary endpoint is average precision. `ranking_score` is a named comparator and cannot replace the primary result after inspection.

Generate the exact tied null reference once for each registered metadata seed:

```bash
python3 -m protenix_ppi.scripts.make_constant_reference \
  --pairs <DATASET_DIR>/pairs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --cohort-membership <DATASET_DIR>/evaluation_cohorts/cohort_membership.csv \
  --benchmark-freeze-manifest <DATASET_DIR>/benchmark_freeze_manifest.json \
  --output-dir <RESULTS_DIR>/C0/seed_42 \
  --seed 42
```

Repeat for seeds 123 and 999. All scores are exactly 0.5 and the seed does not
affect predictions; separate metadata files exist only to align the registered
three-seed comparison. Under the tie-aware AP definition this reference equals
the evaluated cohort prevalence. It is preferred to a single arbitrary random
permutation, which adds avoidable Monte Carlo noise.

After building the five-column comparison manifest, run `b1_vs_constant` with
`evaluate_comparison_suite.py`. The suite verifies the original (unclustered)
protein-table hash used by B1, native rank-0 `iptm` policy, absence of label
fitting, literal 0.5 null scores, all registered seeds, and all four cohorts.

Although `iptm` lies in `[0,1]`, it is not trained as a PPI probability. Any Brier/ECE result is therefore a raw-score diagnostic. Probability calibration belongs to B3 and must be fitted on validation data only.

No B1 biological claim is allowed until all test pairs have real Protenix outputs from a G1-passed environment and the resulting prediction files are evaluated.

## 7. Acceptance criteria

- [ ] G0 and G1 have passed on the actual server.
- [ ] Backbone and condition locks validate.
- [ ] All C3 test pairs have exactly one run-manifest row per seed.
- [ ] Inputs contain exactly the two frozen benchmark protein chains.
- [ ] Five confidence summaries and a rank-0 structure exist for every pair.
- [ ] Adapter completes without hash, schema, or ranking errors.
- [ ] Seeds 42, 123, and 999 are preserved and evaluated separately.
- [ ] Primary `iptm` and comparator `ranking_score` are reported without label fitting.
- [ ] `b1_vs_constant` is evaluated against the exact tied 0.5 reference on all four cohorts.
- [ ] Runtime, peak memory, output failures, and disk footprint are reported.
