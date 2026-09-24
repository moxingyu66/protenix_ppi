# Engineering cost protocol v0.1

Status: cost-ledger schema and summarizer implemented; no real Protenix runtime has been measured

## Purpose

Screening quality is insufficient if a method cannot be run at the intended candidate-pool scale. Every final B0--B5 comparison must therefore preserve failures as well as successful timings and report hardware-specific resource use.

## Two ledgers

### Per-pair ledger

Use `configs/pair_costs.template.csv` for Protenix inference and pair-level feature extraction. Record one row for every attempted C3 test pair, including failures:

- method, registered seed, and stage;
- pair ID and partition;
- success/failure;
- GPU count and exact GPU model;
- peak GPU memory;
- wall time and preprocessing time;
- newly attributable output bytes;
- total input residues;
- input, command, and stdout/stderr SHA-256 digests.

Do not omit OOM, timeout, preprocessing, MSA, or malformed-output failures. Shared checkpoint/MSA caches are counted once in the run ledger, not repeatedly as per-pair output.

### Run ledger

Use `configs/run_costs.template.csv` for whole training runs, shared preprocessing, embedding extraction, and one-time cache construction. Seed-independent work uses `NA`; model training uses exactly 42, 123, and 999.

## Measurement rules

- Reset framework peak-memory counters immediately before the measured stage.
- Record both allocated and reserved GPU memory in raw logs; place the larger justified peak in the summary ledger.
- Preserve `nvidia-smi` telemetry for detecting other GPU users and allocation drift.
- Wall time begins before input/model preparation included in the named stage and stops only after outputs are durably written.
- Report MSA generation separately from cached-model inference.
- Never compare timings from different hardware as if they were method-only differences.
- Warm-cache and cold-cache conditions are separate stages.

## Summary command

```bash
python3 -m protenix_ppi.scripts.summarize_engineering_costs \
  --pair-costs <RESULTS_DIR>/pair_costs.csv \
  --run-costs <RESULTS_DIR>/run_costs.csv \
  --splits <DATASET_DIR>/splits/c3_primary/splits.csv \
  --partition test \
  --output <RESULTS_DIR>/engineering_cost_summary.json
```

The summarizer reports mean, median, 95th percentile, maximum, total wall time, preprocessing time, GPU-hours, peak memory, output bytes, throughput, GPU models, length range, failures, and missing/unexpected pair IDs.

`experiment_execution_complete=true` requires:

- every reported pair-level method/stage to attempt and successfully complete every C3 test pair;
- exact 42/123/999 coverage for every seeded method/stage;
- no failed whole-run entry.

GPU-hours are wall time multiplied by allocated GPU count. They are not electricity, cloud price, or carbon cost. Monetary estimates require a separately recorded rate and billing model.
