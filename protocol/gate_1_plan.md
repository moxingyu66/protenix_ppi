# Gate G1 plan — pinned Protenix execution evidence

Status: prepared, not executed  
Prerequisite: G0 server report and route decision

## Purpose

G1 proves that one exact Protenix source revision and checkpoint can run in the selected environment, produce inspectable complex/confidence outputs, and support the intended gradient path. Installation success alone is not a pass.

## Fixed primary model

- Model: `protenix_base_default_v1.0.0`
- Declared training cutoff: `2021-09-30`
- Default study seed for pipeline debugging: `101`
- Initial smoke-test policy: no template and no MSA, followed by a separately recorded MSA-enabled condition
- Logging policy: local files only; Weights & Biases remains disabled during G1

The exact Git commit and checkpoint SHA-256 are deliberately left unfilled until installation on the real server. A moving `main` branch or filename alone is not a reproducible version.

## G1.1 — source and checkpoint lock

Record all of the following in `backbone_lock.json`:

1. repository URL;
2. 40-character Git commit;
3. installed Protenix package version;
4. Python, PyTorch, CUDA runtime, driver, and kernel choices;
5. checkpoint path and SHA-256;
6. model name and declared cutoff;
7. execution route selected by G0.

Do not use `protenix_base_20250630_v1.0.0` in the primary experiment.

## G1.2 — official-format smoke test

Use the bundled official input first to validate JSON parsing, checkpoint loading, device placement, diffusion sampling, confidence-head execution, and output writing.

The currently published `examples/example_without_msa.json` is not a human heterodimer: it contains two copies of a protein plus DNA and ligand entities. Passing it verifies software plumbing only.

The official CLI pattern is:

```bash
protenix pred \
  --input examples/example_without_msa.json \
  --output_dir <G1_OUTPUT_DIR>/official_format \
  --model_name protenix_base_default_v1.0.0 \
  --seeds 101 \
  --use_msa false \
  --use_template false \
  --enable_cache true
```

Before execution, confirm exact flag names with `protenix pred -h` from the pinned installation. The upstream README uses both long and short aliases across examples; the help output from the installed revision is authoritative.

## G1.3 — PPI-relevant heterodimer smoke test

Run a second condition on a known human heterodimer with an experimental complex structure. The working target is CDK2–Cyclin A2, PDB `1FIN`, because it is a well-characterized direct complex and is suitable for checking interface outputs.

Required preparation:

1. obtain the RCSB mmCIF for 1FIN with source URL and retrieval date recorded;
2. identify the biological assembly rather than treating arbitrary crystal contacts as the target;
3. convert it with the pinned `protenix json` command;
4. inspect that the resulting input contains two distinct protein chains;
5. save the generated JSON and SHA-256 before inference.

Run two separately named conditions:

- `1fin_no_msa_no_template`: plumbing and resource lower bound;
- `1fin_msa_no_template`: PPI-relevant inference condition after MSA generation is validated.

The no-MSA result cannot be used as the final B1 scientific baseline; it is only a G1 diagnostic.

## G1.4 — output integrity checks

For each inference condition, preserve:

- exact command and standard output/error;
- input JSON and SHA-256;
- predicted mmCIF/PDB structure;
- all confidence JSON files;
- seed and model configuration;
- wall-clock time;
- peak GPU memory and GPU model;
- success/failure status.

Visually plausible coordinates are not sufficient. Files must be parseable and the intended two protein chains must be present.

## G1.5 — backward-pass evidence

The official `finetune_demo.sh` currently:

- loads `protenix_base_default_v1.0.0.pt` as both model and EMA checkpoint;
- selects a PDB subset through `base_info.pdb_list`;
- uses structural training losses;
- defaults to `max_steps=100000` and therefore must not be launched unchanged as a smoke test.

G1 requires exactly one controlled optimization step before any PPI head is implemented. The run must record:

- scalar loss is finite;
- selected trainable-parameter names;
- total and trainable parameter counts;
- at least one intended parameter has a finite nonzero gradient;
- frozen parameters remain without gradients;
- optimizer step completes;
- peak GPU memory and elapsed time.

If official preprocessed training data is unavailable, do not download 1.5+ TB merely to satisfy G1. Instead, the next implementation task is a minimal, documented single-complex batch adapter. A synthetic tensor-only test may debug a module but cannot by itself prove that the full Protenix data/model path is trainable.

## G1 pass criteria

- [ ] `backbone_lock.json` is complete and validates.
- [ ] Official-format inference exits successfully.
- [ ] Human heterodimer inference exits successfully.
- [ ] At least one predicted structure file is parseable and contains the intended chains.
- [ ] Confidence output is valid JSON.
- [ ] Fixed-seed rerun produces the expected stable file schema and no unexplained configuration drift.
- [ ] One real Protenix training batch completes forward, backward, and optimizer step.
- [ ] Gradient summary proves intended trainable/frozen behavior.
- [ ] GPU telemetry and wall time are preserved.
- [ ] `validate_g1_artifacts.py` reports PASS.

Any missing item keeps G1 open. An OOM is evidence for route revision, not permission to silently reduce the scientific target.

