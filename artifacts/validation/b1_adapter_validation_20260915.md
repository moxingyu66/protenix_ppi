# B1 zero-shot adapter validation — 2026-09-15

## Scope

This record validates the local conversion and provenance logic only. It does not validate Protenix inference quality and contains no biological result.

The upstream output schema was inspected at official Protenix commit `4c355be4553512f72453ecbfb65e69f4c35d1413`:

- `runner/dumper.py`: prediction directory, native rank filenames, and sorting behavior;
- `protenix/model/sample_confidence.py`: summary fields and native ranking formula;
- `configs/configs_inference.py`: default model, seed, MSA, template, and sorting configuration;
- `inference_demo.sh`: CLI flags and defaults for cycles, diffusion steps, samples, and BF16.

## Frozen local behavior

- primary B1 score: native `iptm` from native rank 0;
- comparator: native `ranking_score` from the same rank-0 sample;
- primary condition: MSA on, template off, 10 cycles, 200 diffusion steps, 5 samples, BF16;
- no label-fitted combination and no label-driven sample selection;
- exact test coverage and two-protein-chain canonical-sequence matching required;
- source commit, checkpoint, condition, input, summary, and structure digests preserved.

## Verification commands

```powershell
python -m compileall -q protenix_ppi
python -W error::ResourceWarning -m unittest discover -s protenix_ppi\tests -v
python -m protenix_ppi.scripts.make_b1_predictions --help
```

Results:

- compilation: PASS;
- full unit suite: PASS, 39 tests;
- B1-specific tests: PASS, 6 tests;
- CLI help/argument parser: PASS.

The B1 tests establish that the adapter selects native rank 0 even when another sample has a larger `iptm`, and rejects input-hash mismatch, non-protein entities, canonical-sequence mismatch, invalid native ranking, and the later-cutoff model.

## Remaining external evidence

G0 and G1 remain open until a suitable Linux GPU environment provides a real environment report, pinned checkpoint/source lock, two-chain inference, confidence files, structures, telemetry, and one real backward step. Synthetic tests cannot satisfy those gates.
