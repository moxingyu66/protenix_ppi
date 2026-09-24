# G4 structural-tuning authorization gate v0.1

Status: executable evidence gate implemented; no real G4 decision has been made

## Purpose

G4 is a permission gate, not a model result. It prevents B4/B5 structural-module
tuning from starting merely because it is technically possible. A passing G4
decision authorizes exactly one evidence-bound B4 or B5 proposal. It does not
prove that the proposed method improves PPI screening or preserves structure.

The gate must be evaluated before inspecting any output from the proposed B4/B5
condition. A failed decision may be re-evaluated after collecting missing
pre-tuning evidence, but the frozen C3 benchmark and completed comparison suites
must not be regenerated or selected retrospectively.

## Required evidence

### 1. Frozen benchmark and completed B0--B3 ladder

The decision must bind by SHA-256 to a G2 manifest with status
`frozen_before_model_scoring` and a passing C3 validation. It must also bind to
the complete four-cohort, three-seed comparison suites for:

- B0b versus B0a;
- B1 versus the constant tied-score reference;
- B2 versus B0b;
- B3 versus B2.

All four suites must use seeds `42`, `123`, and `999`, contain the four frozen
evaluation cohorts, and reference the same benchmark-freeze manifest. Completion
does not mean every exploratory hypothesis is supported. However, structural
tuning is permitted only when the conservative primary support rule passes for
both B2 versus B0b and B3 versus B2.

### 2. Diagnosed B3 scientific limitation

An engineering inconvenience, slow runtime, missing script, or desire for a
higher score is not a scientific rationale for structural tuning. The decision
manifest must record:

- one registered limitation category and affected frozen cohort;
- a reproducible observed error pattern;
- a testable mechanistic hypothesis;
- why the frozen B3 head cannot address it;
- the specific structural mechanism B4/B5 is intended to test;
- a hashed error-analysis artifact;
- a dated human review with role and approval status;
- explicit confirmation that no B4/B5 output informed the diagnosis.

Allowed categories are `interface_representation_bottleneck`,
`ranking_error_subgroup`, `calibration_failure`, `long_range_pair_context`, and
`other_scientific`. The last category still requires the same concrete evidence.

### 3. Independent structural-preservation set

Before B4/B5, freeze at least 30 experimentally determined heteromeric complexes
released after `2021-09-30`. Every reference structure must exist and match its
recorded SHA-256. Every row must pass the protein and 30%-identity homology audit
against the C3 training partition and must have no PPI-training overlap.

The boolean columns in `structural_set.csv` are not sufficient by themselves.
A separate hashed homology-audit JSON must:

- bind to both the structural-set and benchmark-freeze SHA-256;
- record tool name, exact version, and a hashed command/log artifact;
- use the frozen 30% identity and 50% coverage thresholds;
- cover every structural complex exactly once;
- contain no protein or homology-cluster overlap flag;
- confirm that B4/B5 outputs were not consulted.

This G4 check establishes that the preservation set is ready. Actual DockQ,
iRMSD, lRMSD, interface-contact, and geometry results are evaluated only after a
structural candidate exists, using `structural_preservation_protocol_v0.1.md`.

The recommended acquisition path is `fetch_structural_candidates.py`, followed
by `freeze_structural_set.py` after manual review. The latter deliberately emits
`homology_audit_passed=false` and a `frozen_for_homology_audit_only` status, so a
manually selected set cannot accidentally pass G4 before the independent Linux
audit is attached.

The independent audit is executable with
`scripts/audit_structural_homology.py`. It builds a query FASTA from the exact
ATOM chains in each selected assembly, builds a target FASTA from the C3 train
protein pool, runs MMseqs2 `easy-search` with the frozen identity/coverage
parameters, and records direct accession overlap separately from sequence
homology overlap. Both overlap flags must be false for every complex. The audit
JSON is the `homology_audit` input expected by the G4 decision manifest, and its
`structural_set_audited.csv` output is the corresponding `structural_set` input.
The original manually frozen set remains unchanged for provenance.

### 4. Real G1 and proposed-graph capacity evidence

The referenced G1 run must pass `validate_g1_artifacts.py`, including a real
forward, backward, optimizer step, finite nonzero intended gradients, no frozen
gradients, peak-memory telemetry, a predicted structure, and confidence output.

G1 alone does not establish that a specific B4/B5 graph fits. A separate hashed
capacity report must use the same pinned source commit, checkpoint, and primary
model. It must record the exact proposed graph, trainable-parameter evidence,
representative input size, microbatch and accumulation settings, target GPU,
finite loss, completed forward/backward/optimizer step, gradient isolation,
peak/available memory, elapsed time, and absence of OOM. Exact command,
stdout/stderr, and GPU telemetry files are mandatory hashed artifacts. The probe
input must meet the registered length target for the proposed experiment.

## Templates

- `configs/g4_decision_manifest.template.json`
- `configs/structural_homology_audit.template.json`
- `configs/g4_backward_capacity_report.template.json`
- `configs/structural_preservation_set.template.csv`

Template defaults are deliberately non-passing. Do not change `false` to `true`
until the referenced evidence exists.

## Evaluation command

```bash
python3 -m protenix_ppi.scripts.evaluate_g4_gate \
  --decision-manifest <G4_DIR>/g4_decision_manifest.json \
  --output <G4_DIR>/g4_evaluation.json
```

Exit code `0` means the named B4/B5 proposal is authorized. Exit code `2` means
the evidence is valid enough to evaluate but at least one scientific gate failed.
Malformed, missing, placeholder, or hash-mismatched evidence is an input error.

Never describe a passing G4 result as B4/B5 success. B4/B5 must subsequently
beat B3 under the registered PPI rule, pass structural non-inferiority, complete
three seeds, and report engineering costs.
