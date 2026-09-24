# Structural preservation protocol v0.1

Status: evaluator, templates, and non-inferiority gate implemented; independent biological structure set not yet constructed

## Scope

This protocol is required only for a method that changes Protenix structural parameters or generated coordinates, such as B4/B5. B0--B3 do not update the backbone and cannot claim improved structure prediction.

## Independent structure set

Before structural tuning starts, freeze at least 30 experimentally determined heteromeric complexes satisfying all of the following:

- PDB release date is later than the primary Protenix cutoff `2021-09-30`;
- the selected biological assembly, chain mapping, and reference file hash are recorded;
- sequence/homology exposure is audited according to the frozen temporal-holdout rule;
- no complex or homologous group overlaps the PPI training partition;
- selection does not use B4/B5 outputs.

The reference implementation for the independence check is
`scripts/audit_structural_homology.py`. It uses the C3
`protein_pool_assignments.csv` rather than inferring the training pool from pair
edges, so isolated train-pool proteins are included in the audit. It emits a
hashed JSON report with direct accession overlap and sequence-homology overlap
flags for every selected complex.

The set contract is `configs/structural_preservation_set.template.csv`. A boolean audit field is not a substitute for retaining the underlying homology report; the final release must include that report and command/version hashes.

Candidate acquisition is implemented in
`scripts/fetch_structural_candidates.py`. Its frozen RCSB search requires a
post-cutoff experimental heteromeric protein assembly with exactly two protein
instances, human source records, a single UniProt accession per chain, a
resolution at most 4 Å, at least 20 interface residues, and at least 500 Å²
buried surface area. Antibody-like or otherwise ambiguous constructs are
excluded automatically. A complete search universe is scanned, while the
requested candidate count is selected with SHA-256(`selection_seed:assembly_id`)
ordering. The output is explicitly provisional and still requires manual review
and homology audit.

Each row also records the uppercase PDB ID, numeric biological-assembly ID,
JSON mapping from assembly protein-chain IDs to distinct protein accessions, the
matching HTTPS RCSB assembly URL, retrieval timestamp, and fixed selection-rule
version `rcsb_human_heteromer_v0.1`. The validator rejects asymmetric-unit URLs,
unmapped or homomeric chain sets, mappings that do not exactly cover the ATOM
chains in the downloaded assembly mmCIF, and files whose SHA-256 does not match.
Chain keys are mmCIF `label_asym_id` values. This prevents an arbitrary crystal
contact or a mismatched chain table from being treated as the biological PPI.

## Required per-complex measurements

For the identical reference assembly and seed, compare the frozen B3-era backbone with the candidate structurally tuned model:

- DockQ;
- interface RMSD;
- ligand RMSD;
- interface-contact precision and recall;
- chain collapse;
- atomic overlap;
- within-chain geometry failure.

Both predicted structure files and their SHA-256 digests are mandatory. DockQ and geometry-tool versions, commands, chain mappings, failures, and raw outputs must be preserved outside the summary CSV.

## Non-inferiority gate

The candidate passes structural preservation only if all checks pass:

- mean `candidate DockQ - baseline DockQ >= -0.05`;
- paired group-bootstrap 95% lower bound for the DockQ delta is at least `-0.05`;
- acceptable-complex fraction uses `DockQ >= 0.23`;
- acceptable-fraction point delta is at least `-0.10`;
- its paired-bootstrap 95% lower bound is at least `-0.10`;
- there are zero new chain-collapse, atomic-overlap, or within-chain geometry failures relative to baseline.

Bootstrap uses 2,000 replicates and seed `20260915`, resampling stable dependence groups.

## Evaluation command

Populate `configs/structural_preservation_metrics.template.csv` from retained raw tool outputs, then run:

```bash
python3 -m protenix_ppi.scripts.evaluate_structure_preservation \
  --structural-set <STRUCTURE_SET>/structural_set.csv \
  --metrics <RESULTS_DIR>/structural_preservation_metrics.csv \
  --output <RESULTS_DIR>/structural_preservation_evaluation.json
```

Exit code 0 means every non-inferiority and severe-failure check passed. Exit code 2 means a scientifically valid evaluation ran but the candidate failed the preservation gate. Input/schema/provenance errors fail separately.

Passing this gate is necessary but not sufficient for B4/B5 success: the method must also meet the predeclared PPI screening improvement threshold and cost constraints.
