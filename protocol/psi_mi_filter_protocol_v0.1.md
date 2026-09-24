# PSI-MI direct-PPI filtering protocol v0.1

Status: policy compiled against the pinned ontology and applied by streaming the full IntAct positive archive

## Why two ontology axes must stay separate

IntAct MITAB has separate fields for interaction type and experimental detection method. `MI:0407` means **direct interaction** on the interaction-type axis. It is not the code for a laboratory method. Two-hybrid (`MI:0018`), surface plasmon resonance (`MI:0107`), isothermal titration calorimetry (`MI:0065`), and X-ray crystallography (`MI:0114`) are detection methods.

The original broad research note occasionally described `MI:0407` as if it were a detection method. That interpretation is rejected here and in the normalized data contract.

## Pinned ontology

- Repository: `HUPO-PSI/psi-mi-CV`
- Git commit: `13b574767a81580092e7f6b62ff84c0c8f75cf69`
- OBO SHA-256: `b1315efd86a13988df97d2daefed025dbd26b98d66104da76ea3e85706534d2f`
- Parsed terms: 1,655
- License: CC BY 4.0

The human-authored rule is in `configs/psi_mi_filter_policy_v0.1.json`. The generated immutable term sets are in `artifacts/data/psi_mi_filter_compiled_v0.1.json`.

Regenerate them with:

```bash
python3 -m protenix_ppi.scripts.build_psi_mi_filter \
  --ontology "protenix_ppi/data/raw/PSI-MI controlled vocabulary/git_13b574767a81580092e7f6b62ff84c0c8f75cf69/psi-mi.obo" \
  --policy protenix_ppi/configs/psi_mi_filter_policy_v0.1.json \
  --output protenix_ppi/artifacts/data/psi_mi_filter_compiled_v0.1.json
```

## Automatic positive-candidate rule

An IntAct evidence row enters the automatic positive-candidate pool only when all of the following are true:

1. both endpoints have taxid 9606;
2. both endpoint types are protein (`MI:0326`);
3. the row is not marked negative;
4. the interaction-type field contains exactly `MI:0407`;
5. the method field is unambiguous and belongs to the compiled non-obsolete automatic set;
6. both endpoint identifiers and the source interaction identifier are present.

Automatic method roots are deliberately narrow:

- two-hybrid and its valid binary descendants;
- fluorescence polarization and FRET;
- ITC;
- NMR;
- SPR and SPR array;
- X-ray crystallography;
- bio-layer interferometry;
- equilibrium dialysis;
- microscale thermophoresis;
- mass photometry.

## Executed full-archive result

The pinned lowercase `human.zip` archive was streamed without extracting the 9.28 GB member to disk. The exact run is recorded in `data/interim/intact_2026-01-09_direct_v0.1/metadata.json`.

- MITAB evidence rows read: 1,240,727;
- automatic direct-evidence candidates: 5,064;
- manual-review rows: 65,790;
- excluded rows: 1,169,873.

The candidate table contains 2,549 unique raw primary-ID pairs. These remain evidence candidates until UniProt normalization, reciprocal canonicalization, contradiction handling, and leakage checks.

Three descendants are removed from automatic acceptance despite their ontology ancestry:

- protein three-hybrid, because a third partner may bridge the interaction;
- X-ray powder diffraction;
- X-ray fiber diffraction.

## Manual-review queue

The following are not declared negative. They are retained for review:

- descendants of `MI:0407` such as covalent or enzymatic reactions;
- `MI:0407` paired with an unlisted experimental method;
- Co-IP, pull-down, and affinity-chromatography methods;
- multi-valued or obsolete ontology terms;
- missing or non-UniProt primary endpoint identifiers;
- records requiring isoform, participant-expansion, or publication-level interpretation.

Generic physical association (`MI:0915`) and association (`MI:0914`) are excluded from the automatic direct-positive pool. This means “not eligible for the primary label,” not “biologically non-interacting.”

## Why exact `MI:0407`, not all descendants

The current `MI:0407` subtree includes 70 descendants, many of which describe enzymatic reactions, cleavage, covalent modification, or self interaction. Accepting the entire subtree would silently change the task from heterotypic physical PPI screening to a mixture of binding and biochemical reaction prediction.

## Change control

Changing any accepted root, exception, ontology commit, or decision rule creates a new policy version. It must be done before model scores are inspected, followed by rebuilding the normalized dataset and split hashes. Test performance cannot justify an in-place whitelist change.
