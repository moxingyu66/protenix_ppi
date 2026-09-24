# PPI source acquisition protocol v0.1

Status: all eight registered raw sources acquired and integrity-audited; IntAct archive stream-filtered; UniProt mapping snapshot prepared; normalized benchmark remains provisional

## 1. Principle

Raw availability is not biological eligibility. A file is first acquired under a versioned path with its URL, retrieval timestamp, license status, HTTP metadata, byte count, and SHA-256. Only later stages may map identifiers, resolve taxonomy, apply PSI-MI rules, canonicalize pairs, and decide eligibility.

No absent edge is converted to a negative. No source count from a paper or web page replaces the count observed in the pinned raw file.

## 2. Frozen source registry

The authoritative acquisition registry is `configs/source_registry_v0.1.json`. It distinguishes:

- `core`: small sources fetched by default;
- `restricted_core`: small sources with unresolved terms or transport problems;
- `large`: sources over 100 MB requiring explicit storage approval.

Normal core acquisition:

```bash
python3 -m protenix_ppi.scripts.fetch_raw_sources \
  --registry protenix_ppi/configs/source_registry_v0.1.json \
  --output-root protenix_ppi/data/raw
```

Negatome's official server currently presents an invalid TLS certificate chain. Its two pinned files may be acquired only with the explicit switch below; exact byte size and pre-recorded SHA-256 remain mandatory:

```bash
python3 -m protenix_ppi.scripts.fetch_raw_sources \
  --registry protenix_ppi/configs/source_registry_v0.1.json \
  --output-root protenix_ppi/data/raw \
  --source-id negatome2_manual_stringent \
  --source-id negatome2_supplement \
  --allow-insecure-tls
```

Do not redistribute Negatome raw files until its terms are confirmed. Internal accessibility and a download link are not a license.

## 3. Source-specific interpretation

### HuRI

- The exact portal file is a headerless two-column Ensembl gene-pair table.
- The download page reports 52,569 interactions, but the pinned file contains 52,548 rows. The file hash and observed rows are authoritative for this study.
- It contains 480 self-pairs, which are biologically meaningful homomer candidates but outside the primary heterotypic question; quarantine them rather than silently deleting them.
- Portal documentation states that mappings use GENCODE v27. Map to a pinned UniProt release explicitly; do not assume every Ensembl gene maps one-to-one to one canonical sequence.
- The download page states CC BY 4.0 and requests citation of the portal and HuRI paper.

### Negatome 2.0 manual stringent

- The primary negative anchor is `manual_stringent.txt`, not the PDB-derived or combined file.
- It has four columns: endpoint A, endpoint B, publication identifier, and experimental detection method.
- It spans multiple species and contains duplicates/reciprocals, isoform-specific accessions, and self-pairs.
- Taxonomy is absent from the file and must be resolved using a pinned UniProt mapping snapshot.
- Its 1,991 raw rows do not imply 1,991 human heterotypic canonical negatives.

### IntAct

- Release `2026-01-09` is frozen. Lowercase `human.zip` is the intended species bundle; uppercase `Human.zip` contains virus-named exports and is not interchangeable.
- A species export can contain cross-species interactions. Require taxid 9606 for both endpoints.
- Require protein type for both endpoints and resolve non-UniProt primary identifiers explicitly.
- `MI:0407` is an interaction-type term. Detection methods occupy a different MITAB column and must be filtered with a separate ontology descendant set.
- Generic physical association (`MI:0915`) or association (`MI:0914`) is not automatically a direct physical positive.
- Negative records remain stratified by provenance and experimental definition.
- EMBL-EBI states it imposes no additional restriction beyond the data owner and expects attribution; source-owner terms may still apply.

### PSI-MI ontology

- Ontology commit: `13b574767a81580092e7f6b62ff84c0c8f75cf69`.
- OBO SHA-256: `b1315efd86a13988df97d2daefed025dbd26b98d66104da76ea3e85706534d2f`.
- License: CC BY 4.0, stored with the ontology.
- Interaction-type descendants and detection-method descendants must be generated separately and frozen before IntAct positive filtering.

## 4. Current observed audit

The machine-readable and rendered reports are:

- `artifacts/data/raw_source_audit_20260915.json`
- `artifacts/data/raw_source_audit_20260915.md`

Current verified facts include:

| Source | Raw rows | Immediate exclusions/quarantine |
|---|---:|---|
| HuRI | 52,548 | 480 self-pairs; identifier mapping pending |
| Negatome manual stringent | 1,991 | 70 self-pairs, 55 isoform rows, 35 duplicate/reciprocal rows; taxonomy pending |
| IntAct human negative | 946 | 903 rows have two human protein endpoints; only 781 use UniProt as both primary IDs |
| PSI-MI OBO | 1,655 terms | descendant policy frozen and compiled; IntAct stream filter completed |

The lowercase IntAct `human.zip` archive has also been acquired and verified:

- archive bytes: 1,171,103,835;
- archive SHA-256: `4464f68a9b405bb5fddaced6e0a6b1b8e4386b65956c41cb22763c81d5337a46`;
- contained `human.txt`: 9,280,698,017 uncompressed bytes;
- streamed MITAB rows: 1,240,727;
- conservative automatic direct-evidence candidates: 5,064;
- manual-review rows: 65,790.

The PSI-MI policy is frozen in `configs/psi_mi_filter_policy_v0.1.json` and its compiled artifact. The 5,064 candidate rows are evidence candidates, not automatically accepted PPI labels.

The IntAct negative export contains 905 rows with taxid 9606 on both endpoints and 903 with both endpoints typed as proteins. It must not be treated as 946 ready-to-use human negatives.

## 5. Validation command

```bash
python3 -m protenix_ppi.scripts.audit_raw_sources \
  --registry protenix_ppi/configs/source_registry_v0.1.json \
  --raw-root protenix_ppi/data/raw \
  --output-json protenix_ppi/artifacts/data/raw_source_audit_20260915.json \
  --output-md protenix_ppi/artifacts/data/raw_source_audit_20260915.md
```

## 6. Gate status and next action

Raw-source acquisition, the conservative PSI-MI filter, the UniProt `2026_03` mapping snapshot, and extraction of experimentally screened-negative Luck et al. 2020 RRS controls are complete. A provisional normalized v0.3 dataset now passes schema/source validation, but this does not close G2. Random-reference status alone is not a negative label: only pairwise-assayed RRS rows with an explicit negative result are admitted as `screen_negative`.

G2 remains open until:

1. the remaining identifier/source quarantines and positive-negative contradictions are reported;
2. Negatome redistribution terms receive human confirmation or its rows are kept out of any distributed release;
3. MMseqs2 30%-identity clusters are generated on Linux with the frozen command;
4. the C3 split is generated, hashed, and validated;
5. no model score has been inspected while these choices are being frozen.

The next executable data action is `scripts/run_mmseqs2_clustering.sh` on a Linux host with MMseqs2 installed.
