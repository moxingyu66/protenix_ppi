# UniProt identifier normalization protocol v0.1

Status: executed through conservative two-tier resolution on 2026-09-15

## Purpose

Normalize HuRI Ensembl genes and IntAct/Negatome accession strings to exact human protein sequences without silently collapsing isoforms, selecting among ambiguous mappings, or treating cross-species records as human.

## Version lock

- UniProt release: `2026_03`
- release date: `02-Sep-2026`
- release-evidence SHA-256: `733913b362b641ac2d8de42a6220e794abde0e47d57d6feb57e442c6bf3dac6e`
- human reviewed snapshot SHA-256: `d3a3359d1dff9757c0109effde71972f3d74fe6c6c068393a1f01953e82c3f68`
- snapshot manifest: `data/raw/UniProt/release_2026_03/manifest.json`

Every REST response must report `X-UniProt-Release: 2026_03` and `X-UniProt-Release-Date: 02-September-2026`.

## Resolution policy

1. Preserve the source namespace and raw identifier.
2. Isoform-specific identifiers remain quarantined; they are not folded into a canonical sequence.
3. Non-UniProt IntAct primary identifiers remain in manual review; alternate identifiers are not substituted automatically.
4. Accept a mapping immediately when exactly one target is an reviewed human UniProt entry.
5. For mappings with no reviewed human target, fetch the exact current target records.
6. Accept an unreviewed record only when the mapping has exactly one target and that current target has taxid 9606.
7. Exclude targets demonstrated to be non-human.
8. Quarantine one-to-many, multiple-reviewed, missing-record, obsolete, and unresolved cases.

The policy deliberately does not pick the only human target from a mixed one-to-many mapping; the source mapping itself must be unique for second-tier acceptance.

## Executed stages

### Identifier inventory

```bash
python3 -m protenix_ppi.scripts.build_identifier_inventory \
  --raw-audit protenix_ppi/artifacts/data/raw_source_audit_20260915.json \
  --intact-interim-dir protenix_ppi/data/interim/intact_2026-01-09_direct_v0.1 \
  --output-dir protenix_ppi/data/interim/identifier_inventory_v0.1
```

Observed inventory:

- 12,887 source/identifier rows;
- 8,272 unique HuRI Ensembl genes submitted;
- 3,617 unique UniProt accession strings submitted;
- 500 source-level isoform records quarantined;
- 39 source-level non-UniProt primary identifiers sent to review.

### Pinned ID mappings

```bash
python3 -m protenix_ppi.scripts.fetch_uniprot_mappings \
  --ensembl-ids protenix_ppi/data/interim/identifier_inventory_v0.1/ensembl_gene_ids.txt \
  --uniprot-ids protenix_ppi/data/interim/identifier_inventory_v0.1/uniprotkb_accession_ids.txt \
  --release-evidence protenix_ppi/data/raw/UniProt/release_2026_03/reldate.txt \
  --output-dir protenix_ppi/data/interim/uniprot_mapping_2026_03_v0.1
```

Raw mapping results are immutable snapshots. Their one-to-many outputs are not decisions.

### Conservative resolution

The final executed resolver is v0.2 and includes exact metadata for deferred targets. It accepted 10,851 source/identifier rows representing 9,255 unique protein records. It retained 1,340 rows in quarantine, excluded 645 demonstrated non-human rows, retained 39 for manual review, and left 12 missing historical records deferred.

## Provisional normalized output

`data/processed/human_ppi_2026_03_v0.3` contains the main-source resolution above plus conservatively resolved Luck et al. 2020 RRS assay negatives:

- 9,377 proteins actually referenced by a normalized pair;
- 53,486 canonical undirected pairs;
- 56,655 evidence assertions;
- 52,964 currently eligible pairs;
- 51,446 eligible direct positives;
- 781 eligible curated negatives and 737 eligible screen negatives;
- 522 quarantined pairs;
- 5,138 source rows quarantined before pairing.

The validation report has zero errors. Ten sequences contain ambiguous/non-standard residue symbols and remain explicit warnings. The dataset is not frozen until MMseqs2 clustering and C3 splitting pass.
