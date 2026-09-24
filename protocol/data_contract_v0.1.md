# G2 data contract v0.1 — human physical PPI benchmark

Status: schema prepared; no biological records have been accepted yet

## 1. Design principle

The benchmark uses normalized tables rather than one spreadsheet that mixes proteins, pair labels, evidence, and split assignments. Raw source files remain immutable. Every processed record must be reproducible from a source version and transformation log.

## 2. Required tables

### `proteins.csv`

One row per canonical protein sequence.

| Field | Meaning |
|---|---|
| `uniprot_accession` | Canonical UniProt accession used as the entity key |
| `gene_symbol` | HGNC gene symbol when available |
| `taxid` | Must be 9606 for the primary human benchmark |
| `sequence` | Uppercase canonical amino-acid sequence |
| `sequence_length` | Length computed from `sequence` |
| `sequence_sha256` | SHA-256 of the exact uppercase sequence string |
| `uniprot_release` | UniProt release used for mapping |
| `sequence_version` | UniProt sequence version where available |
| `is_reviewed` | `true` for Swiss-Prot reviewed entries |
| `homology_cluster_30` | MMseqs2 cluster ID under the frozen 30% identity protocol |
| `retrieved_at` | ISO date of retrieval |

Isoforms must not be silently folded into a canonical accession. If an experiment is isoform-specific, preserve it outside the primary canonical benchmark until an explicit isoform policy is approved.

### `pairs.csv`

One row per undirected protein pair after evidence aggregation.

| Field | Meaning |
|---|---|
| `pair_id` | `HUMAN_<lexicographically smaller accession>_<larger accession>` |
| `uniprot_a`, `uniprot_b` | Canonically ordered endpoints |
| `label` | `1`, `0`, or blank for unlabeled |
| `evidence_class` | `direct_positive`, `curated_negative`, `screen_negative`, `compartment_negative`, or `unlabeled` |
| `complex_group_id` | Stable group for pairs derived from the same complex; blank if not applicable |
| `pair_status` | `eligible`, `quarantine`, or `excluded` |
| `exclusion_reason` | Required when excluded; informative when quarantined |

Self-pairs are not silently deleted. They must be quarantined because homomeric interaction is biologically meaningful but outside the primary heterotypic PPI question.

### `evidence.csv`

One row per traceable evidence assertion. Multiple rows may support one pair.

| Field | Meaning |
|---|---|
| `evidence_id` | Locally stable evidence key |
| `pair_id` | Foreign key to `pairs.csv` |
| `source_database` | HuRI, IntAct, BioGRID, Negatome, or another approved source |
| `source_version` | Exact release/version/date |
| `source_record_id` | Original database identifier |
| `publication_id` | PMID/DOI where available |
| `interaction_type_mi` | PSI-MI interaction type, e.g. `MI:0407` for direct interaction |
| `detection_method_mi` | PSI-MI experimental detection method, not the interaction type |
| `evidence_polarity` | `positive`, `negative`, or `context_only` |
| `negative_definition` | Experimental negative, screen negative, compartment rule, or blank |
| `license` | Source license or terms identifier |
| `retrieved_at` | ISO retrieval date |

Important: `MI:0407` and a detection-method term represent different ontology axes. The exact accepted PSI-MI descendant whitelist must be resolved against a pinned PSI-MI ontology release before biological filtering begins.

### `splits.csv`

One row per pair assignment per split scheme/fold.

| Field | Meaning |
|---|---|
| `split_scheme` | `pair_random_diagnostic`, `c3_primary`, `homology_primary`, or `temporal_external` |
| `fold` | Fold identifier, normally `0` for a fixed holdout |
| `pair_id` | Foreign key to `pairs.csv` |
| `partition` | `train`, `validation`, or `test` |
| `pm_class` | C1/C2/C3 classification relative to training proteins where applicable |

Split files are generated once, hashed, and frozen before model scores are inspected.

## 3. Label rules

| Evidence class | Pair label | Required interpretation |
|---|---:|---|
| `direct_positive` | 1 | Direct/binary physical interaction supported by accepted evidence |
| `curated_negative` | 0 | Literature-curated experimental non-interaction |
| `screen_negative` | 0 | Explicitly tested and negative in a defined screen; not equivalent to universal non-binding |
| `compartment_negative` | 0 | Rule-derived physiological non-contact stratum; kept separate from experimental negatives |
| `unlabeled` | blank | Unknown state; never treated as a negative by default |

The evaluation report must remain stratified by negative type. A pooled score alone is insufficient because an easy compartment-negative test can hide failure on same-compartment hard negatives.

## 4. Leakage invariants

- Reciprocal pairs are canonicalized and cannot appear twice.
- For `c3_primary`, protein endpoint sets must be disjoint across train, validation, and test.
- For `homology_primary`, 30% identity clusters must be disjoint across partitions.
- A `complex_group_id` cannot cross partitions within a split scheme/fold.
- An evidence source record cannot silently support contradictory pair labels.
- Test labels and predictions cannot enter preprocessing, feature selection, threshold selection, or early stopping.
- Protenix inference features are generated only after splits are frozen; feature filenames include pair ID, backbone lock digest, input-policy digest, and seed.

## 5. Raw-data rule

Raw downloads belong under `data/raw/<source>/<version>/` and are immutable. Each source directory must contain a checksum manifest, retrieval URL, retrieval date, license note, and transformation command log. Generated canonical tables belong under `data/processed/<dataset_version>/`.

## 6. G2 acceptance criteria

- [ ] All four tables validate without errors.
- [ ] Source versions and licenses are recorded.
- [ ] PSI-MI ontology release and accepted terms are frozen.
- [ ] Every eligible positive has traceable direct evidence.
- [ ] Every labeled negative has an explicit negative definition.
- [ ] Random, C3, homology, and temporal schemes are stored separately.
- [ ] C3 and homology leakage checks report zero violations.
- [ ] Split files and processed tables have SHA-256 manifests.
- [ ] Dataset statistics are reported by label, evidence class, source, length, degree, and partition.

