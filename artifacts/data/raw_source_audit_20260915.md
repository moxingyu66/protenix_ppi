# Raw PPI source audit

- Registry: `human_ppi_sources_20260915`
- Status: **PASS**
- Acquired sources: 8
- Not acquired: none
- Integrity failures: none

This is an acquisition and format audit, not a frozen biological benchmark.

## Observed profiles

### huri_pairs

```json
{
  "format": "headerless_two_column_ensembl_gene_pairs",
  "rows": 52548,
  "invalid_rows": 0,
  "first_invalid_lines": [],
  "self_pairs_to_quarantine": 480,
  "unique_directed_pairs": 52548,
  "unique_undirected_pairs": 52548,
  "unique_ensembl_genes": 8272,
  "required_next_step": "Map GENCODE-v27 Ensembl genes to an explicitly pinned UniProt canonical release; never guess one-to-one mappings."
}
```

### negatome2_manual_stringent

```json
{
  "format": "four_columns_accession_accession_publication_detection_method",
  "rows": 1991,
  "valid_four_column_rows": 1991,
  "invalid_column_count_rows": 0,
  "self_pairs_to_quarantine": 70,
  "isoform_specific_rows_to_quarantine": 55,
  "unique_directed_pairs": 1964,
  "unique_undirected_pairs": 1956,
  "duplicate_or_reciprocal_rows": 35,
  "unique_accession_strings": 1733,
  "unique_publication_identifiers": 652,
  "top_detection_methods": [
    [
      "MI:0019",
      711
    ],
    [
      "MI:0018",
      507
    ],
    [
      "MI:0096",
      127
    ],
    [
      "MI:0007",
      124
    ],
    [
      "MI:0004",
      114
    ],
    [
      "MI:0059",
      53
    ],
    [
      "MI:0405",
      34
    ],
    [
      "MI:0006",
      34
    ],
    [
      "MI:0110",
      29
    ],
    [
      "MI:0054",
      20
    ],
    [
      "MI:0045",
      16
    ],
    [
      "MI:0030",
      16
    ],
    [
      "MI:0411",
      15
    ],
    [
      "MI:0027",
      15
    ],
    [
      "MI:0892",
      15
    ]
  ],
  "required_next_step": "Resolve taxonomy and current accessions through a pinned UniProt release; preserve isoforms and self-pairs in quarantine, and do not redistribute until license terms are confirmed."
}
```

### intact_human_negative

```json
{
  "format": "PSI_MITAB_2.7_negative_export",
  "columns": 42,
  "rows": 946,
  "negative_flag_counts": {
    "true": 946
  },
  "both_interactors_human_rows": 905,
  "both_human_protein_rows": 903,
  "both_human_protein_primary_uniprot_rows": 781,
  "unique_human_protein_endpoint_pairs": 867,
  "unique_human_protein_primary_uniprot_pairs": 745,
  "unique_interaction_identifiers": 899,
  "top_interaction_types_all_rows": [
    [
      "psi-mi:\"MI:0915\"(physical association)",
      896
    ],
    [
      "psi-mi:\"MI:0407\"(direct interaction)",
      30
    ],
    [
      "psi-mi:\"MI:0403\"(colocalization)",
      11
    ],
    [
      "psi-mi:\"MI:0914\"(association)",
      7
    ],
    [
      "psi-mi:\"MI:0570\"(protein cleavage)",
      1
    ],
    [
      "psi-mi:\"MI:2364\"(proximity)",
      1
    ]
  ],
  "top_detection_methods_all_rows": [
    [
      "psi-mi:\"MI:0397\"(two hybrid array)",
      763
    ],
    [
      "psi-mi:\"MI:0096\"(pull down)",
      67
    ],
    [
      "psi-mi:\"MI:0007\"(anti tag coimmunoprecipitation)",
      48
    ],
    [
      "psi-mi:\"MI:0018\"(two hybrid)",
      24
    ],
    [
      "psi-mi:\"MI:0006\"(anti bait coimmunoprecipitation)",
      18
    ],
    [
      "psi-mi:\"MI:0416\"(fluorescence microscopy)",
      6
    ],
    [
      "psi-mi:\"MI:0663\"(confocal microscopy)",
      5
    ],
    [
      "psi-mi:\"MI:0055\"(fluorescent resonance energy transfer)",
      3
    ],
    [
      "psi-mi:\"MI:0065\"(isothermal titration calorimetry)",
      2
    ],
    [
      "psi-mi:\"MI:0019\"(coimmunoprecipitation)",
      2
    ]
  ],
  "required_next_step": "Require taxid 9606 and protein type for both endpoints, resolve non-UniProt IDs explicitly, and retain this source as a distinct negative stratum."
}
```

### psi_mi_obo

```json
{
  "term_count": 1655,
  "essential_terms": {
    "MI:0407": {
      "expected_name": "direct interaction",
      "observed_name": "direct interaction",
      "parents": [
        "MI:0915"
      ],
      "present": true
    },
    "MI:0915": {
      "expected_name": "physical association",
      "observed_name": "physical association",
      "parents": [
        "MI:0914"
      ],
      "present": true
    },
    "MI:0914": {
      "expected_name": "association",
      "observed_name": "association",
      "parents": [
        "MI:2232"
      ],
      "present": true
    },
    "MI:0208": {
      "expected_name": "genetic interaction (sensu unexpected)",
      "observed_name": "genetic interaction (sensu unexpected)",
      "parents": [
        "MI:2402"
      ],
      "present": true
    },
    "MI:0018": {
      "expected_name": "two hybrid",
      "observed_name": "two hybrid",
      "parents": [
        "MI:0232"
      ],
      "present": true
    },
    "MI:0004": {
      "expected_name": "affinity chromatography technology",
      "observed_name": "affinity chromatography technology",
      "parents": [
        "MI:0091",
        "MI:0400"
      ],
      "present": true
    },
    "MI:0006": {
      "expected_name": "anti bait coimmunoprecipitation",
      "observed_name": "anti bait coimmunoprecipitation",
      "parents": [
        "MI:0019"
      ],
      "present": true
    },
    "MI:0114": {
      "expected_name": "x-ray crystallography",
      "observed_name": "x-ray crystallography",
      "parents": [
        "MI:0013",
        "MI:0659"
      ],
      "present": true
    }
  },
  "essential_term_mismatches": [],
  "required_next_step": "Generate and freeze interaction-type and detection-method descendant sets separately before filtering IntAct."
}
```

## Warning

Acquisition and format integrity do not establish biological eligibility. Identifier mapping, taxonomy, ontology filtering, contradiction resolution, and leakage control remain mandatory.
