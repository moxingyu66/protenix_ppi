# B0b frozen protein-language-model baseline protocol v0.1

Status: extractor, embedding contract, and classifier prepared; no ESM checkpoint downloaded

## Purpose

B0b is the strong sequence-only comparator required before claiming that frozen Protenix structural representations add information beyond modern protein language models.

## Working encoder choice

- Model family: ESM-2
- Working checkpoint: `facebook/esm2_t33_650M_UR50D`
- Immutable Hugging Face revision: `08e4846e537177426273712802403f7ba8261b6c`
- `model.safetensors` SHA-256: `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`
- Parameters: frozen
- Representation: final hidden layer, residue tokens only
- Per-protein pooling: arithmetic mean over residue embeddings
- Output storage dtype: float32

The exact immutable model revision and checkpoint-file SHA-256 values must be recorded at extraction time. The symbolic model name alone is insufficient. If a different ESM-2 scale is chosen because of server constraints, it becomes a separately named experimental condition and cannot silently replace B0b.

## Long-sequence policy

The working maximum is 1,022 residues per window. Longer proteins use overlapping windows:

- window size: 1,022 residues;
- stride: 511 residues;
- residue embeddings in overlaps are averaged across windows;
- the final protein vector is the mean of the reconstructed per-residue embeddings.

Window starts advance from residue 0 by the fixed stride until the preceding
window covers the sequence end. The terminal window may therefore be shorter
than 1,022 residues; it is not shifted backward and no sequence is truncated.

Extract the primary bundle on the selected GPU server with:

```bash
python3 -m protenix_ppi.scripts.extract_esm2_embeddings \
  --proteins protenix_ppi/data/processed/human_ppi_2026_03_v0.3/mmseqs30/proteins.clustered.csv \
  --output-dir protenix_ppi/artifacts/embeddings/esm2_t33_650m_ur50d_primary \
  --model-id facebook/esm2_t33_650M_UR50D \
  --revision 08e4846e537177426273712802403f7ba8261b6c \
  --device cuda \
  --compute-dtype float32 \
  --batch-size 1
```

The extractor downloads only the safetensors checkpoint and tokenizer/config
files, verifies the frozen revision and checkpoint SHA-256, and records them in
the bundle. Existing final bundle files are never overwritten. Increase batch
size only after a batch-1 smoke test; a different compute dtype must be
reported as a separately named condition.

Truncating long proteins without a separately reported sensitivity analysis is not permitted. The exact tokenizer, special-token handling, layer, window size, stride, and pooling rule must be recorded.

## Embedding bundle

Each frozen bundle contains:

```text
embeddings.npy
embedding_index.csv
embedding_metadata.json
```

`embedding_index.csv` binds each array row to a canonical accession and exact sequence SHA-256. `embedding_metadata.json` records model ID, immutable revision, layer, pooling, chunk policy, dimension, dtype, source protein-table digest, checkpoint-file digests, and extraction software.

The bundle is rejected if:

- an accession is duplicated or missing;
- a sequence hash differs from `proteins.csv`;
- row indices are not an exact permutation of the array rows;
- dimensions or dtype disagree with metadata;
- any embedding value is NaN or infinite;
- model revision or checkpoint digest is absent.

## Pair representation and classifier

For frozen protein vectors `h_a` and `h_b`, construct the symmetric feature:

```text
[ (h_a + h_b)/2, abs(h_a - h_b), h_a * h_b,
  mean(log lengths), abs(log-length difference), min/max length ratio ]
```

Use the same standardized, class-balanced logistic-regression protocol and frozen `balanced_explicit_1to1` validation cohort as B0a. This controls the classifier so that B0a/B0b differences primarily reflect representation quality.

## Required comparisons

- B0b vs B0a: value of a modern sequence representation over composition shortcuts.
- B2 vs B0b: value of frozen Protenix structural representation over the strong sequence baseline.

H2 is not supported unless B2 is compared with B0b on exactly the same C3 test pairs.
