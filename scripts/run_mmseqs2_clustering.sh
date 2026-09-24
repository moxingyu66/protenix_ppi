#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <proteins.csv> <output-dir> [threads]" >&2
  exit 2
fi

proteins_csv=$1
output_dir=$2
threads=${3:-8}

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "MMseqs2 is not installed or not on PATH" >&2
  exit 1
fi

mkdir -p "$output_dir"
python3 -m protenix_ppi.scripts.export_protein_fasta \
  --proteins "$proteins_csv" \
  --fasta "$output_dir/proteins.fasta" \
  --metadata "$output_dir/fasta_metadata.json"

mmseqs_version=$(mmseqs version)
mmseqs easy-cluster \
  "$output_dir/proteins.fasta" \
  "$output_dir/mmseqs30" \
  "$output_dir/tmp" \
  --min-seq-id 0.30 \
  -c 0.50 \
  --cov-mode 0 \
  --alignment-mode 3 \
  --cluster-mode 0 \
  -s 7.5 \
  --threads "$threads" \
  2>&1 | tee "$output_dir/mmseqs.log"

python3 -m protenix_ppi.scripts.apply_mmseqs_clusters \
  --proteins "$proteins_csv" \
  --cluster-tsv "$output_dir/mmseqs30_cluster.tsv" \
  --output "$output_dir/proteins.clustered.csv" \
  --metadata "$output_dir/mmseqs_cluster_metadata.json" \
  --mmseqs-version "$mmseqs_version"
