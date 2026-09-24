#!/usr/bin/env python3
"""Extract the frozen ESM-2 B0b protein embedding bundle.

Heavy dependencies are imported only by the production backend so that the
windowing and bundle-writing logic can be tested without PyTorch installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from protenix_ppi.scripts.embedding_bundle import sha256_file


PRIMARY_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
PRIMARY_MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
PRIMARY_CHECKPOINT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
PRIMARY_LAYER = 33
PRIMARY_DIMENSION = 1280
PRIMARY_WINDOW_SIZE = 1022
PRIMARY_WINDOW_STRIDE = 511
PRIMARY_POOLING = "mean_residue_tokens_after_overlap_average"
HEX_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class ProteinRecord:
    accession: str
    sequence: str
    sequence_sha256: str


@dataclass(frozen=True)
class WindowTask:
    row_index: int
    accession: str
    start: int
    sequence: str
    is_last: bool


WindowEmbedder = Callable[[Sequence[str]], list[np.ndarray]]


def read_proteins(proteins_path: Path) -> list[ProteinRecord]:
    with proteins_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("proteins.csv has no header")
        required = {"uniprot_accession", "sequence", "sequence_sha256", "sequence_length"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"proteins.csv is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("proteins.csv is empty")

    records: list[ProteinRecord] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        accession = row["uniprot_accession"].strip()
        sequence = row["sequence"].strip().upper()
        sequence_hash = row["sequence_sha256"].strip().lower()
        if not accession or accession in seen:
            raise ValueError(f"proteins.csv:{line}: missing or duplicate accession")
        if not sequence:
            raise ValueError(f"proteins.csv:{line}: empty sequence for {accession}")
        try:
            declared_length = int(row["sequence_length"])
        except ValueError as exc:
            raise ValueError(f"proteins.csv:{line}: invalid sequence_length") from exc
        if declared_length != len(sequence):
            raise ValueError(f"proteins.csv:{line}: sequence length mismatch for {accession}")
        observed_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if sequence_hash != observed_hash:
            raise ValueError(f"proteins.csv:{line}: sequence hash mismatch for {accession}")
        seen.add(accession)
        records.append(ProteinRecord(accession, sequence, sequence_hash))
    return records


def window_starts(length: int, window_size: int, stride: int) -> list[int]:
    """Return fixed-stride starts, stopping once the final window covers the end."""
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if window_size <= 0 or stride <= 0 or stride > window_size:
        raise ValueError("window_size and stride must satisfy 0 < stride <= window_size")
    starts = [0]
    while starts[-1] + window_size < length:
        starts.append(starts[-1] + stride)
    return starts


def iter_window_tasks(
    proteins: Sequence[ProteinRecord], window_size: int, stride: int
) -> Iterator[WindowTask]:
    for row_index, protein in enumerate(proteins):
        starts = window_starts(len(protein.sequence), window_size, stride)
        for position, start in enumerate(starts):
            yield WindowTask(
                row_index=row_index,
                accession=protein.accession,
                start=start,
                sequence=protein.sequence[start : start + window_size],
                is_last=position == len(starts) - 1,
            )


def _chunks(items: Iterator[WindowTask], size: int) -> Iterator[list[WindowTask]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[WindowTask] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_bundle_with_embedder(
    proteins_path: Path,
    output_dir: Path,
    embed_windows: WindowEmbedder,
    backend_metadata: dict[str, Any],
    *,
    embedding_dim: int,
    window_size: int = PRIMARY_WINDOW_SIZE,
    stride: int = PRIMARY_WINDOW_STRIDE,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Write a complete bundle using an injected residue-window embedder."""
    proteins = read_proteins(proteins_path)
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [
        output_dir / "embeddings.npy",
        output_dir / "embedding_index.csv",
        output_dir / "embedding_metadata.json",
    ]
    if any(path.exists() for path in final_paths):
        raise ValueError("embedding bundle output already exists; use a new directory")

    incomplete_embeddings = output_dir / "embeddings.incomplete.npy"
    incomplete_index = output_dir / "embedding_index.incomplete.csv"
    incomplete_metadata = output_dir / "embedding_metadata.incomplete.json"
    for path in (incomplete_index, incomplete_metadata):
        if path.exists():
            path.unlink()
    matrix = np.lib.format.open_memmap(
        incomplete_embeddings,
        mode="w+",
        dtype=np.float32,
        shape=(len(proteins), embedding_dim),
    )
    accumulators: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    completed = 0
    window_count = 0
    started = time.perf_counter()

    try:
        tasks = iter_window_tasks(proteins, window_size, stride)
        for batch in _chunks(tasks, batch_size):
            residue_embeddings = embed_windows([task.sequence for task in batch])
            if len(residue_embeddings) != len(batch):
                raise ValueError("embedder returned a different number of windows")
            for task, values in zip(batch, residue_embeddings):
                values = np.asarray(values, dtype=np.float32)
                expected_shape = (len(task.sequence), embedding_dim)
                if values.shape != expected_shape:
                    raise ValueError(
                        f"embedder shape mismatch for {task.accession}: "
                        f"expected {expected_shape}, observed {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite residue embedding for {task.accession}")
                if task.row_index not in accumulators:
                    length = len(proteins[task.row_index].sequence)
                    accumulators[task.row_index] = (
                        np.zeros((length, embedding_dim), dtype=np.float32),
                        np.zeros(length, dtype=np.uint16),
                    )
                sums, counts = accumulators[task.row_index]
                stop = task.start + len(task.sequence)
                sums[task.start:stop] += values
                counts[task.start:stop] += 1
                window_count += 1
                if task.is_last:
                    if np.any(counts == 0):
                        raise ValueError(f"window policy left uncovered residues for {task.accession}")
                    reconstructed = sums / counts[:, None]
                    pooled = reconstructed.mean(axis=0, dtype=np.float64).astype(np.float32)
                    if not np.isfinite(pooled).all():
                        raise ValueError(f"non-finite pooled embedding for {task.accession}")
                    matrix[task.row_index] = pooled
                    del accumulators[task.row_index]
                    completed += 1
        if completed != len(proteins) or accumulators:
            raise ValueError("embedding extraction did not complete every protein")
        matrix.flush()
        del matrix

        with incomplete_index.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["uniprot_accession", "sequence_sha256", "row_index"],
            )
            writer.writeheader()
            for row_index, protein in enumerate(proteins):
                writer.writerow(
                    {
                        "uniprot_accession": protein.accession,
                        "sequence_sha256": protein.sequence_sha256,
                        "row_index": row_index,
                    }
                )

        elapsed = time.perf_counter() - started
        metadata = {
            "schema_version": "1.0",
            **backend_metadata,
            "pooling": PRIMARY_POOLING,
            "window_tail_policy": "fixed_stride_until_previous_window_covers_sequence_end",
            "max_residues_per_window": window_size,
            "window_stride": stride,
            "embedding_dim": embedding_dim,
            "dtype": "float32",
            "protein_table_sha256": sha256_file(proteins_path),
            "sequence_count": len(proteins),
            "window_count": window_count,
            "batch_size": batch_size,
            "wall_time_seconds": elapsed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        incomplete_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        incomplete_embeddings.replace(output_dir / "embeddings.npy")
        incomplete_index.replace(output_dir / "embedding_index.csv")
        incomplete_metadata.replace(output_dir / "embedding_metadata.json")
        return metadata
    except Exception:
        try:
            matrix.flush()
        except Exception:
            pass
        raise


def _checkpoint_digests(snapshot_path: Path) -> dict[str, str]:
    candidates = sorted(snapshot_path.glob("*.safetensors"))
    if not candidates:
        candidates = sorted(snapshot_path.glob("pytorch_model*.bin"))
    if not candidates:
        raise ValueError("no safetensors or PyTorch checkpoint files found in model snapshot")
    return {path.name: sha256_file(path) for path in candidates}


class HuggingFaceEsmEmbedder:
    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str,
        compute_dtype: str,
        cache_dir: Path | None,
    ) -> None:
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer, EsmModel
        except ImportError as exc:
            raise RuntimeError(
                "ESM extraction requires torch, transformers, huggingface_hub, and safetensors"
            ) from exc

        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                ],
            )
        ).resolve()
        resolved_revision = snapshot.name.lower()
        if not HEX_REVISION.fullmatch(resolved_revision):
            raise ValueError(
                "Hugging Face snapshot did not resolve to an immutable hexadecimal revision"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if device == "cpu" and compute_dtype != "float32":
            raise ValueError("CPU extraction is restricted to float32 compute")

        checkpoint_digests = _checkpoint_digests(snapshot)
        if model_id == PRIMARY_MODEL_ID:
            if resolved_revision != PRIMARY_MODEL_REVISION:
                raise ValueError("primary ESM-2 revision differs from the frozen B0b lock")
            if checkpoint_digests != {"model.safetensors": PRIMARY_CHECKPOINT_SHA256}:
                raise ValueError("primary ESM-2 checkpoint SHA-256 differs from the frozen B0b lock")

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = EsmModel.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=dtype_map[compute_dtype],
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        layer_count = int(model.config.num_hidden_layers)
        hidden_size = int(model.config.hidden_size)
        if model_id == PRIMARY_MODEL_ID and (
            layer_count != PRIMARY_LAYER or hidden_size != PRIMARY_DIMENSION
        ):
            raise ValueError("primary ESM-2 architecture does not match the frozen B0b contract")

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.embedding_dim = hidden_size
        self.metadata = {
            "model_id": model_id,
            "model_revision": resolved_revision,
            "checkpoint_files_sha256": checkpoint_digests,
            "layer": layer_count,
            "tokenizer_version": f"{tokenizer.__class__.__name__}; transformers={transformers.__version__}",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
            "device": device,
            "compute_dtype": compute_dtype,
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        }

    def __call__(self, sequences: Sequence[str]) -> list[np.ndarray]:
        torch = self.torch
        encoded = self.tokenizer(
            list(sequences),
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        attention_mask = encoded["attention_mask"]
        special_tokens_mask = encoded.pop("special_tokens_mask")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.model(**model_inputs).last_hidden_state

        outputs: list[np.ndarray] = []
        for index, sequence in enumerate(sequences):
            residue_mask = attention_mask[index].bool() & ~special_tokens_mask[index].bool()
            residue_values = hidden[index, residue_mask.to(hidden.device)]
            if residue_values.shape[0] != len(sequence):
                raise ValueError(
                    "tokenizer residue-token count differs from sequence length; "
                    f"expected {len(sequence)}, observed {residue_values.shape[0]}"
                )
            outputs.append(residue_values.float().cpu().numpy())
        return outputs

    def telemetry(self) -> dict[str, Any]:
        torch = self.torch
        if not self.device.startswith("cuda"):
            return {"peak_gpu_memory_bytes": None}
        return {"peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(self.device))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=PRIMARY_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=PRIMARY_MODEL_REVISION,
        help="Immutable Hugging Face revision (defaults to the frozen primary B0b commit).",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--compute-dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    backend = HuggingFaceEsmEmbedder(
        args.model_id, args.revision, args.device, args.compute_dtype, args.cache_dir
    )
    metadata = extract_bundle_with_embedder(
        args.proteins,
        args.output_dir,
        backend,
        backend.metadata,
        embedding_dim=backend.embedding_dim,
        batch_size=args.batch_size,
    )
    metadata.update(backend.telemetry())
    metadata_path = args.output_dir / "embedding_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
