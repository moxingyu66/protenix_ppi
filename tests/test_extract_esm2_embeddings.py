import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from protenix_ppi.scripts.embedding_bundle import load_bundle
from protenix_ppi.scripts.extract_esm2_embeddings import (
    ProteinRecord,
    extract_bundle_with_embedder,
    iter_window_tasks,
    window_starts,
)


def write_proteins(path: Path, sequences: dict[str, str]) -> None:
    fields = ["uniprot_accession", "sequence", "sequence_length", "sequence_sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for accession, sequence in sequences.items():
            writer.writerow(
                {
                    "uniprot_accession": accession,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
            )


def backend_metadata() -> dict:
    return {
        "model_id": "fixture/esm2",
        "model_revision": "a" * 40,
        "checkpoint_files_sha256": {"model.safetensors": "b" * 64},
        "layer": 1,
        "tokenizer_version": "fixture",
        "python_version": "fixture",
        "torch_version": "fixture",
        "transformers_version": "fixture",
        "device": "cpu",
        "compute_dtype": "float32",
    }


class ExtractEsm2EmbeddingsTests(unittest.TestCase):
    def test_window_starts_have_fixed_stride_and_complete_coverage(self):
        self.assertEqual(window_starts(5, 5, 2), [0])
        self.assertEqual(window_starts(6, 5, 2), [0, 2])
        self.assertEqual(window_starts(11, 5, 2), [0, 2, 4, 6])
        starts = window_starts(11, 5, 2)
        coverage = np.zeros(11, dtype=int)
        for start in starts:
            coverage[start : start + 5] += 1
        self.assertTrue(np.all(coverage > 0))

    def test_task_order_marks_only_final_window(self):
        proteins = [ProteinRecord("P1", "ABCDEFG", "x"), ProteinRecord("P2", "ABC", "y")]
        tasks = list(iter_window_tasks(proteins, 5, 2))
        self.assertEqual(
            [(task.accession, task.start) for task in tasks],
            [("P1", 0), ("P1", 2), ("P2", 0)],
        )
        self.assertEqual([task.is_last for task in tasks], [False, True, True])

    def test_overlap_is_averaged_per_residue_before_pooling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins = root / "proteins.csv"
            write_proteins(proteins, {"P1": "AAAAAA"})
            call_count = 0

            def fake_embedder(sequences):
                nonlocal call_count
                outputs = []
                for sequence in sequences:
                    call_count += 1
                    outputs.append(np.full((len(sequence), 2), call_count, dtype=np.float32))
                return outputs

            bundle = root / "bundle"
            extract_bundle_with_embedder(
                proteins,
                bundle,
                fake_embedder,
                backend_metadata(),
                embedding_dim=2,
                window_size=5,
                stride=2,
                batch_size=2,
            )
            matrix = np.load(bundle / "embeddings.npy", allow_pickle=False)
            # Windows [0:5]=1 and [2:6]=2 reconstruct [1,1,1.5,1.5,1.5,2].
            np.testing.assert_allclose(matrix[0], [1.4166666, 1.4166666], rtol=1e-6)
            _, summary = load_bundle(bundle, proteins)
            self.assertEqual(summary.sequence_count, 1)
            metadata = json.loads(
                (bundle / "embedding_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["window_count"], 2)
            self.assertEqual(
                metadata["window_tail_policy"],
                "fixed_stride_until_previous_window_covers_sequence_end",
            )

    def test_extractor_rejects_wrong_residue_shape_without_finalizing_bundle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proteins = root / "proteins.csv"
            write_proteins(proteins, {"P1": "AAAA"})

            def bad_embedder(sequences):
                return [np.zeros((len(sequences[0]) - 1, 2), dtype=np.float32)]

            bundle = root / "bundle"
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                extract_bundle_with_embedder(
                    proteins,
                    bundle,
                    bad_embedder,
                    backend_metadata(),
                    embedding_dim=2,
                )
            self.assertFalse((bundle / "embeddings.npy").exists())
            self.assertFalse((bundle / "embedding_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
