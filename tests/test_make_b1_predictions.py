import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.make_b1_predictions import build_predictions
from protenix_ppi.scripts.evaluate_predictions import read_predictions
from protenix_ppi.tests.test_validate_dataset import build_tables, read_rows, write_csv


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_run_hash(root: Path) -> None:
    rows = read_rows(root / "b1_runs.csv")
    rows[0]["input_sha256"] = sha256_file(root / rows[0]["input_json"])
    write_csv(root / "b1_runs.csv", list(rows[0]), rows)


def prepare_bundle(root: Path) -> None:
    build_tables(root)
    sample_name = "HUMAN_P00005_P00006"
    input_path = root / "inputs" / "pair.json"
    write_json(
        input_path,
        [
            {
                "name": sample_name,
                "sequences": [
                    {"proteinChain": {"sequence": "FGHIKL", "count": 1, "id": ["A"]}},
                    {"proteinChain": {"sequence": "MNPQRS", "count": 1, "id": ["B"]}},
                ],
            }
        ],
    )
    predictions_dir = root / "outputs" / sample_name / "seed_42" / "predictions"
    predictions_dir.mkdir(parents=True)
    ranking_scores = [0.90, 0.80, 0.70, 0.60, 0.50]
    iptm_scores = [0.40, 0.95, 0.60, 0.50, 0.30]
    for rank, (ranking_score, iptm) in enumerate(zip(ranking_scores, iptm_scores)):
        write_json(
            predictions_dir / f"{sample_name}_summary_confidence_sample_{rank}.json",
            {
                "iptm": iptm,
                "ranking_score": ranking_score,
                "ptm": 0.70,
                "plddt": 80.0,
                "chain_iptm": [iptm, iptm],
                "chain_pair_iptm": [[0.0, iptm], [iptm, 0.0]],
                "has_clash": False,
                "num_recycles": 10,
            },
        )
    (predictions_dir / f"{sample_name}_sample_0.cif").write_text(
        "data_synthetic\n#\n", encoding="utf-8"
    )
    write_csv(
        root / "b1_runs.csv",
        ["pair_id", "sample_name", "input_json", "input_sha256", "predictions_dir"],
        [
            {
                "pair_id": sample_name,
                "sample_name": sample_name,
                "input_json": "inputs/pair.json",
                "input_sha256": sha256_file(input_path),
                "predictions_dir": f"outputs/{sample_name}/seed_42/predictions",
            }
        ],
    )
    write_json(
        root / "backbone_lock.json",
        {
            "schema_version": "1.0",
            "repository_url": "https://github.com/bytedance/Protenix.git",
            "source_commit": "a" * 40,
            "package_version": "1.0.0",
            "model_name": "protenix_base_default_v1.0.0",
            "declared_training_cutoff": "2021-09-30",
            "checkpoint_path": "/checkpoint/model.pt",
            "checkpoint_sha256": "b" * 64,
            "g0_route": "B",
            "python_version": "3.11.9",
            "torch_version": "2.5.1",
            "cuda_runtime": "12.4",
            "nvidia_driver": "550.54",
            "triangle_attention_kernel": "cuequivariance",
            "triangle_multiplicative_kernel": "cuequivariance",
        },
    )
    write_json(
        root / "condition_lock.json",
        {
            "schema_version": "1.0",
            "condition_id": "B1_primary_msa_no_template_seed_42",
            "model_name": "protenix_base_default_v1.0.0",
            "seed": 42,
            "use_msa": True,
            "use_template": False,
            "msa_pair_as_unpair": True,
            "n_cycle": 10,
            "n_step": 200,
            "n_sample": 5,
            "dtype": "bf16",
            "use_default_params": True,
            "enable_tf32": True,
            "enable_efficient_fusion": True,
            "enable_diffusion_shared_vars_cache": True,
            "sorted_by_ranking_score": True,
            "sample_selection": "native_rank_0",
            "primary_score_field": "iptm",
            "comparator_score_fields": ["ranking_score"],
            "label_fitted_combination": False,
        },
    )


def run_bundle(root: Path):
    return build_predictions(
        root / "proteins.csv",
        root / "pairs.csv",
        root / "evidence.csv",
        root / "splits.csv",
        root / "b1_runs.csv",
        root / "backbone_lock.json",
        root / "condition_lock.json",
        root / "result",
        "c3_primary",
        "0",
    )


class MakeB1PredictionsTests(unittest.TestCase):
    def test_end_to_end_uses_native_rank_zero_and_writes_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            metadata = run_bundle(root)

            primary = read_predictions(root / "result" / "predictions.csv", "test")
            comparator = read_predictions(
                root / "result" / "predictions_ranking_score.csv", "test"
            )
            self.assertEqual(len(primary), 1)
            self.assertEqual(primary[0].pair_id, "HUMAN_P00005_P00006")
            self.assertEqual(primary[0].score, 0.40)
            self.assertEqual(comparator[0].score, 0.90)
            self.assertEqual(metadata["primary_score"], "iptm")
            self.assertFalse(metadata["label_fitted_combination"])

            provenance = read_rows(root / "result" / "source_manifest.csv")
            self.assertEqual(provenance[0]["seed"], "42")
            self.assertEqual(provenance[0]["selected_rank"], "0")
            self.assertEqual(len(provenance[0]["summary_sha256"]), 64)
            self.assertEqual(len(provenance[0]["structure_sha256"]), 64)

    def test_rejects_input_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            rows = read_rows(root / "b1_runs.csv")
            rows[0]["input_sha256"] = "0" * 64
            write_csv(root / "b1_runs.csv", list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "input SHA-256 mismatch"):
                run_bundle(root)

    def test_rejects_non_protein_entity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            input_path = root / "inputs" / "pair.json"
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            payload[0]["sequences"].append({"ligand": {"ligand": "CCD_ATP", "count": 1}})
            write_json(input_path, payload)
            rewrite_run_hash(root)
            with self.assertRaisesRegex(ValueError, "proteinChain entities only"):
                run_bundle(root)

    def test_rejects_sequence_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            input_path = root / "inputs" / "pair.json"
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            payload[0]["sequences"][1]["proteinChain"]["sequence"] = "AAAAAA"
            write_json(input_path, payload)
            rewrite_run_hash(root)
            with self.assertRaisesRegex(ValueError, "do not match the frozen benchmark pair"):
                run_bundle(root)

    def test_rejects_rank_zero_that_is_not_native_best(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            sample_name = "HUMAN_P00005_P00006"
            summary_path = (
                root
                / "outputs"
                / sample_name
                / "seed_42"
                / "predictions"
                / f"{sample_name}_summary_confidence_sample_1.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["ranking_score"] = 0.99
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "rank 0 is not the highest"):
                run_bundle(root)

    def test_rejects_later_cutoff_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_bundle(root)
            lock_path = root / "backbone_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["model_name"] = "protenix_base_20250630_v1.0.0"
            lock["declared_training_cutoff"] = "2025-06-30"
            write_json(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "B1 primary model"):
                run_bundle(root)


if __name__ == "__main__":
    unittest.main()
