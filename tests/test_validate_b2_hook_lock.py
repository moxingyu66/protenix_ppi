import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.validate_b2_hook_lock import validate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    commit = "a" * 40
    checkpoint = "b" * 64
    backbone = root / "backbone_lock.json"
    write_json(
        backbone,
        {
            "repository_url": "https://github.com/bytedance/Protenix.git",
            "source_commit": commit,
            "package_version": "fixture",
            "model_name": "protenix_base_default_v1.0.0",
            "declared_training_cutoff": "2021-09-30",
            "checkpoint_sha256": checkpoint,
            "g0_route": "B",
            "python_version": "3.11",
            "torch_version": "fixture",
            "cuda_runtime": "fixture",
            "nvidia_driver": "fixture",
        },
    )
    g1_manifest = root / "manifest.json"
    write_json(
        g1_manifest,
        {
            "model_name": "protenix_base_default_v1.0.0",
            "source_commit": commit,
            "checkpoint_sha256": checkpoint,
            "exit_code": 0,
        },
    )
    hook = root / "hook_evidence.json"
    source_files = {
        "protenix/model/protenix.py": "c" * 64,
        "protenix/model/modules/pairformer.py": "d" * 64,
    }
    write_json(
        hook,
        {
            "schema_version": "1.0",
            "model_name": "protenix_base_default_v1.0.0",
            "declared_training_cutoff": "2021-09-30",
            "source_commit": commit,
            "backbone_lock_sha256": digest(backbone),
            "g1_manifest_sha256": digest(g1_manifest),
            "module_call": "Protenix.get_pairformer_output",
            "candidate_tensor": "z",
            "capture_stage": "after_final_recycle_before_diffusion",
            "last_recycle_confirmed": True,
            "before_diffusion_confirmed": True,
            "label_blind": True,
            "labels_read": False,
            "split_assignments_read": False,
            "use_msa": True,
            "use_template": False,
            "tensor_shape_before_pooling": [1, 32, 32, 128],
            "c_z": 128,
            "source_files": source_files,
            "swap_invariance": {
                "pair_count": 3,
                "tolerance": 1e-5,
                "maximum_absolute_delta": 2e-6,
                "passed": True,
            },
        },
    )
    lock = root / "feature_extraction_lock.json"
    write_json(
        lock,
        {
            "schema_version": "1.0",
            "model_name": "protenix_base_default_v1.0.0",
            "declared_training_cutoff": "2021-09-30",
            "source_commit": commit,
            "backbone_lock_sha256": digest(backbone),
            "g1_manifest_sha256": digest(g1_manifest),
            "hook_evidence_sha256": digest(hook),
            "inference_condition_lock_sha256": "e" * 64,
            "seed_policy": "registered_seed_42_123_999",
            "sample_policy": "native_rank_zero_pairformer_z",
            "label_blind": True,
            "swap_invariant": True,
            "swap_invariance_audit": {
                "pair_count": 3,
                "tolerance": 1e-5,
                "maximum_absolute_delta": 2e-6,
            },
            "feature_components": [
                {
                    "name": "protenix_pair_z_cross_chain_mean_std",
                    "source_tensor": "Protenix.get_pairformer_output:return[2]:z",
                    "pooling": "cross_chain_direction_symmetrized_mean_and_std",
                }
            ],
        },
    )
    return lock, backbone, g1_manifest, hook


class ValidateB2HookLockTests(unittest.TestCase):
    def test_complete_evidence_passes(self):
        with TemporaryDirectory() as directory:
            paths = make_fixture(Path(directory))
            result = validate(*paths)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["hook_evidence"]["c_z"], 128)

    def test_wrong_capture_stage_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock, backbone, manifest, hook = make_fixture(root)
            value = json.loads(hook.read_text(encoding="utf-8"))
            value["capture_stage"] = "after_diffusion"
            write_json(hook, value)
            with self.assertRaisesRegex(ValueError, "capture_stage"):
                validate(lock, backbone, manifest, hook)

    def test_label_aware_evidence_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock, backbone, manifest, hook = make_fixture(root)
            value = json.loads(hook.read_text(encoding="utf-8"))
            value["labels_read"] = True
            write_json(hook, value)
            with self.assertRaisesRegex(ValueError, "label-blind"):
                validate(lock, backbone, manifest, hook)

    def test_swap_audit_failure_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock, backbone, manifest, hook = make_fixture(root)
            value = json.loads(hook.read_text(encoding="utf-8"))
            value["swap_invariance"]["maximum_absolute_delta"] = 1.0
            write_json(hook, value)
            with self.assertRaisesRegex(ValueError, "swap-invariance"):
                validate(lock, backbone, manifest, hook)

    def test_feature_lock_hash_drift_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock, backbone, manifest, hook = make_fixture(root)
            value = json.loads(lock.read_text(encoding="utf-8"))
            value["hook_evidence_sha256"] = "f" * 64
            write_json(lock, value)
            with self.assertRaisesRegex(ValueError, "hook evidence hash"):
                validate(lock, backbone, manifest, hook)


if __name__ == "__main__":
    unittest.main()
