import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.validate_g1_artifacts import validate_run


COMMIT = "a" * 40
DIGEST = "b" * 64


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_valid_run(root: Path):
    input_path = root / "inputs" / "input.json"
    write_json(input_path, {"sequences": []})
    input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()

    lock = {
        "repository_url": "https://github.com/bytedance/Protenix.git",
        "source_commit": COMMIT,
        "package_version": "1.0.0",
        "model_name": "protenix_base_default_v1.0.0",
        "declared_training_cutoff": "2021-09-30",
        "checkpoint_sha256": DIGEST,
        "g0_route": "A",
        "python_version": "3.10.15",
        "torch_version": "2.5.1",
        "cuda_runtime": "12.4",
        "nvidia_driver": "550.54.15",
    }
    write_json(root / "backbone_lock.json", lock)

    for relative in ("logs/command.txt", "logs/stdout_stderr.txt", "telemetry/nvidia_smi.csv"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")

    gradient = {
        "loss_value": 1.5,
        "loss_finite": True,
        "trainable_parameter_count": 100,
        "parameters_with_finite_nonzero_gradients": 2,
        "frozen_parameters_with_gradients": 0,
        "optimizer_step_completed": True,
        "peak_gpu_memory_mib": 1024,
    }
    write_json(root / "backward" / "gradient_summary.json", gradient)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "prediction.cif").write_text("data_test\n", encoding="utf-8")
    write_json(root / "outputs" / "confidence.json", {"score": 0.5})

    manifest = {
        "model_name": lock["model_name"],
        "source_commit": COMMIT,
        "checkpoint_sha256": DIGEST,
        "input_sha256": input_digest,
        "command_file": "logs/command.txt",
        "stdout_file": "logs/stdout_stderr.txt",
        "telemetry_file": "telemetry/nvidia_smi.csv",
        "gradient_summary_file": "backward/gradient_summary.json",
        "wall_time_seconds": 10.0,
        "exit_code": 0,
    }
    write_json(root / "manifest.json", manifest)


class ValidateG1ArtifactsTests(unittest.TestCase):
    def test_valid_bundle_passes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_valid_run(root)
            self.assertEqual(validate_run(root), [])

    def test_missing_gradient_evidence_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_valid_run(root)
            (root / "backward" / "gradient_summary.json").unlink()
            errors = validate_run(root)
            self.assertTrue(any("gradient summary" in error for error in errors))

    def test_input_digest_mismatch_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_valid_run(root)
            (root / "inputs" / "input.json").write_text("{}", encoding="utf-8")
            errors = validate_run(root)
            self.assertIn("input JSON SHA-256 does not match manifest", errors)

    def test_later_cutoff_model_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_valid_run(root)
            lock_path = root / "backbone_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["model_name"] = "protenix_base_20250630_v1.0.0"
            write_json(lock_path, lock)
            errors = validate_run(root)
            self.assertTrue(any("primary model" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

