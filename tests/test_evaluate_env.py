from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.evaluate_env import (
    choose_route,
    detect_docker,
    evaluate,
    parse_gpus,
    parse_maximum_free_disk_bytes,
)


WINDOWS_SAMPLE = """
===== NVIDIA inventory =====
0, NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 616.92

===== Filesystem capacity =====
Name         Used         Free Root
C    112264491008 262352912384 C:\\
D    192327065600 431732617216 D:\\

===== Docker version =====
UNAVAILABLE: docker not found
"""

LINUX_SAMPLE = """
===== NVIDIA inventory =====
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 550.54.15

===== Filesystem capacity =====
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  3.5T  900G  2.6T  26% /data

===== Docker =====
Docker version 27.0.1, build test
"""


class EvaluateEnvironmentTests(unittest.TestCase):
    def test_parse_windows_report(self):
        gpus = parse_gpus(WINDOWS_SAMPLE)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].memory_mib, 4096)
        self.assertIn("RTX 3050", gpus[0].name)
        self.assertEqual(parse_maximum_free_disk_bytes(WINDOWS_SAMPLE), 431732617216)
        self.assertFalse(detect_docker(WINDOWS_SAMPLE))
        self.assertEqual(choose_route(gpus[0].memory_mib)[0], "D")

    def test_parse_linux_report(self):
        gpus = parse_gpus(LINUX_SAMPLE)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(min(gpu.memory_mib for gpu in gpus), 81920)
        self.assertGreater(parse_maximum_free_disk_bytes(LINUX_SAMPLE), 2 * 1024**4)
        self.assertTrue(detect_docker(LINUX_SAMPLE))
        self.assertEqual(choose_route(81920)[0], "A")

    def test_evaluate_emits_multi_gpu_caution(self):
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.txt"
            report.write_text(LINUX_SAMPLE, encoding="utf-8")
            decision = evaluate(report)
        self.assertEqual(decision.route, "A")
        self.assertTrue(decision.full_official_training_data_feasible)
        self.assertTrue(any("Multiple GPUs" in item for item in decision.cautions))

    def test_unknown_when_inventory_missing(self):
        self.assertEqual(choose_route(None)[0], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()

