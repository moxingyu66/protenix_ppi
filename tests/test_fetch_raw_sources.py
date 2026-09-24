import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protenix_ppi.scripts.fetch_raw_sources import fetch_one, load_registry


def source_record(url: str, content: bytes) -> dict:
    return {
        "source_id": "synthetic_source",
        "source_database": "Synthetic",
        "source_version": "v1",
        "url": url,
        "filename": "source.tsv",
        "expected_bytes": len(content),
        "expected_sha256": hashlib.sha256(content).hexdigest(),
        "expected_line_count": len(content.splitlines()),
        "download_class": "core",
        "requires_insecure_tls": False,
        "license_status": "test_only",
        "license_evidence_url": "https://example.invalid/license",
        "citation": "synthetic test",
        "notes": "test data",
    }


class FetchRawSourcesTests(unittest.TestCase):
    def test_local_fetch_validates_and_reuses_immutable_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"A\tB\nC\tD\n"
            origin = root / "origin.tsv"
            origin.write_bytes(content)
            source = source_record(origin.as_uri(), content)
            first = fetch_one(source, root / "raw", False, False)
            target = root / "raw" / "Synthetic" / "v1" / "source.tsv"
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(first["observed"]["line_count"], 2)
            self.assertEqual(first["observed"]["sha256"], source["expected_sha256"])

            origin.write_bytes(b"changed upstream\n")
            second = fetch_one(source, root / "raw", False, False)
            self.assertEqual(second, first)
            self.assertEqual(target.read_bytes(), content)

    def test_hash_mismatch_is_preserved_as_invalid_and_not_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"actual\n"
            origin = root / "origin.tsv"
            origin.write_bytes(content)
            source = source_record(origin.as_uri(), content)
            source["expected_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "failed validation"):
                fetch_one(source, root / "raw", False, False)
            target_dir = root / "raw" / "Synthetic" / "v1"
            self.assertFalse((target_dir / "source.tsv").exists())
            self.assertTrue((target_dir / "source.tsv.partial.invalid").exists())

    def test_large_download_requires_explicit_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"x\n"
            origin = root / "origin.tsv"
            origin.write_bytes(content)
            source = source_record(origin.as_uri(), content)
            source["download_class"] = "large"
            with self.assertRaisesRegex(ValueError, "allow-large-download"):
                fetch_one(source, root / "raw", False, False)

    def test_insecure_tls_requires_explicit_flag(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"x\n"
            origin = root / "origin.tsv"
            origin.write_bytes(content)
            source = source_record(origin.as_uri(), content)
            source["requires_insecure_tls"] = True
            with self.assertRaisesRegex(ValueError, "allow-insecure-tls"):
                fetch_one(source, root / "raw", False, False)

    def test_registry_rejects_duplicate_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"x\n"
            origin = root / "origin.tsv"
            source = source_record(origin.as_uri(), content)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps({"sources": [source, dict(source)]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_registry(registry)


if __name__ == "__main__":
    unittest.main()
