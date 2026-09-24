import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from protenix_ppi.scripts.filter_intact_archive import filter_archive


FIELDS = [
    "#ID(s) interactor A",
    "ID(s) interactor B",
    "Alt. ID(s) interactor A",
    "Alt. ID(s) interactor B",
    "Interaction detection method(s)",
    "Publication Identifier(s)",
    "Taxid interactor A",
    "Taxid interactor B",
    "Interaction type(s)",
    "Source database(s)",
    "Interaction identifier(s)",
    "Expansion method(s)",
    "Type(s) interactor A",
    "Type(s) interactor B",
    "Negative",
]


def row(identifier: str, interaction: str, method: str, taxid_b: str = "taxid:9606(human)"):
    return {
        "#ID(s) interactor A": "uniprotkb:P00001",
        "ID(s) interactor B": "uniprotkb:P00002",
        "Alt. ID(s) interactor A": "-",
        "Alt. ID(s) interactor B": "-",
        "Interaction detection method(s)": f'psi-mi:"{method}"(method)',
        "Publication Identifier(s)": "pubmed:1",
        "Taxid interactor A": "taxid:9606(human)",
        "Taxid interactor B": taxid_b,
        "Interaction type(s)": f'psi-mi:"{interaction}"(type)',
        "Source database(s)": 'psi-mi:"MI:0469"(intact)',
        "Interaction identifier(s)": f"intact:{identifier}",
        "Expansion method(s)": "-",
        "Type(s) interactor A": 'psi-mi:"MI:0326"(protein)',
        "Type(s) interactor B": 'psi-mi:"MI:0326"(protein)',
        "Negative": "false",
    }


def build_fixture(root: Path):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row("EBI-1", "MI:0407", "MI:0018"))
    writer.writerow(row("EBI-2", "MI:0915", "MI:0018"))
    writer.writerow(row("EBI-3", "MI:0407", "MI:0096"))
    writer.writerow(row("EBI-4", "MI:0407", "MI:0018", "taxid:10090(mouse)"))
    multi = row("EBI-5", "MI:0407", "MI:0018")
    multi["Interaction detection method(s)"] += '|psi-mi:"MI:0107"(method)'
    writer.writerow(multi)

    archive = root / "human.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("human.txt", buffer.getvalue().encode("utf-8"))
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_id": "intact_human_archive",
                "source_version": "test",
                "observed": {
                    "valid": True,
                    "bytes": archive.stat().st_size,
                    "sha256": archive_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    policy = root / "compiled.json"
    named = lambda values: [{"id": value, "name": value} for value in values]
    policy.write_text(
        json.dumps(
            {
                "policy_id": "test",
                "ontology_commit": "a" * 40,
                "ontology_sha256": "b" * 64,
                "interaction_type": {
                    "auto_accept_exact": named(["MI:0407"]),
                    "direct_descendants_manual_review": named(["MI:0195"]),
                    "never_auto_accept_exact": named(["MI:0914", "MI:0915"]),
                },
                "detection_method": {
                    "auto_accept_non_obsolete_closure": named(["MI:0018", "MI:0107"]),
                    "manual_review_members_removed_from_auto_closure": named([]),
                    "never_auto_accept_closure": named(["MI:0004", "MI:0096"]),
                },
            }
        ),
        encoding="utf-8",
    )
    return archive, manifest, policy


class FilterIntactArchiveTests(unittest.TestCase):
    def test_stream_filter_separates_candidate_review_and_excluded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest, policy = build_fixture(root)
            result = filter_archive(archive, manifest, policy, root / "result", 0)
            self.assertEqual(result["counts"]["rows_total"], 5)
            self.assertEqual(result["counts"]["decision_candidate"], 1)
            self.assertEqual(result["counts"]["decision_review"], 2)
            self.assertEqual(result["counts"]["decision_excluded"], 2)
            with (root / "result" / "intact_direct_candidates.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidates = list(csv.DictReader(handle, delimiter="\t"))
            with (root / "result" / "intact_manual_review.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                reviews = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([item["source_record_id"] for item in candidates], ["intact:EBI-1"])
            self.assertEqual(
                {item["decision_reason"] for item in reviews},
                {"detection_method_requires_review", "missing_or_ambiguous_detection_method"},
            )

    def test_rejects_archive_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest, policy = build_fixture(root)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["observed"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                filter_archive(archive, manifest, policy, root / "result", 0)

    def test_rejects_wrong_zip_member(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest, policy = build_fixture(root)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/human.txt", "not the expected member")
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["observed"]["bytes"] = archive.stat().st_size
            data["observed"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root-level human.txt"):
                filter_archive(archive, manifest, policy, root / "result", 0)


if __name__ == "__main__":
    unittest.main()
