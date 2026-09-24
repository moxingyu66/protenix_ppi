#!/usr/bin/env python3
"""Fetch pinned UniProt ID-mapping snapshots for a prepared inventory.

The UniProt REST service is queried only for identifiers listed in the hashed
request files.  Raw mapping tables are preserved, and one-to-many or unmapped
identifiers are reported rather than resolved heuristically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


API_ROOT = "https://rest.uniprot.org"
RELEASE_RE = re.compile(r"UniProt Knowledgebase Release (\d{4}_\d{2})")
RELEASE_DATE_RE = re.compile(r"Release \d{4}_\d{2} of (\d{2}-[A-Za-z]{3}-\d{4})")
USER_AGENT = "Protenix-PPI-benchmark/0.1 (reproducible academic mapping snapshot)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_request_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"mapping request is empty: {path}")
    if ids != sorted(set(ids)):
        raise ValueError(f"mapping request must be sorted and unique: {path}")
    if any("," in item or any(char.isspace() for char in item) for item in ids):
        raise ValueError(f"mapping request contains an invalid identifier: {path}")
    return ids


def read_release_evidence(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    release_match = RELEASE_RE.search(text)
    dates = set(RELEASE_DATE_RE.findall(text))
    if not release_match or len(dates) != 1:
        raise ValueError("UniProt release evidence is malformed or internally inconsistent")
    return release_match.group(1), next(iter(dates))


def parse_mapping_tsv(payload: bytes, requested: set[str]) -> tuple[list[tuple[str, str]], dict]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["From", "To"]:
        raise ValueError(f"unexpected UniProt mapping columns: {reader.fieldnames}")
    rows: list[tuple[str, str]] = []
    targets: defaultdict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(reader, start=2):
        source = (row.get("From") or "").strip()
        target = (row.get("To") or "").strip()
        if source not in requested:
            raise ValueError(f"mapping line {line_number} contains an unrequested source: {source}")
        if not target:
            raise ValueError(f"mapping line {line_number} has an empty target")
        rows.append((source, target))
        targets[source].add(target)
    mapped = set(targets)
    multiplicity = Counter(len(values) for values in targets.values())
    summary = {
        "requested_ids": len(requested),
        "mapping_rows": len(rows),
        "mapped_source_ids": len(mapped),
        "unmapped_source_ids": len(requested - mapped),
        "one_to_one_source_ids": multiplicity.get(1, 0),
        "one_to_many_source_ids": sum(count for degree, count in multiplicity.items() if degree > 1),
        "unique_target_ids": len({target for values in targets.values() for target in values}),
    }
    return rows, summary


def _request(url: str, data: bytes | None = None, timeout: int = 120) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), {key.lower(): value for key, value in response.headers.items()}
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == 4:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"UniProt request failed after retries: {url}") from last_error


def _submit_job(from_namespace: str, ids: list[str]) -> str:
    body = urllib.parse.urlencode(
        {"from": from_namespace, "to": "UniProtKB", "ids": ",".join(ids)}
    ).encode("ascii")
    payload, _ = _request(f"{API_ROOT}/idmapping/run", data=body)
    response = json.loads(payload)
    job_id = response.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"UniProt did not return a job ID: {response}")
    return job_id


def _wait_for_job(job_id: str, timeout_seconds: int, poll_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload, _ = _request(f"{API_ROOT}/idmapping/details/{job_id}")
        details = json.loads(payload)
        if details.get("redirectURL"):
            return details
        time.sleep(poll_seconds)
    raise TimeoutError(f"UniProt mapping job {job_id} did not finish within {timeout_seconds}s")


def _validate_release_headers(headers: dict[str, str], release: str, release_date: str) -> None:
    observed_release = headers.get("x-uniprot-release")
    observed_date = headers.get("x-uniprot-release-date")
    if observed_release != release:
        raise ValueError(f"UniProt API release drift: expected {release}, observed {observed_release}")
    expected_api_date = time.strftime("%d-%B-%Y", time.strptime(release_date, "%d-%b-%Y"))
    if observed_date != expected_api_date:
        raise ValueError(
            f"UniProt API release-date drift: expected {expected_api_date}, observed {observed_date}"
        )


def fetch_one_mapping(
    namespace: str,
    request_path: Path,
    output_dir: Path,
    release: str,
    release_date: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict:
    ids = read_request_ids(request_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "mapping.tsv"
    failed_path = output_dir / "unmapped_ids.txt"
    metadata_path = output_dir / "metadata.json"
    if any(path.exists() for path in (mapping_path, failed_path, metadata_path)):
        raise ValueError(f"mapping output already exists; preserve immutable snapshots: {output_dir}")

    job_id = _submit_job(namespace, ids)
    details = _wait_for_job(job_id, timeout_seconds, poll_seconds)
    encoded_job = urllib.parse.quote(job_id, safe="")
    payload, headers = _request(f"{API_ROOT}/idmapping/stream/{encoded_job}?format=tsv", timeout=300)
    _validate_release_headers(headers, release, release_date)
    rows, summary = parse_mapping_tsv(payload, set(ids))
    mapped_sources = {source for source, _ in rows}
    unmapped = sorted(set(ids) - mapped_sources)

    mapping_partial = mapping_path.with_suffix(".tsv.partial")
    failed_partial = failed_path.with_suffix(".txt.partial")
    mapping_partial.write_bytes(payload)
    failed_partial.write_text("\n".join(unmapped) + ("\n" if unmapped else ""), encoding="utf-8")
    os.replace(mapping_partial, mapping_path)
    os.replace(failed_partial, failed_path)

    metadata = {
        "schema_version": "1.0",
        "method": "uniprot_idmapping_rest_v0.1",
        "from": namespace,
        "to": "UniProtKB",
        "job_id": job_id,
        "redirect_url": details.get("redirectURL"),
        "uniprot_release": release,
        "uniprot_release_date": release_date,
        "api_headers": {
            "x-uniprot-release": headers.get("x-uniprot-release"),
            "x-uniprot-release-date": headers.get("x-uniprot-release-date"),
            "x-api-deployment-date": headers.get("x-api-deployment-date"),
        },
        "request": {
            "path": str(request_path.resolve()),
            "sha256": sha256_file(request_path),
            "count": len(ids),
        },
        "mapping": {
            "path": str(mapping_path.resolve()),
            "sha256": sha256_file(mapping_path),
            **summary,
        },
        "unmapped": {
            "path": str(failed_path.resolve()),
            "sha256": sha256_file(failed_path),
            "count": len(unmapped),
        },
        "decision_warning": (
            "The raw mapping snapshot does not authorize automatic selection among multiple targets. "
            "One-to-many and unmapped sources must remain quarantined."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembl-ids", type=Path)
    parser.add_argument("--uniprot-ids", type=Path)
    parser.add_argument("--release-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=3)
    args = parser.parse_args()
    release, release_date = read_release_evidence(args.release_evidence)
    requests = [
        item
        for item in (
            ("Ensembl", args.ensembl_ids, "ensembl_to_uniprotkb"),
            ("UniProtKB_AC-ID", args.uniprot_ids, "uniprotkb_to_uniprotkb"),
        )
        if item[1] is not None
    ]
    if not requests:
        parser.error("at least one of --ensembl-ids or --uniprot-ids is required")
    results = {}
    for namespace, request_path, dirname in requests:
        results[namespace] = fetch_one_mapping(
            namespace,
            request_path,
            args.output_dir / dirname,
            release,
            release_date,
            args.timeout_seconds,
            args.poll_seconds,
        )
    manifest = {
        "schema_version": "1.0",
        "snapshot_id": f"uniprot_mapping_{release}",
        "uniprot_release": release,
        "uniprot_release_date": release_date,
        "release_evidence": {
            "path": str(args.release_evidence.resolve()),
            "sha256": sha256_file(args.release_evidence),
        },
        "mappings": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
