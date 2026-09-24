#!/usr/bin/env python3
"""Fetch versioned PPI sources with immutable manifests and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOWNLOAD_CLASSES = {"core", "restricted_core", "large"}
LARGE_DEFAULT_LIMIT = 100 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
        raise ValueError("source registry must be a JSON object containing a sources list")
    seen: set[str] = set()
    for index, source in enumerate(value["sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"sources[{index}] must be an object")
        required = {
            "source_id", "source_database", "source_version", "url", "filename",
            "expected_bytes", "expected_sha256", "expected_line_count", "download_class",
            "requires_insecure_tls", "license_status", "license_evidence_url", "citation", "notes",
        }
        missing = required - set(source)
        if missing:
            raise ValueError(f"sources[{index}] is missing: {', '.join(sorted(missing))}")
        source_id = str(source["source_id"])
        if not source_id or source_id in seen:
            raise ValueError(f"duplicate or empty source_id: {source_id!r}")
        seen.add(source_id)
        filename = str(source["filename"])
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError(f"{source_id}: filename must be a plain basename")
        if not str(source["url"]).startswith(("https://", "file://")):
            raise ValueError(f"{source_id}: only HTTPS or local file URLs are allowed")
        if not isinstance(source["expected_bytes"], int) or source["expected_bytes"] <= 0:
            raise ValueError(f"{source_id}: expected_bytes must be a positive integer")
        expected_hash = source["expected_sha256"]
        if expected_hash is not None and not SHA256_RE.fullmatch(str(expected_hash)):
            raise ValueError(f"{source_id}: expected_sha256 is invalid")
        expected_lines = source["expected_line_count"]
        if expected_lines is not None and (not isinstance(expected_lines, int) or expected_lines <= 0):
            raise ValueError(f"{source_id}: expected_line_count must be a positive integer or null")
        if source["download_class"] not in DOWNLOAD_CLASSES:
            raise ValueError(f"{source_id}: invalid download_class")
        if not isinstance(source["requires_insecure_tls"], bool):
            raise ValueError(f"{source_id}: requires_insecure_tls must be boolean")
        for key in ("license_status", "license_evidence_url", "citation"):
            if not str(source[key]).strip():
                raise ValueError(f"{source_id}: {key} must not be empty")
    return value


def validate_download(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    actual_lines = count_lines(path) if source["expected_line_count"] is not None else None
    errors: list[str] = []
    if actual_bytes != source["expected_bytes"]:
        errors.append(f"byte size expected {source['expected_bytes']}, observed {actual_bytes}")
    if source["expected_sha256"] is not None and actual_hash != source["expected_sha256"]:
        errors.append(f"SHA-256 expected {source['expected_sha256']}, observed {actual_hash}")
    if source["expected_line_count"] is not None and actual_lines != source["expected_line_count"]:
        errors.append(
            f"line count expected {source['expected_line_count']}, observed {actual_lines}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "line_count": actual_lines,
    }


def fetch_one(
    source: dict[str, Any],
    output_root: Path,
    allow_insecure_tls: bool,
    allow_large_download: bool,
    large_limit: int = LARGE_DEFAULT_LIMIT,
) -> dict[str, Any]:
    source_id = source["source_id"]
    if source["requires_insecure_tls"] and not allow_insecure_tls:
        raise ValueError(
            f"{source_id}: official host requires disabled TLS verification; rerun with "
            "--allow-insecure-tls only if exact hash verification is acceptable"
        )
    if (source["download_class"] == "large" or source["expected_bytes"] > large_limit) and not allow_large_download:
        raise ValueError(
            f"{source_id}: download is {source['expected_bytes']} bytes; rerun with "
            "--allow-large-download after storage review"
        )
    target_dir = output_root / source["source_database"] / source["source_version"]
    target = target_dir / source["filename"]
    manifest_path = target_dir / f"{source_id}.manifest.json"
    if target.exists():
        validation = validate_download(target, source)
        if not validation["valid"]:
            raise ValueError(
                f"{source_id}: existing immutable raw file is invalid; preserve and investigate: "
                + "; ".join(validation["errors"])
            )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["observed"] = validation
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return manifest
        response_metadata = {"reused_existing_file": True}
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        if partial.exists():
            raise ValueError(f"{source_id}: partial download already exists: {partial}")
        context = ssl._create_unverified_context() if source["requires_insecure_tls"] else None
        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Protenix-PPI-source-audit/0.1", "Accept-Encoding": "identity"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                response_metadata = {
                    "reused_existing_file": False,
                    "final_url": response.geturl(),
                    "http_status": getattr(response, "status", None),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                    "content_length_header": response.headers.get("Content-Length"),
                }
                with partial.open("xb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise
        validation = validate_download(partial, source)
        if not validation["valid"]:
            invalid_path = partial.with_name(partial.name + ".invalid")
            os.replace(partial, invalid_path)
            raise ValueError(
                f"{source_id}: downloaded content failed validation and was preserved at {invalid_path}: "
                + "; ".join(validation["errors"])
            )
        os.replace(partial, target)

    validation = validate_download(target, source)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "source_id": source_id,
        "source_database": source["source_database"],
        "source_version": source["source_version"],
        "registry_url": source["url"],
        "local_file": source["filename"],
        "retrieved_at": retrieved_at,
        "tls_verification_disabled": bool(source["requires_insecure_tls"]),
        "license_status": source["license_status"],
        "license_evidence_url": source["license_evidence_url"],
        "citation": source["citation"],
        "notes": source["notes"],
        "observed": validation,
        "response": response_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--class", dest="download_class", choices=sorted(DOWNLOAD_CLASSES))
    parser.add_argument("--allow-insecure-tls", action="store_true")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    sources = registry["sources"]
    by_id = {source["source_id"]: source for source in sources}
    if args.source_id:
        unknown = sorted(set(args.source_id) - set(by_id))
        if unknown:
            raise ValueError(f"unknown source IDs: {', '.join(unknown)}")
        selected = [by_id[source_id] for source_id in args.source_id]
    elif args.download_class:
        selected = [source for source in sources if source["download_class"] == args.download_class]
    else:
        selected = [source for source in sources if source["download_class"] == "core"]
    if not selected:
        raise ValueError("no sources selected")
    manifests = []
    for source in selected:
        manifest = fetch_one(
            source,
            args.output_root,
            args.allow_insecure_tls,
            args.allow_large_download,
        )
        manifests.append(manifest)
        print(f"PASS {source['source_id']}: {manifest['observed']['sha256']}")
    summary = {
        "registry_id": registry.get("registry_id"),
        "selected_source_ids": [source["source_id"] for source in selected],
        "manifest_count": len(manifests),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
