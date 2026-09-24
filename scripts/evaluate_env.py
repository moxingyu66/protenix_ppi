#!/usr/bin/env python3
"""Classify a Protenix execution environment from a G0 collector report.

The classifier intentionally uses per-GPU memory, not aggregate GPU memory.
It emits a compact, shareable summary and never claims that a route has passed
the Protenix smoke test; that requires actual inference and backward evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


GIB = 1024**3
GPU_LINE = re.compile(
    r"^\s*(?P<index>\d+)\s*,\s*(?P<name>.+?)\s*,\s*"
    r"(?P<memory>\d+)\s*MiB\s*,\s*(?P<driver>[^\r\n]+?)\s*$",
    re.MULTILINE,
)
WINDOWS_DRIVE_LINE = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9_-]*\s+\d+\s+(?P<free>\d+)\s+[A-Za-z]:\\?\s*$",
    re.MULTILINE,
)
LINUX_DF_LINE = re.compile(
    r"^\S+\s+(?P<size>\S+)\s+(?P<used>\S+)\s+(?P<avail>\S+)\s+\S+\s+\S+\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_mib: int
    driver: str


@dataclass(frozen=True)
class EnvironmentDecision:
    source_report: str
    gpu_count: int
    minimum_memory_mib: int | None
    gpu_names: list[str]
    driver_versions: list[str]
    maximum_free_disk_gib: float | None
    docker_detected: bool
    route: str
    primary_capability: str
    full_official_training_data_feasible: bool | None
    cautions: list[str]


def _section(text: str, title: str) -> str:
    marker = f"===== {title} ====="
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    next_marker = text.find("=====", start)
    return text[start:] if next_marker < 0 else text[start:next_marker]


def _size_to_bytes(value: str) -> int | None:
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([kmgtpe]?)(?:i?b)?", value.strip())
    if not match:
        return None
    number = float(match.group(1))
    units = {"": 1, "k": 1024, "m": 1024**2, "g": GIB, "t": 1024**4, "p": 1024**5, "e": 1024**6}
    return int(number * units[match.group(2).lower()])


def parse_gpus(text: str) -> list[GpuInfo]:
    inventory = _section(text, "NVIDIA inventory")
    return [
        GpuInfo(
            index=int(match.group("index")),
            name=match.group("name").strip(),
            memory_mib=int(match.group("memory")),
            driver=match.group("driver").strip(),
        )
        for match in GPU_LINE.finditer(inventory)
    ]


def parse_maximum_free_disk_bytes(text: str) -> int | None:
    section = _section(text, "Filesystem capacity")
    candidates = [int(match.group("free")) for match in WINDOWS_DRIVE_LINE.finditer(section)]

    for match in LINUX_DF_LINE.finditer(section):
        available = _size_to_bytes(match.group("avail"))
        if available is not None:
            candidates.append(available)

    return max(candidates) if candidates else None


def detect_docker(text: str) -> bool:
    section = _section(text, "Docker version") or _section(text, "Docker")
    lowered = section.lower()
    return bool(section.strip()) and "unavailable" not in lowered and "not found" not in lowered


def choose_route(minimum_memory_mib: int | None) -> tuple[str, str]:
    if minimum_memory_mib is None:
        return "UNKNOWN", "GPU inventory must be collected before an execution route can be selected."
    if minimum_memory_mib >= 80 * 1024:
        return "A", "Full staged validation is technically plausible, subject to a smoke test and storage audit."
    if minimum_memory_mib >= 40 * 1024:
        return "B", "Inference, frozen features, head tuning, and tightly controlled cropped fine-tuning are plausible."
    if minimum_memory_mib >= 20 * 1024:
        return "C", "Prioritize inference and frozen features; structural backward passes need a separate feasibility gate."
    return "D", "This GPU is unsuitable for primary Protenix; use it only for CPU work or a Mini smoke test."


def evaluate(report_path: Path) -> EnvironmentDecision:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    gpus = parse_gpus(text)
    minimum_memory_mib = min((gpu.memory_mib for gpu in gpus), default=None)
    maximum_free_disk_bytes = parse_maximum_free_disk_bytes(text)
    route, capability = choose_route(minimum_memory_mib)

    disk_gib = None if maximum_free_disk_bytes is None else maximum_free_disk_bytes / GIB
    full_data_feasible = None if disk_gib is None else disk_gib >= 1.5 * 1024
    cautions: list[str] = []

    if len(gpus) > 1:
        cautions.append("Multiple GPUs do not combine into one larger per-sample memory pool under ordinary data parallelism.")
    if not detect_docker(text):
        cautions.append("Docker was not detected; confirm an alternative reproducible Linux runtime such as Apptainer or Conda.")
    if full_data_feasible is False:
        cautions.append("No single reported filesystem has 1.5 TiB free; do not download the official full training package here.")
    if minimum_memory_mib is not None and minimum_memory_mib < 20 * 1024:
        cautions.append("Do not use this machine to judge primary-model inference or gradient feasibility.")
    if not gpus:
        cautions.append("No parseable NVIDIA inventory was found.")

    return EnvironmentDecision(
        source_report=str(report_path),
        gpu_count=len(gpus),
        minimum_memory_mib=minimum_memory_mib,
        gpu_names=sorted({gpu.name for gpu in gpus}),
        driver_versions=sorted({gpu.driver for gpu in gpus}),
        maximum_free_disk_gib=None if disk_gib is None else round(disk_gib, 1),
        docker_detected=detect_docker(text),
        route=route,
        primary_capability=capability,
        full_official_training_data_feasible=full_data_feasible,
        cautions=cautions,
    )


def render_markdown(decision: EnvironmentDecision) -> str:
    memory = "unknown" if decision.minimum_memory_mib is None else f"{decision.minimum_memory_mib / 1024:.1f} GiB"
    disk = "unknown" if decision.maximum_free_disk_gib is None else f"{decision.maximum_free_disk_gib:.1f} GiB"
    gpu_names = ", ".join(decision.gpu_names) or "none parsed"
    lines = [
        "# Protenix-PPI G0 automatic environment decision",
        "",
        f"- Source report: `{decision.source_report}`",
        f"- GPU count: {decision.gpu_count}",
        f"- GPU model(s): {gpu_names}",
        f"- Minimum memory per GPU: {memory}",
        f"- Maximum free filesystem: {disk}",
        f"- Docker detected: {'yes' if decision.docker_detected else 'no'}",
        f"- Route: **{decision.route}**",
        "",
        decision.primary_capability,
        "",
        "This route classification is not a G1 pass. Protenix inference and a real backward step still require direct execution evidence.",
    ]
    if decision.cautions:
        lines.extend(["", "## Cautions", ""])
        lines.extend(f"- {item}" for item in decision.cautions)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to a collect_env report")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    parser.add_argument("--output", type=Path, help="Optional output path")
    args = parser.parse_args()

    decision = evaluate(args.report.resolve())
    content = json.dumps(asdict(decision), indent=2, ensure_ascii=False) + "\n" if args.json else render_markdown(decision)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

