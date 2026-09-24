# Gate G0 result — current Windows host

Date: 2026-09-15  
Evidence: `../artifacts/env/environment_windows_20260915_110802.txt`  
Decision: **NEEDS INFORMATION — a separate Linux GPU execution environment is required**

## Observed facts

| Item | Observed value | Interpretation |
|---|---|---|
| GPU | 1 × NVIDIA GeForce RTX 3050 Laptop GPU | Development/display GPU only |
| GPU memory | 4096 MiB | Insufficient for the primary Protenix model |
| Driver | 616.92 | Recorded; compatibility must be checked on the actual execution node instead |
| CUDA reported by driver | CUDA UMD 13.4 | This is not proof that a compatible training toolkit or PyTorch build is installed |
| Free space on D: | approximately 432 GB | Insufficient for the official full training-data package (at least 1.5 TB) |
| WSL | not usable in the collected environment | No Linux training runtime available locally |
| Docker | unavailable | No containerized training runtime available locally |
| Python | 3.14.6 | Not selected for Protenix; a supported isolated environment will be needed |
| Git | 2.55.0 | Available |
| CPU/RAM | not collected because CIM access was denied | Must be collected on the actual execution node |

## Decision

This Windows laptop must not be used to judge Protenix inference or fine-tuning feasibility. Four gigabytes of VRAM is below the intended operating range even for the primary smoke test, and the available local disk cannot host the full official training corpus.

The next G0 action is to run `../scripts/collect_env.sh` on the actual Linux GPU server, Slurm node, or approved cloud instance. G0 remains open until that report records per-GPU memory, usable storage, runtime isolation, scheduler limits, and software-installation policy.

## Safe use of the current host

The current machine remains suitable for:

- protocol and code development;
- dataset metadata cleaning;
- small CPU baselines;
- split generation and leakage audits;
- report writing and result visualization.

It is not suitable for primary Protenix inference, representation extraction, or gradient-based experiments.

