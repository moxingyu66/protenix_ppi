# Gate G0 checklist — environment and backbone feasibility

## Evidence required

- [ ] The actual execution machine or cluster has been identified.
- [ ] GPU model, count, and memory per GPU are recorded.
- [ ] NVIDIA driver is visible from the intended runtime.
- [ ] CPU, RAM, fast-disk capacity, and general-disk capacity are recorded.
- [ ] Docker, Apptainer, Conda, or another reproducible runtime has been selected.
- [ ] Scheduler partition and maximum wall time are recorded, if applicable.
- [ ] Internet and software-installation restrictions are known.
- [ ] The primary model is fixed to `protenix_base_default_v1.0.0`.
- [ ] The primary pretraining cutoff is recorded as `2021-09-30`.
- [ ] The later-cutoff `protenix_base_20250630_v1.0.0` is excluded from the primary temporal evaluation.
- [ ] No full training database has been downloaded prematurely.
- [ ] No PPI result has been used to revise the preregistered primary test definition.

## Route chosen after review

- [ ] Route A: at least one 80 GB GPU and sufficient storage; full staged validation is technically plausible.
- [ ] Route B: 40–48 GB per GPU; inference, frozen features, head tuning, and tightly cropped experiments are plausible.
- [ ] Route C: approximately 24 GB per GPU; prioritize inference/frozen features; structural backward passes require a separate feasibility decision.
- [ ] Route D: below 24 GB; use Mini only for smoke testing and secure a larger execution environment for the primary study.

## Gate result

- Decision: PASS / FAIL / NEEDS INFORMATION
- Reviewer:
- Date:
- Evidence file:
- Notes:

