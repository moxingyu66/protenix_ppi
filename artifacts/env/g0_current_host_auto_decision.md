# Protenix-PPI G0 automatic environment decision

- Source report: `D:\hitsz\research\protenix_ppi\artifacts\env\environment_windows_20260915_110802.txt`
- GPU count: 1
- GPU model(s): NVIDIA GeForce RTX 3050 Laptop GPU
- Minimum memory per GPU: 4.0 GiB
- Maximum free filesystem: 402.1 GiB
- Docker detected: no
- Route: **D**

This GPU is unsuitable for primary Protenix; use it only for CPU work or a Mini smoke test.

This route classification is not a G1 pass. Protenix inference and a real backward step still require direct execution evidence.

## Cautions

- Docker was not detected; confirm an alternative reproducible Linux runtime such as Apptainer or Conda.
- No single reported filesystem has 1.5 TiB free; do not download the official full training package here.
- Do not use this machine to judge primary-model inference or gradient feasibility.
