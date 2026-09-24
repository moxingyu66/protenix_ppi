# Metric cross-check

Date: 2026-09-15  
Local Python: 3.14.6  
Reference package: scikit-learn 1.9.0  
Random generator seed: 20260915  
Sample size: 200  
Positive prevalence: 0.25

The dependency-free project implementation was compared with scikit-learn on the same deterministic, non-tied predictions.

| Metric | Project implementation | scikit-learn | Absolute difference |
|---|---:|---:|---:|
| Average precision | 0.299944053606367 | 0.299944053606367 | 0 |
| AUROC | 0.602533333333333 | 0.602533333333333 | 0 |

The command asserted both absolute differences were below `1e-12` and exited successfully. Tie behavior is covered separately by unit tests: equal-score groups enter at one threshold for AP, receive 0.5 credit in AUROC, and use expected class composition at a top-K boundary.

