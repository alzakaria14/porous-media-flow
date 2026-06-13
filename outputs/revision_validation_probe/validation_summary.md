# Multi-seed validation summary

Runs: 1
Seeds: 42
Device: NVIDIA GeForce RTX 3050
FDM grid: 51 x 51, Nt=60, dt=0.100000 h, rx+ry=0.044444

## Averaged over forecast hours 4-6

| Metric | NN mean ± SD | PINN mean ± SD | Mean reduction | paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE (m) | 0.032617 ± nan | 0.021839 ± nan | 33.04% ± nan% | nan | 0.5 |
| MAE (m) | 0.027984 ± nan | 0.015566 ± nan | 44.38% ± nan% | nan | 0.5 |

## Timing

Mean NN training time: 4.89 s
Mean PINN training time: 54.30 s
Mean FDM solve time: 0.0271 s
