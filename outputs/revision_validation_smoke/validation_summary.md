# Multi-seed validation summary

Runs: 1
Seeds: 1001
Device: NVIDIA GeForce RTX 3050
FDM grid: 51 x 51, Nt=60, dt=0.100000 h, rx+ry=0.044444

## Averaged over forecast hours 4-6

| Metric | NN mean ± SD | PINN mean ± SD | Mean reduction | paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE (m) | 0.029316 ± nan | 0.023576 ± nan | 19.58% ± nan% | nan | 0.5 |
| MAE (m) | 0.026387 ± nan | 0.016013 ± nan | 39.31% ± nan% | nan | 0.5 |

## Timing

Mean NN training time: 1.88 s
Mean PINN training time: 12.59 s
Mean FDM solve time: 0.0287 s
