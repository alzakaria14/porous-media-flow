# Multi-seed validation summary

Runs: 10
Seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
Device: NVIDIA GeForce RTX 3050
FDM grid: 51 x 51, Nt=60, dt=0.100000 h, rx+ry=0.044444

## Averaged over forecast hours 4-6

| Metric | NN mean ± SD | PINN mean ± SD | Mean reduction | paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE (m) | 0.031647 ± 0.002066 | 0.022489 ± 0.000583 | 28.59% ± 6.13% | 5.265e-07 | 0.0009766 |
| MAE (m) | 0.027571 ± 0.001682 | 0.016020 ± 0.000459 | 41.63% ± 5.06% | 1.198e-08 | 0.0009766 |

## Timing

Mean NN training time: 4.49 s
Mean PINN training time: 53.22 s
Mean FDM solve time: 0.0242 s
