# Multi-seed validation summary

Runs: 1
Seeds: 42
Device: NVIDIA GeForce RTX 3050
FDM grid: 51 x 51, Nt=60, dt=0.100000 h, rx+ry=0.044444

## Averaged over forecast hours 4-6

| Metric | NN mean ± SD | PINN mean ± SD | Mean reduction | paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE (m) | 0.033852 ± nan | 0.022723 ± nan | 32.88% ± nan% | nan | 0.5 |
| MAE (m) | 0.030332 ± nan | 0.015782 ± nan | 47.97% ± nan% | nan | 0.5 |

## Timing

Mean NN training time: 1.63 s
Mean PINN training time: 13.06 s
Mean FDM solve time: 0.0291 s
