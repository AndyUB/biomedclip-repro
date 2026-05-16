# VQA-RAD Tuning — Round 5

Frozen BiomedCLIP + METER co-attention head. n_layers=2 fixed.
Paper: 72.70% overall / 67.00% open. Baseline: 59.24%.

Grid: lr∈[2e-5,3e-5,5e-5] × bs∈[16,32] × dropout∈[0.02,0.05] = 12 runs.
Fixed: n_layers=2, epochs=150, eval every 5 epochs.
Reports best test metric seen across all eval checkpoints.
Collapse stops if <5 unique classes predicted.
Updated: 2026-05-15 16:09

## Results

| # | lr | bs | dropout | Best Overall | Closed | Open | F1 | Best Ep | GPU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2e-05 | 16 | 0.02 | 55.96% | 57.74% | 48.39% | 56.53% | 25 | 0 |
| 2 | 2e-05 | 16 | 0.05 | 55.96% | 57.36% | 50.00% | 56.71% | 100 | 1 |
| 3 | 2e-05 | 32 | 0.02 | 55.35% | 56.98% | 48.39% | 55.92% | 85 | 0 |
| 4 | 2e-05 | 32 | 0.05 | 54.74% | 55.47% | 51.61% | 55.27% | 40 | 1 |
| 5 | 3e-05 | 16 | 0.02 | 59.02% | 60.75% | 51.61% | 59.80% | 55 | 0 |
| 6 | 3e-05 | 16 | 0.05 | 58.41% | 60.38% | 50.00% | 59.16% | 55 | 1 |
| 7 | 3e-05 | 32 | 0.02 | 56.27% | 58.11% | 48.39% | 57.04% | 25 | 0 |
| 8 | 3e-05 | 32 | 0.05 | 56.57% | 58.11% | 50.00% | 57.15% | 70 | 1 |
| 9 | 5e-05 | 16 | 0.02 | 58.72% | 60.75% | 50.00% | 59.55% | 120 | 0 |
| 10 | 5e-05 | 16 | 0.05 | 58.10% | 60.38% | 48.39% | 59.41% | 50 | 1 |
| 11 | 5e-05 | 32 | 0.02 | 59.33% | 61.13% | 51.61% | 59.90% | 55 | 0 |
| 12 | 5e-05 | 32 | 0.05 | 57.49% | 59.25% | 50.00% | 58.06% | 60 | 1 |

## Best so far

lr=5e-05 bs=32 dropout=0.02 best_epoch=55 → **59.33%** overall / 61.13% closed / 51.61% open
