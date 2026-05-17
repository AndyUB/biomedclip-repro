# BiomedCLIP Reproduction: Results Comparison Table

| Category | Dataset | Metric | Paper | Ours | Gap |
|---|---|---|---|---|---|
| Classification | PCam | Accuracy | 73.41% | 72.67% | −0.74 pp |
| Classification | RSNA | Accuracy | 78.95% | 79.27% | +0.32 pp |
| Classification | LC25000 Lung | Accuracy | 65.23% | 65.23% | 0.00 pp |
| Classification | LC25000 Colon | Accuracy | 92.98% | 92.98% | 0.00 pp |
| Classification | TCGA-TIL | AUROC | 67.04% | 67.78% | +0.74 pp |
| VQA | VQA-RAD Open | Accuracy | 67.00% | 51.61% | −15.39 pp |
| VQA | VQA-RAD Closed | Accuracy | 76.50% | 61.13% | −15.37 pp |
| VQA | VQA-RAD Overall | Accuracy | 72.70% | 59.33% | −13.37 pp |
| VQA | VQA-RAD | Token F1 | 73.13% | 60.05% | −13.08 pp |
| VQA | SLAKE Open | Accuracy | 84.30% | 80.00% | −4.30 pp |
| VQA | SLAKE Closed | Accuracy | 88.90% | 85.82% | −3.08 pp |
| VQA | SLAKE Overall | Accuracy | 86.10% | 82.28% | −3.82 pp |
| VQA | SLAKE | Token F1 | 88.60% | 85.19% | −3.41 pp |

**Notes:**
- VQA-RAD: best result uses frozen BiomedCLIP encoders + 2-layer co-attention head (59.33%), found via 5-round hyperparameter grid search (108 trials). The from-scratch METER baseline achieved only 41.06%.
- TCGA-TIL: dataset version mismatch — our test split has 1,937 patches (27.7% TIL-positive) vs. the paper's 2,480 patches (5.9% TIL-positive). AUROC is comparable despite the distribution shift.
