# BiomedCLIP Reproduction Summary

Reproduction of zero-shot classification and VQA benchmarks from:
> Xu et al., "A Large-Scale Biomedical Vision-Language Model" (BiomedCLIP), arXiv 2303.00915.

All classification tasks use BiomedCLIP zero-shot (no fine-tuning).
VQA tasks use METER fine-tuned with BiomedCLIP as the vision encoder.

---

## Classification Results

| Benchmark     | Ours   | Paper  | Gap      | Notes |
|---------------|--------|--------|----------|-------|
| PCam          | 72.67% | 73.41% | −0.74 pp | |
| RSNA          | 79.27% | 78.95% | +0.32 pp | |
| LC25000 Lung  | 65.23% | 65.23% |  0.00 pp | Exact match |
| LC25000 Colon | 92.98% | 92.98% |  0.00 pp | Exact match |
| TCGA-TIL      | 67.78% AUROC | 67.04% AUROC | +0.74 pp | Dataset version mismatch (see below) |

## VQA Results

| Benchmark | Metric   | Ours   | Paper  | Gap      |
|-----------|----------|--------|--------|----------|
| VQA-RAD   | Overall  | 41.06% | 72.70% | −31.6 pp | Pretraining mismatch (see below) |
| VQA-RAD   | Closed   | 57.79% | 79.07% | −21.3 pp | |
| VQA-RAD   | Open     | 15.61% | 63.32% | −47.7 pp | |
| VQA-RAD   | Token F1 | 43.25% | 73.13% | −29.9 pp | |
| SLAKE     | Overall  | 82.28% | 86.10% | −3.82 pp | |
| SLAKE     | Closed   | 85.82% | 88.90% | −3.08 pp | |
| SLAKE     | Open     | 80.00% | 84.30% | −4.30 pp | |
| SLAKE     | Token F1 | 85.19% | 88.60% | −3.41 pp | |

---

## Notes

### TCGA-TIL Dataset Mismatch
The paper reports results on 2,480 LUAD patches with a 5.9% TIL-positive rate.
The Zenodo 6604094 LUAD test split used here contains 1,937 patches with a 27.7%
positive rate — a significant distribution shift. Despite this, the AUROC (a
threshold-free metric) landed within 0.74 pp of the paper target. Exact
reproduction would require the paper's specific patch selection.

### VQA-RAD Mismatch
The paper's METER results used a pretrained cross-modal checkpoint unavailable
publicly. Our run fine-tunes METER from scratch with BiomedCLIP as the vision
backbone, which explains the large gap especially on open-ended questions.
SLAKE, also fine-tuned from scratch, shows a much smaller gap (~3–4 pp), likely
because its answer vocabulary (221 answers) is smaller and easier to learn.

### Retrieval (PMC-15M)
Not reproduced — dataset is not publicly available.
