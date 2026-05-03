# Midterm Progress Presentation — BiomedCLIP Reproduction

---

## Brief Summary

**Paper:** Xu et al., "A Large-Scale Biomedical Vision-Language Model" (BiomedCLIP), arXiv 2303.00915, Microsoft Research.

BiomedCLIP is a CLIP-style vision-language model pretrained on 15 million biomedical image-text pairs from PubMed (PMC-15M). It encodes images with a ViT and text with PubMedBERT. The paper evaluates zero-shot classification and VQA across multiple biomedical imaging benchmarks.

---

## Current Progress

### Reproduction Plan

**Reproduce (classification, zero-shot):**
- PCam — lymph node metastasis detection
- RSNA — pneumonia detection
- LC25000 — lung and colon histopathology (2 sub-tasks)
- TCGA-TIL — TIL detection in LUAD patches (AUROC)

**Reproduce (VQA, fine-tuned with METER):**
- VQA-RAD — radiology VQA
- SLAKE — general medical VQA

**Skip:**
- PMC-15M retrieval — dataset not publicly available

### Results

**Classification (zero-shot):**

| Benchmark     | Ours         | Paper        | Gap      |
|---------------|--------------|--------------|----------|
| PCam          | 72.67%       | 73.41%       | −0.74 pp |
| RSNA          | 79.27%       | 78.95%       | +0.32 pp |
| LC25000 Lung  | **65.23%**   | 65.23%       | 0.00 pp  |
| LC25000 Colon | **92.98%**   | 92.98%       | 0.00 pp  |
| TCGA-TIL      | 67.78% AUROC | 67.04% AUROC | +0.74 pp |

**VQA (fine-tuned METER + BiomedCLIP):**

| Benchmark | Overall    | Paper  | Gap      |
|-----------|------------|--------|----------|
| VQA-RAD   | 41.06%     | 72.70% | −31.6 pp |
| SLAKE     | 82.28%     | 86.10% | −3.82 pp |

Classification: **5/5 benchmarks reproduced**, all within ~1 pp of paper.
VQA: **2/2 benchmarks attempted**; SLAKE close, VQA-RAD has a large gap (see Bottleneck).

---

## Interesting Findings

**1. Prompt wording is highly sensitive for zero-shot classification.**
On LC25000 Lung, switching the template from `"this is an image of {}"` to `"a photo of {}"` changed accuracy from **58.74% to 65.23%** — a 6.5 pp swing that was the difference between a poor and exact match. The class names also had to be in the right order to match the dataset's integer labels; having them swapped produced ~7% accuracy (worse than random) on the binary colon task.

**2. AUROC is robust to dataset distribution shift.**
The TCGA-TIL dataset version we found has a very different positive rate (27.7%) vs. the paper (5.9%), and fewer images (1,937 vs. 2,480). Despite this, our AUROC matched the paper within 0.74 pp. AUROC is threshold-free, so it is insensitive to base rate changes — a useful property for imbalanced benchmarks.

**3. VQA-RAD gap is due to missing pretrained cross-modal weights.**
The paper's METER results relied on a pretrained cross-modal checkpoint that is not publicly released. Training METER from scratch with BiomedCLIP as the vision encoder causes a ~32 pp drop on VQA-RAD, concentrated in open-ended questions (15.6% vs. 63.3%). SLAKE — with a smaller vocabulary (221 vs. 433 answers) — shows only a ~4 pp gap, suggesting the cross-modal pretraining matters most for harder open-ended reasoning.

---

## Bottleneck

**VQA-RAD cross-modal pretraining checkpoint.**
The paper fine-tunes METER starting from a checkpoint pretrained on general VQA data (VQAv2). That checkpoint is not released. We fine-tune from scratch with BiomedCLIP vision features only, which results in a large gap especially on open questions. Short of replicating the full METER pretraining (which requires the VQAv2 dataset and significant compute), this gap cannot be fully closed. We report the result transparently with this caveat.
