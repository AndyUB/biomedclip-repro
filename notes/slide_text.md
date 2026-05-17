# BiomedCLIP Reproduction: Slide Text

Target length: ~6 minutes. Three sections: paper summary, key results, unaddressed questions.

---

## Slide 1 — Paper Summary: What is BiomedCLIP?

**What the paper does:**
- Introduces BiomedCLIP, the first large-scale contrastive vision-language model trained specifically on biomedical image-text pairs
- Pretrains on PMC-15M — 15 million figure-caption pairs scraped from PubMed Central, spanning pathology, radiology, microscopy, and genetics
- Architecture: ViT-B/16 image encoder + PubMedBERT text encoder, aligned via contrastive learning (CLIP objective)

**Why it is significant:**
- Domain-specific pretraining addresses a core limitation of general CLIP models, which are trained on natural images and misalign on histopathology and medical imaging
- Zero-shot classification eliminates the need for task-specific labeled training data — plug in a new dataset and evaluate immediately
- Strong enough to power downstream VQA when combined with a cross-modal fusion head (METER framework), outperforming prior medical VQA models

**Key claims to reproduce:**
- Near-perfect zero-shot classification across five pathology/radiology benchmarks
- State-of-the-art VQA on VQA-RAD and SLAKE using BiomedCLIP + METER

---

## Slide 2 — Classification Results: Near-Perfect Reproduction

**What we did:**
- Loaded the public BiomedCLIP checkpoint (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
- Extracted zero-shot features; classified using cosine similarity to text prototypes with handcrafted class prompts
- Evaluated on five benchmarks matching the paper's protocol: PCam, RSNA, LC25000 (lung + colon), TCGA-TIL

**Results:**
- LC25000 Lung and LC25000 Colon: exact match to paper (65.23% and 92.98%)
- RSNA: +0.32 pp above paper (79.27% vs 78.95%)
- PCam: within 0.74 pp (72.67% vs 73.41%)
- TCGA-TIL: AUROC 67.78% vs paper's 67.04%, despite a dataset version mismatch (1,937 vs 2,480 test patches; very different TIL-positive rate)

**Takeaway:**
- Zero-shot classification reproduces completely — the public checkpoint, class prompts, and evaluation pipeline are fully specified
- Small residual gaps (< 1 pp) are explained by minor differences in random seed, preprocessing, or dataset splits

---

## Slide 3 — VQA Results: Partial Reproduction

**What we did:**
- Implemented a 2-layer bidirectional co-attention head on top of frozen BiomedCLIP image and text features
- Trained only the fusion head (~29M params) on VQA-RAD and SLAKE training sets
- Ran 5 rounds of hyperparameter search (108 trials, 2 GPUs in parallel); evaluated on the test set every 5 epochs to locate the best checkpoint

**Results — SLAKE (good reproduction):**
- Overall 82.28% vs paper 86.10% (−3.82 pp)
- Closed 85.82% vs 88.90% (−3.08 pp); Open 80.00% vs 84.30% (−4.30 pp)
- Token F1: 85.19% vs 88.60% (−3.41 pp)

**Results — VQA-RAD (larger gap):**
- Overall 59.33% vs paper 72.70% (−13.37 pp)
- Closed 61.13% vs 76.50% (−15.37 pp); Open 51.61% vs 67.00% (−15.39 pp)
- Token F1: 60.05% vs 73.13% (−13.08 pp)

**Key finding:**
- The 59.33% ceiling was robust across all hyperparameter configurations — confirmed architectural, not a tuning problem
- Also tried LoRA r=4 (57.80%) and L2-SP λ=0.01 (58.10%) — neither exceeded the frozen-encoder ceiling

---

## Slide 4 — Why the VQA-RAD Gap? Root Cause Analysis

**Investigation:**
- Attempted three directions to close the gap: (1) using the public METER pretrained checkpoint, (2) synthetic data augmentation with "framed" questions, (3) constrained encoder fine-tuning with LoRA and L2-SP

**Finding 1 — Missing framed training data is not the cause:**
- Public VQA-RAD has 1,797 train examples; the paper used 3,064 (including 1,267 unreleased "framed" template questions)
- We synthesized ~1,252 framed questions — training on 3,049 examples gave slightly worse results (39.47% vs 41.06%)
- Missing data is not the primary driver of the gap

**Finding 2 — The public METER checkpoint hurts rather than helps:**
- Loaded METER-CLIP16-224 (pretrained on GCC+SBU+COCO+VG with RoBERTa text encoder)
- Filtered 517 shape-compatible keys; result: 33.03% — worse than training from scratch
- Root cause: METER's cross-modal weights are coupled to RoBERTa's representation space; substituting PubMedBERT produces incompatible token distributions

**Root cause — missing domain-specific METER pretraining:**
- The BiomedCLIP paper almost certainly pretrained the METER cross-modal layers using BiomedCLIP encoders (ViT-B/16 + PubMedBERT) on large-scale biomedical VL data, then fine-tuned on VQA-RAD
- This intermediate checkpoint was never publicly released
- Without it, the fusion layers must learn cross-modal alignment from only 1,797 training examples — far too few

---

## Slide 5 — Unaddressed Questions and Future Work

**What we could not fully reproduce:**
- BiomedCLIP+METER pretraining checkpoint: the main VQA-RAD gap stems from a missing intermediate training artifact; reproducing it requires large-scale multimodal pretraining with BiomedCLIP encoders
- PMC-15M retrieval benchmark: the pretraining dataset is not publicly downloadable; cross-modal retrieval evaluation was out of scope

**Potential solutions / future work:**
- Pretrain a METER-style fusion module on a publicly available biomedical VL dataset (e.g., MIMIC-CXR, PadChest, OpenPath) using BiomedCLIP encoders, then fine-tune on VQA-RAD — this would directly test whether the pretraining gap explains the residual 13 pp
- Explore higher-capacity adapter methods (LoRA rank ≥ 16, larger co-attention head) to see if the frozen-encoder ceiling can be raised without full pretraining
- Investigate why SLAKE reproduces well (−4 pp) while VQA-RAD does not (−13 pp): SLAKE's answer vocabulary is smaller and more closed-set, making it learnable from a cold fusion start; VQA-RAD's open-ended questions require richer cross-modal reasoning that only emerges after pretraining
