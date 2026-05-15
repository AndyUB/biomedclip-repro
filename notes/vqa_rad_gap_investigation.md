# VQA-RAD Reproduction Gap Investigation

Paper target: Overall 72.70% / Closed 76.50% / Open 67.00% / F1 73.13%

## Experiment Summary

| Approach | Overall | Closed | Open | F1 | Notes |
|---|---|---|---|---|---|
| From scratch (20 ep, lr=5e-6) | 41.06% | 57.79% | 15.61% | 43.25% | Best from-scratch result |
| Pretrained METER ckpt (20 ep) | 33.03% | 50.57% | 6.36% | 33.18% | Worse than scratch |
| Synthetic framed augmentation | 39.47% | 58.46% | 10.61% | 41.56% | 3049 train, slightly worse |
| Grid search (best, 10 ep) | 33.03% | 50.57% | 6.36% | 33.18% | lr=5e-6, head=100 |
| Paper (METER + BiomedCLIP) | **72.70%** | **76.50%** | **67.00%** | **73.13%** | |

Gap to paper: −31.64 pp overall, −51.39 pp open-ended.

---

## Finding 1: Missing training data (framed questions) is not the cause

The public VQA-RAD JSON has 1,797 train examples. The paper used 3,064 (including 1,267
"framed" template-style questions never publicly released).

We generated 1,252 synthetic framed questions by appending "in this image?" to existing
questions. Training on the augmented set (3,049 examples) gave slightly worse results
(39.47% vs 41.06%). Investigation showed 99.4% of rewrites were trivially appending
"in this image?" — not the kind of meaningful template restructuring in the original
paper. Even with truly reconstructed framed questions, the gain is unlikely to explain
a 31 pp gap.

**Conclusion:** Missing framed training data is not the primary cause of the gap.

---

## Finding 2: Pretrained METER cross-modal weights don't transfer

We downloaded the public METER-CLIP16-224 pretrained checkpoint (pretrained on
GCC+SBU+COCO+VG with CLIP ViT-B/16 + RoBERTa). We loaded 517 shape-compatible keys
(including the 320 cross-modal attention keys) via filtered `strict=False` loading,
skipping RoBERTa embedding layers that have different vocab sizes.

Result: **worse** than from-scratch (33.03% vs 41.06%). Convergence was also slower
(30.3% at epoch 18 vs 36.7% at epoch 9 from scratch).

**Why it hurts:** The pretrained cross-modal layers learned to process RoBERTa's
representation space. Swapping in PubMedBERT produces incompatible text token
distributions, so the pretrained cross-modal weights fight against PubMedBERT's
outputs rather than helping. The pretrained weights are not encoder-agnostic.

**Conclusion:** To benefit from METER pretraining, we would need a checkpoint pretrained
jointly with BiomedCLIP encoders (ViT-B/16 + PubMedBERT). The BiomedCLIP authors
likely performed this pretraining internally. Without it, the cross-modal layers must
be learned from only ~1,797 VQA-RAD examples, which is far too little for 314M
trainable parameters.

---

## Finding 3: Overfitting, not underfitting

Training accuracy reaches ~83% by epoch 20 while test accuracy plateaus at ~41%.
The model memorizes training answers but fails to generalize. Root cause: 314M
trainable parameters fine-tuned on ~1,797 examples from random cross-modal initialization.

Grid search over learning rates confirmed: rates ≥ 1e-5 collapse to 0% open accuracy
(predicting only yes/no). The original lr=5e-6 was already near-optimal for from-scratch
training.

---

## Root Cause Assessment

The reproduction gap is most likely due to **missing domain-specific METER pretraining**.
The BiomedCLIP paper almost certainly:
1. Pretrained METER cross-modal layers on large-scale vision-language data using
   BiomedCLIP (ViT-B/16) + PubMedBERT encoders.
2. Used that pretrained checkpoint to initialize VQA-RAD fine-tuning.

This checkpoint was never publicly released. Without it, the cross-modal fusion layers
start from random initialization and cannot learn meaningful image-text alignment from
only 1,797 training examples.

The gap in open-ended accuracy (15.61% vs 67.00%) is particularly telling: open-ended
VQA requires rich cross-modal reasoning (anatomy, location, modality recognition) that
can only be learned with a properly pretrained fusion module.
