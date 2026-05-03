# VQA-RAD Custom Model Results

## Architecture

Frozen BiomedCLIP encoders + trained 2-layer bidirectional co-attention head:

- Image: BiomedCLIP ViT-B/16 → patch tokens (B, 197, 768) [frozen]
- Text: BiomedCLIP PubMedBERT → token features (B, L, 768) [frozen]
- Co-attention: 2-layer bidirectional cross-modal attention [trained]
- Classifier: MLP on concatenated CLS tokens → answer class [trained]

Answer vocabulary: answers appearing ≥2 times in training set (216 classes).

## Best result (vqa_rad2.log)

50 epochs, no validation split, all training data used.

| Metric   | Ours   | Paper  | Gap      |
|----------|--------|--------|----------|
| Overall  | 59.24% | 72.70% | −13.5 pp |
| Closed   | 61.96% | 79.07% | −17.1 pp |
| Open     | 47.46% | 67.00% | −19.5 pp |
| Token F1 | 60.29% | 73.13% | −12.8 pp |

## Comparison across approaches

| Run | Overall | Closed | Open | F1 |
|-----|---------|--------|------|----|
| Linear probe (vqa_rad.log) | 42.11% | 48.05% | 19.40% | — |
| Co-attention, no val split, 50 ep (vqa_rad2.log) | **59.24%** | 61.96% | 47.46% | 60.29% |
| Co-attention, val + early stopping (vqa_rad3.log) | 51.29% | 52.55% | 45.45% | 52.68% |
| Full METER fine-tuned from scratch | 41.06% | 57.79% | 15.61% | 43.25% |
| **Paper (METER + pretrained checkpoint)** | **72.70%** | **79.07%** | **63.32%** | **73.13%** |

## Key finding

The custom co-attention head (59.24%) outperformed the full METER fine-tuned from
scratch (41.06%). METER has far more parameters and relies on cross-modal
pretraining to converge well — without the pretrained checkpoint, the lightweight
co-attention head on frozen BiomedCLIP features is more sample-efficient.
