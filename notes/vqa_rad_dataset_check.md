# VQA-RAD Dataset Consistency Check

## Dataset source

We use the standard public `VQA_RAD Dataset Public.json` (2,248 QA pairs total).

## Split counts

| Split | Freeform | Paraphrase | Total |
|-------|----------|------------|-------|
| Train | 1,206    | 591        | 1,797 |
| Test  | 308      | 143        | 451   |

Answer vocabulary built from training data: **433 unique answers**.

## Missing images

The JSON references 203 unique test images, but **8 are absent** from `data/vqa_rad/images/`:

```
synpic676.jpg    synpic33302.jpg  synpic39086.jpg  synpic41788.jpg
synpic47974.jpg  synpic53867.jpg  synpic55286.jpg  synpic55948.jpg
```

These 8 files account for **15 test QA pairs** that are silently skipped during
Arrow conversion. Our evaluation runs on **436 QA pairs** rather than the full 451.

The paper almost certainly evaluated on all 451. The missing images can be
retrieved from the official VQA-RAD release at osf.io/89kps.

## Impact on results

The 15 dropped QA pairs are **not** the cause of the large VQA-RAD gap:
- Best-case correction (all 15 answered correctly): ~+3 pp
- Actual gap vs. paper: ~32 pp overall, ~48 pp on open questions

The gap is due to the unavailable METER pretrained cross-modal checkpoint
(see `results/SUMMARY.md`).
