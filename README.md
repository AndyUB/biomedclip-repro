# BiomedCLIP Reproduction

Reproduction of zero-shot and linear-probe results from:
> BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs (arXiv 2303.00915)

## Benchmarks

| Benchmark | Task | Paper result |
|-----------|------|-------------|
| PCam | Zero-shot classification | 73.41% |
| RSNA Pneumonia | Zero-shot classification | 78.95% |
| VQA-RAD | VQA (linear probe) | ~73% overall |

## Setup

```bash
conda create -n biomedclip python=3.10 -y
conda activate biomedclip
pip install -r requirements.txt
```

## Data

See `scripts/download_pcam.md`, `scripts/download_rsna.md`, `scripts/download_vqa_rad.md`.

Expected layout after download:

```
data/
  pcam/
    camelyonpatch_level_2_split_test_x.h5
    camelyonpatch_level_2_split_test_y.h5
  rsna/
    stage_2_train_images/
    stage_2_train_labels.csv
  vqa_rad/
    VQA_RAD Dataset Public.json
    VQA_RAD_testset.json
    images/
```

## Running

All scripts are run from `src/`.

```bash
cd src

# Smoke-test model loading
python load_biomedclip.py

# PCam zero-shot
python eval_pcam.py --data_dir ../data/pcam

# RSNA zero-shot
python eval_rsna.py --data_dir ../data/rsna

# VQA-RAD linear probe
python eval_vqa_rad.py --data_dir ../data/vqa_rad
```

Results are written to `results/`.

## Deviations from paper

- RSNA: evaluated on all labeled images (no official split file available); paper uses 18,678 / 4,003 / 9,069 split.
- VQA-RAD: uses a simple linear classifier on frozen BiomedCLIP features instead of full METER fine-tuning.
