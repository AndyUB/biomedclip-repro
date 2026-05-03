"""Zero-shot evaluation of BiomedCLIP on LC25000 (Lung and Colon subsets).

Paper targets:
  LC25000 Lung:  65.23% accuracy (3-way)
  LC25000 Colon: 92.98% accuracy (2-way)

Supports two data layouts:
  1. Parquet (Hugging Face mirror): data/*.parquet with columns image/organ/label
  2. Folder layout: lung_colon_image_set/{lung,colon}_image_sets/<class>/

Usage:
    cd biomedclip-repro/src
    python eval_lc25000.py --data_dir ../data/lc25000
"""

import argparse
import glob
import io
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from load_biomedclip import load_model
from zeroshot_classification import zero_shot_predict
from metrics import report

LUNG_CLASSES = [
    "normal lung tissue",        # label 0 (lungn)
    "lung adenocarcinomas",      # label 1 (lungaca)
    "lung squamous cell carcinomas",  # label 2 (lungscc)
]
LUNG_TEMPLATES = [
    "a photo of {}",
    "{} presented in image",
]

COLON_CLASSES = [
    "normal colonic tissue",     # label 0 (colonn)
    "colon adenocarcinomas",     # label 1 (colonca)
]
COLON_TEMPLATES = [
    "a photo of {}",
    "{} presented in image",
]

# organ column: 0=lung, 1=colon
ORGAN_ID = {"lung": 0, "colon": 1}

# Folder-layout: maps folder name -> (subset, class_index)
FOLDER_MAP = {
    "lung_aca":  ("lung",  0),
    "lung_n":    ("lung",  1),
    "lung_scc":  ("lung",  2),
    "colon_aca": ("colon", 0),
    "colon_n":   ("colon", 1),
}


class ParquetSubsetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.images = df["image"].tolist()   # list of dicts with 'bytes'
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class FolderSubsetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_parquet(data_dir: Path, subset: str, transform):
    shards = sorted(glob.glob(str(data_dir / "data" / "*.parquet")))
    if not shards:
        return None
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df[df["organ"] == ORGAN_ID[subset]].reset_index(drop=True)
    return ParquetSubsetDataset(df, transform)


def load_folders(data_dir: Path, subset: str, transform):
    if subset == "lung":
        folder_names = ["lung_aca", "lung_n", "lung_scc"]
        roots = [data_dir / "lung_colon_image_set" / "lung_image_sets", data_dir]
    else:
        folder_names = ["colon_aca", "colon_n"]
        roots = [data_dir / "lung_colon_image_set" / "colon_image_sets", data_dir]

    paths, labels = [], []
    for root in roots:
        if not root.exists():
            continue
        for folder in folder_names:
            folder_path = root / folder
            if not folder_path.exists():
                continue
            _, class_idx = FOLDER_MAP[folder]
            imgs = sorted(
                p for p in folder_path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            paths.extend(imgs)
            labels.extend([class_idx] * len(imgs))
        if paths:
            return FolderSubsetDataset(paths, labels, transform)
    return None


def load_split(data_dir: Path, subset: str, transform):
    dataset = load_parquet(data_dir, subset, transform)
    if dataset is not None:
        return dataset
    dataset = load_folders(data_dir, subset, transform)
    if dataset is not None:
        return dataset
    raise FileNotFoundError(
        f"No images found for subset '{subset}' under {data_dir}. "
        "Check that LC25000 is downloaded."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/lc25000")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model, preprocess, tokenizer, device = load_model()

    for subset, classes, templates in [
        ("lung",  LUNG_CLASSES,  LUNG_TEMPLATES),
        ("colon", COLON_CLASSES, COLON_TEMPLATES),
    ]:
        dataset = load_split(data_dir, subset, preprocess)
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        print(f"\nLC25000 {subset}: {len(dataset)} images, {len(classes)} classes")

        preds, labels = zero_shot_predict(
            model, preprocess, tokenizer, loader,
            classes, templates, device
        )
        report(f"lc25000_{subset}", preds, labels, classes, results_dir="../results")


if __name__ == "__main__":
    main()
