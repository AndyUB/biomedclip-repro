"""Zero-shot evaluation of BiomedCLIP on TCGA-TIL (LUAD, test split).

Paper target: 67.04% AUROC.

Dataset layout after extracting TCGA-TILs.tar.gz (Zenodo 6604094):
    TCGA-TILs/images-tcga-tils/luad/test/til-positive/*.png
    TCGA-TILs/images-tcga-tils/luad/test/til-negative/*.png

WARNING: Dataset version mismatch is possible here. If image count or positive
rate differs from the paper's 2,480 / 5.9%, this is noted in the saved results.

Usage:
    cd biomedclip-repro/src
    python eval_tcga_til.py --data_dir ../data/tcga_til
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

from load_biomedclip import load_model
from zeroshot_classification import build_text_features

CLASSES = ["none", "tumor infiltrating lymphocytes"]
TEMPLATES = ["a photo of {}", "{} presented in image"]
POS_IDX = 1  # index of the TIL-positive class


class TcgaTilDataset(Dataset):
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


def load_test_split(data_dir: Path):
    """
    Walk TCGA-TILs folder structure to collect LUAD test images.
    Labels come from folder names (til-positive=1, til-negative=0).
    Falls back to any cancer type if luad is not found.
    """
    # Canonical path: TCGA-TILs/images-tcga-tils/luad/test/
    candidates = [
        data_dir / "TCGA-TILs" / "images-tcga-tils" / "luad" / "test",
        data_dir / "images-tcga-tils" / "luad" / "test",
        data_dir / "luad" / "test",
    ]
    luad_test = next((p for p in candidates if p.is_dir()), None)

    if luad_test is None:
        # Try finding any 'test' directory containing til-positive/til-negative
        for p in sorted(data_dir.rglob("test")):
            if (p / "til-positive").is_dir() or (p / "til-negative").is_dir():
                luad_test = p
                print(f"Warning: luad/test not found, falling back to {luad_test}")
                break

    if luad_test is None:
        raise FileNotFoundError(
            f"No LUAD test directory found under {data_dir}. "
            "Check that TCGA-TILs.tar.gz has been extracted."
        )

    print(f"Loading from: {luad_test}")
    paths, labels = [], []
    for class_dir, label in [("til-positive", 1), ("til-negative", 0)]:
        folder = luad_test / class_dir
        if not folder.is_dir():
            print(f"  Warning: {folder} not found")
            continue
        imgs = sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        paths.extend(imgs)
        labels.extend([label] * len(imgs))
        print(f"  {class_dir}: {len(imgs)} images")

    return paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/tcga_til")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths, labels = load_test_split(data_dir)

    n_total = len(paths)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    pos_rate = n_pos / n_total if n_total else 0
    print(f"\nTCGA-TIL LUAD test: {n_total} images, "
          f"{n_pos} positive ({pos_rate*100:.1f}%), {n_neg} negative")

    version_warnings = []
    if abs(n_total - 2480) > 50:
        msg = (f"Found {n_total} test images; paper reports 2,480. "
               "Results may not be directly comparable.")
        print(f"WARNING: {msg}")
        version_warnings.append(msg)
    if abs(pos_rate - 0.059) > 0.02:
        msg = (f"Positive rate {pos_rate*100:.1f}%; paper reports 5.9%.")
        print(f"WARNING: {msg}")
        version_warnings.append(msg)

    model, preprocess, tokenizer, device = load_model()
    dataset = TcgaTilDataset(paths, labels, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    text_features = build_text_features(model, tokenizer, CLASSES, TEMPLATES, device)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            image_features = F.normalize(model.encode_image(images), dim=-1)
            logits = image_features @ text_features.T   # (batch, 2)
            probs = torch.softmax(logits, dim=-1)[:, POS_IDX]
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(lbls.tolist())

    auroc = roc_auc_score(all_labels, all_scores)
    preds = [1 if s > 0.5 else 0 for s in all_scores]
    acc = accuracy_score(all_labels, preds)

    print(f"\n=== TCGA-TIL ===")
    print(f"AUROC:    {auroc*100:.2f}%   (paper target: 67.04%)")
    print(f"Accuracy: {acc*100:.2f}%  (sanity check only)")

    result = {
        "dataset": "tcga_til",
        "auroc": auroc,
        "accuracy": acc,
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": pos_rate,
        "paper_target_auroc": 0.6704,
        "paper_n_total": 2480,
        "paper_positive_rate": 0.059,
    }
    if version_warnings:
        result["version_warnings"] = version_warnings

    out_dir = Path("../results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "tcga_til_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
