"""Zero-shot evaluation of BiomedCLIP on PCam test set.

Paper target: ~73.41% accuracy.

Usage:
    cd biomedclip-repro/src
    python eval_pcam.py --data_dir ../data/pcam
"""

import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from load_biomedclip import load_model
from zeroshot_classification import zero_shot_predict
from metrics import report

PCAM_CLASSES = ["normal lymph node", "lymph node metastasis"]
PCAM_TEMPLATES = ["this is an image of {}", "{} presented in image"]


class PCamDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.h5x = h5py.File(x_path, "r")
        self.h5y = h5py.File(y_path, "r")
        self.x = self.h5x["x"]  # (N, 96, 96, 3) uint8
        self.y = self.h5y["y"]  # (N, 1, 1, 1) uint8
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]           # (96, 96, 3)
        label = int(self.y[idx][0][0][0])
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label

    def close(self):
        self.h5x.close()
        self.h5y.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/pcam")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_model()

    x_path = f"{args.data_dir}/camelyonpatch_level_2_split_test_x.h5"
    y_path = f"{args.data_dir}/camelyonpatch_level_2_split_test_y.h5"
    dataset = PCamDataset(x_path, y_path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    print(f"PCam test set: {len(dataset)} images")
    preds, labels = zero_shot_predict(
        model, preprocess, tokenizer, loader,
        PCAM_CLASSES, PCAM_TEMPLATES, device
    )
    report("pcam", preds, labels, PCAM_CLASSES, results_dir="../results")
    dataset.close()


if __name__ == "__main__":
    main()
