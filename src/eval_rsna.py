"""Zero-shot evaluation of BiomedCLIP on RSNA Pneumonia.

Paper target: ~78.95% accuracy.

Usage:
    cd biomedclip-repro/src
    python eval_rsna.py --data_dir ../data/rsna
"""

import argparse
import os
import pandas as pd
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from load_biomedclip import load_model
from zeroshot_classification import zero_shot_predict
from metrics import report

RSNA_CLASSES = ["normal lung", "pneumonia"]
RSNA_TEMPLATES = ["a photo of {}", "{} presented in image"]


def build_image_labels(labels_csv):
    """Collapse box-level annotations to image-level binary labels."""
    df = pd.read_csv(labels_csv)
    # Each patientId may appear multiple times (multiple bounding boxes).
    # A patient is positive (pneumonia=1) if ANY box has Target=1.
    img_labels = df.groupby("patientId")["Target"].max().reset_index()
    return img_labels


class RSNADataset(Dataset):
    def __init__(self, img_labels, images_dir, transform=None):
        self.img_labels = img_labels.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        patient_id = row["patientId"]
        label = int(row["Target"])

        dcm_path = os.path.join(self.images_dir, f"{patient_id}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.uint8)
        # DICOM is grayscale; convert to RGB
        img = Image.fromarray(arr).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/rsna")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_model()

    labels_csv = os.path.join(args.data_dir, "stage_2_train_labels.csv")
    images_dir = os.path.join(args.data_dir, "stage_2_train_images")

    img_labels = build_image_labels(labels_csv)
    print(f"RSNA: {len(img_labels)} unique patients "
          f"({img_labels['Target'].sum()} pneumonia, "
          f"{(img_labels['Target']==0).sum()} normal)")

    dataset = RSNADataset(img_labels, images_dir, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    preds, labels = zero_shot_predict(
        model, preprocess, tokenizer, loader,
        RSNA_CLASSES, RSNA_TEMPLATES, device
    )
    report("rsna", preds, labels, RSNA_CLASSES, results_dir="../results")


if __name__ == "__main__":
    main()
