"""METER-style VQA evaluation of BiomedCLIP on VQA-RAD.

Architecture (faithful to BiomedCLIP paper):
  image  → BiomedCLIP ViT-B/16 trunk → patch tokens  (B, 197, 768)  [frozen]
  question → BiomedCLIP PubMedBERT   → text tokens   (B, 256, 768)  [frozen]
  patch tokens + text tokens → N-layer co-attention fusion            [trained]
  fused [CLS] representation → MLP classifier → answer class         [trained]

Paper targets (Figure 4):
  Open-ended accuracy:   67.00%
  Closed-ended accuracy: 76.50%
  Overall accuracy:      72.70%
  Token-level F1:        73.13%

Usage:
    cd biomedclip-repro/src
    python eval_vqa_rad.py --data_dir ../data/vqa_rad
"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from load_biomedclip import load_model, CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_split(json_path, images_dir):
    with open(json_path) as f:
        data = json.load(f)
    data = [ex for ex in data
            if os.path.exists(os.path.join(images_dir, ex["image_name"]))]
    train = [ex for ex in data if not ex["phrase_type"].startswith("test_")]
    test  = [ex for ex in data if ex["phrase_type"].startswith("test_")]
    return train, test


def build_answer_vocab(train_data, min_freq=2):
    counts = Counter(str(ex["answer"]).lower().strip() for ex in train_data)
    answers = sorted(a for a, c in counts.items() if c >= min_freq)
    ans2idx = {a: i for i, a in enumerate(answers)}
    idx2ans = {i: a for a, i in ans2idx.items()}
    return ans2idx, idx2ans


class VQARadDataset(Dataset):
    def __init__(self, data, images_dir, ans2idx, transform=None):
        self.data = data
        self.images_dir = images_dir
        self.ans2idx = ans2idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = Image.open(os.path.join(self.images_dir, ex["image_name"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        question = ex["question"]
        answer = str(ex["answer"]).lower().strip()
        label = self.ans2idx.get(answer, -1)
        is_closed = ex.get("answer_type", "OPEN").strip().upper() == "CLOSED"
        return img, question, label, is_closed, answer


# ---------------------------------------------------------------------------
# METER co-attention head
# ---------------------------------------------------------------------------

class CoAttentionLayer(nn.Module):
    """One round of bidirectional cross-modal attention (image ↔ text)."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        ff_dim = dim * 4
        self.img_cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.txt_cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.img_ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim)
        )
        self.txt_ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim)
        )
        self.img_norm1 = nn.LayerNorm(dim)
        self.img_norm2 = nn.LayerNorm(dim)
        self.txt_norm1 = nn.LayerNorm(dim)
        self.txt_norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, img, txt, txt_pad_mask=None):
        # img: (B, N_img, dim)   txt: (B, N_txt, dim)
        # txt_pad_mask: (B, N_txt) — True for padding positions

        # image queries, text keys/values
        img2, _ = self.img_cross(img, txt, txt, key_padding_mask=txt_pad_mask)
        img = self.img_norm1(img + self.drop(img2))
        img = self.img_norm2(img + self.drop(self.img_ffn(img)))

        # text queries, image keys/values
        txt2, _ = self.txt_cross(txt, img, img)
        txt = self.txt_norm1(txt + self.drop(txt2))
        txt = self.txt_norm2(txt + self.drop(self.txt_ffn(txt)))

        return img, txt


class METERHead(nn.Module):
    def __init__(self, dim=768, n_heads=8, n_layers=2, num_classes=128, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CoAttentionLayer(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img_tokens, txt_tokens, txt_pad_mask=None):
        img, txt = img_tokens, txt_tokens
        for layer in self.layers:
            img, txt = layer(img, txt, txt_pad_mask)
        # CLS token from each modality (index 0)
        fused = torch.cat([img[:, 0], txt[:, 0]], dim=-1)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Feature extraction (frozen encoders)
# ---------------------------------------------------------------------------

def get_image_tokens(clip_model, images):
    """(B, 197, 768) patch tokens from frozen ViT trunk."""
    with torch.no_grad():
        return clip_model.visual.trunk.forward_features(images)


def get_text_tokens(clip_model, tokens):
    """(B, L, 768) token features + (B, L) bool padding mask from frozen BERT."""
    with torch.no_grad():
        attn_mask = (tokens != 0).long()
        out = clip_model.text.transformer(input_ids=tokens, attention_mask=attn_mask)
    pad_mask = (tokens == 0)  # True = ignore (PyTorch convention)
    return out.last_hidden_state, pad_mask


# ---------------------------------------------------------------------------
# Token-level F1 (GPT-2 tokenization, as in the paper)
# ---------------------------------------------------------------------------

_gpt2_tok = None

def _get_gpt2():
    global _gpt2_tok
    if _gpt2_tok is None:
        from transformers import GPT2Tokenizer
        _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    return _gpt2_tok


def token_f1(pred_str, gold_str):
    tok = _get_gpt2()
    pred_tokens = tok.tokenize(pred_str.lower())
    gold_tokens = tok.tokenize(gold_str.lower())
    if not pred_tokens or not gold_tokens:
        return float(pred_str.lower() == gold_str.lower())
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(pred_tokens)
    rec = n_common / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def run_epoch(clip_model, meter_head, tokenizer, loader, optimizer, device, train=True):
    meter_head.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for imgs, questions, labels, _, _ in tqdm(loader, desc="train" if train else "eval", leave=False):
        imgs = imgs.to(device)
        tokens = tokenizer(list(questions), context_length=CONTEXT_LENGTH).to(device)
        labels = labels.to(device)

        img_tokens = get_image_tokens(clip_model, imgs)
        txt_tokens, txt_pad_mask = get_text_tokens(clip_model, tokens)

        logits = meter_head(img_tokens, txt_tokens, txt_pad_mask)

        valid = labels >= 0
        if not valid.any():
            continue

        loss = F.cross_entropy(logits[valid], labels[valid])

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * valid.sum().item()
        correct += (logits[valid].argmax(-1) == labels[valid]).sum().item()
        total += valid.sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(clip_model, meter_head, tokenizer, loader, idx2ans, device):
    meter_head.eval()
    all_preds, all_labels, all_gold_strs, all_closed = [], [], [], []

    with torch.no_grad():
        for imgs, questions, labels, is_closed, gold_strs in tqdm(loader, desc="evaluating", leave=False):
            imgs = imgs.to(device)
            tokens = tokenizer(list(questions), context_length=CONTEXT_LENGTH).to(device)

            img_tokens = get_image_tokens(clip_model, imgs)
            txt_tokens, txt_pad_mask = get_text_tokens(clip_model, tokens)
            logits = meter_head(img_tokens, txt_tokens, txt_pad_mask)
            preds = logits.argmax(-1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)
            all_gold_strs.extend(gold_strs)
            all_closed.append(is_closed)

    preds   = torch.cat(all_preds).numpy()
    labels  = torch.cat(all_labels).numpy()
    closed  = torch.cat(all_closed).numpy()

    valid = labels >= 0
    preds_v, labels_v, closed_v = preds[valid], labels[valid], closed[valid]
    gold_v = [all_gold_strs[i] for i in range(len(all_gold_strs)) if valid[i]]
    pred_strs = [idx2ans.get(int(p), "") for p in preds_v]

    overall  = (preds_v == labels_v).mean()
    closed_acc = (preds_v[closed_v] == labels_v[closed_v]).mean() if closed_v.any() else float("nan")
    open_acc   = (preds_v[~closed_v] == labels_v[~closed_v]).mean() if (~closed_v).any() else float("nan")
    avg_f1     = sum(token_f1(p, g) for p, g in zip(pred_strs, gold_v)) / len(gold_v)

    print(f"\n=== VQA-RAD ===")
    print(f"Overall accuracy:        {overall*100:.2f}%  (paper: 72.70%)")
    print(f"Closed-ended accuracy:   {closed_acc*100:.2f}%  (paper: 76.50%)")
    print(f"Open-ended accuracy:     {open_acc*100:.2f}%  (paper: 67.00%)")
    print(f"Token-level F1:          {avg_f1*100:.2f}%  (paper: 73.13%)")
    return overall, closed_acc, open_acc, avg_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="../data/vqa_rad")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--n_layers",   type=int, default=2)
    parser.add_argument("--n_heads",    type=int, default=8)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--min_freq",   type=int, default=2)
    args = parser.parse_args()

    clip_model, preprocess, tokenizer, device = load_model()
    # Keep BiomedCLIP encoders frozen; only METER head is trained
    for p in clip_model.parameters():
        p.requires_grad_(False)

    json_path   = os.path.join(args.data_dir, "VQA_RAD Dataset Public.json")
    images_dir  = os.path.join(args.data_dir, "images")

    train_data, test_data = load_and_split(json_path, images_dir)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")

    ans2idx, idx2ans = build_answer_vocab(train_data, min_freq=args.min_freq)
    num_classes = len(ans2idx)
    print(f"Answer vocabulary: {num_classes} classes (answers with ≥{args.min_freq} train examples)")

    train_ds = VQARadDataset(train_data, images_dir, ans2idx, transform=preprocess)
    test_ds  = VQARadDataset(test_data,  images_dir, ans2idx, transform=preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    meter_head = METERHead(
        dim=768, n_heads=args.n_heads, n_layers=args.n_layers,
        num_classes=num_classes, dropout=args.dropout,
    ).to(device)
    print(f"METER head parameters: {sum(p.numel() for p in meter_head.parameters()):,}")

    optimizer = optim.AdamW(meter_head.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining METER head for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = run_epoch(clip_model, meter_head, tokenizer, train_loader, optimizer, device, train=True)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  epoch {epoch}/{args.epochs}  loss={loss:.4f}  train_acc={acc:.4f}")

    evaluate(clip_model, meter_head, tokenizer, test_loader, idx2ans, device)


if __name__ == "__main__":
    main()
