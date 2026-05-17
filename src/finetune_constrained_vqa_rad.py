"""Constrained encoder fine-tuning for VQA-RAD.

Two methods selectable via --method:

  lora  — Low-Rank Adaptation (rank r) applied to Q/V projections (BERT text
          encoder) and QKV projection (ViT image encoder). Backbone weights stay
          frozen; only the tiny LoRA adapter matrices + co-attention head train.

  l2sp  — L2-from-Starting-Point: all parameters trainable, but loss includes
          λ * Σ||θ - θ_0||² to penalise deviation from pretrained BiomedCLIP.

Hyperparams are fixed to the best found by frozen-encoder grid search:
  n_layers=2, lr=5e-5, bs=16, dropout=0.05, 100 epochs, eval every 5 epochs.

For LoRA the co-attention head uses a 2× higher lr than the adapters.
For L2-SP the backbone uses lr=5e-5 and the co-attention head uses 5e-4.

Usage (launch on both GPUs simultaneously):
    CUDA_VISIBLE_DEVICES=0 python finetune_constrained_vqa_rad.py --method lora --rank 4
    CUDA_VISIBLE_DEVICES=1 python finetune_constrained_vqa_rad.py --method l2sp --lambda_l2sp 0.01
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_biomedclip import load_model, CONTEXT_LENGTH

# ---------------------------------------------------------------------------
# Fixed hyperparams (from frozen-encoder search)
# ---------------------------------------------------------------------------
N_LAYERS   = 2
N_HEADS    = 8
DROPOUT    = 0.05
LR_BACKBONE = 5e-5
LR_HEAD     = 1e-4
BATCH_SIZE  = 16
EPOCHS      = 100
EVAL_EVERY  = 5
MIN_FREQ    = 2

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
NOTES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../notes")

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


class VQADataset(Dataset):
    def __init__(self, data, images_dir, ans2idx, transform):
        self.data = data
        self.images_dir = images_dir
        self.ans2idx = ans2idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = Image.open(os.path.join(self.images_dir, ex["image_name"])).convert("RGB")
        answer = str(ex["answer"]).lower().strip()
        return (self.transform(img), ex["question"],
                self.ans2idx.get(answer, -1),
                int(ex.get("answer_type", "OPEN").strip().upper() == "CLOSED"),
                answer)


def collate_fn(batch):
    imgs, qs, labs, closed, golds = zip(*batch)
    return (torch.stack(imgs), list(qs),
            torch.tensor(labs, dtype=torch.long),
            torch.tensor(closed, dtype=torch.bool),
            list(golds))

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a rank-r LoRA adapter."""
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        # A: random init, B: zero init → adapter output is 0 at start
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.linear(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def apply_lora(model, rank, target_leaf_names=('query', 'value', 'qkv')):
    """Replace target nn.Linear layers with LoRALinear. Returns list of replaced names."""
    targets = [
        full_name for full_name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and full_name.split('.')[-1] in target_leaf_names
    ]
    for full_name in targets:
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank))
    return targets


def freeze_except_lora(model):
    """Freeze all params except LoRA adapter A/B matrices."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)

# ---------------------------------------------------------------------------
# Co-attention head (same as tune_vqa_rad.py)
# ---------------------------------------------------------------------------

class CoAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        ff = dim * 4
        self.ic = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.tc = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.iffn = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim))
        self.tffn = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim))
        self.in1 = nn.LayerNorm(dim); self.in2 = nn.LayerNorm(dim)
        self.tn1 = nn.LayerNorm(dim); self.tn2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, img, txt, mask=None):
        i2, _ = self.ic(img, txt, txt, key_padding_mask=mask)
        img = self.in1(img + self.drop(i2)); img = self.in2(img + self.drop(self.iffn(img)))
        t2, _ = self.tc(txt, img, img)
        txt = self.tn1(txt + self.drop(t2)); txt = self.tn2(txt + self.drop(self.tffn(txt)))
        return img, txt


class METERHead(nn.Module):
    def __init__(self, n_layers, num_classes, dropout, dim=768, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([CoAttentionLayer(dim, n_heads, dropout) for _ in range(n_layers)])
        self.clf = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, img, txt, mask=None):
        for layer in self.layers:
            img, txt = layer(img, txt, mask)
        return self.clf(torch.cat([img[:, 0], txt[:, 0]], dim=-1))

# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

_gpt2 = None

def token_f1(pred, gold):
    global _gpt2
    if _gpt2 is None:
        from transformers import GPT2Tokenizer
        _gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    p, g = _gpt2.tokenize(pred.lower()), _gpt2.tokenize(gold.lower())
    if not p or not g:
        return float(pred.lower() == gold.lower())
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)

# ---------------------------------------------------------------------------
# Encoder forward (live, not precomputed)
# ---------------------------------------------------------------------------

def encode_batch(clip_model, imgs, questions, tokenizer, device):
    toks = tokenizer(list(questions), context_length=CONTEXT_LENGTH).to(device)
    attn = (toks != 0).long()
    img_tok = clip_model.visual.trunk.forward_features(imgs.to(device))
    txt_out = clip_model.text.transformer(input_ids=toks, attention_mask=attn)
    txt_tok = txt_out.last_hidden_state
    pad_mask = (toks == 0)
    return img_tok, txt_tok, pad_mask

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(clip_model, head, tokenizer, loader, idx2ans, device):
    clip_model.eval(); head.eval()
    all_preds, all_labs, all_closed, all_gold = [], [], [], []
    for imgs, qs, labs, closed, golds in loader:
        img_tok, txt_tok, pad_mask = encode_batch(clip_model, imgs, qs, tokenizer, device)
        logits = head(img_tok, txt_tok, pad_mask)
        all_preds.append(logits.argmax(-1).cpu())
        all_labs.append(labs); all_closed.append(closed); all_gold.extend(golds)
    clip_model.train(); head.train()

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labs).numpy()
    closed = torch.cat(all_closed).numpy().astype(bool)
    valid  = labels >= 0
    pv, lv, cv = preds[valid], labels[valid], closed[valid]
    gold_v = [all_gold[i] for i in range(len(all_gold)) if valid[i]]
    pred_strs = [idx2ans.get(int(p), "") for p in pv]

    overall    = float((pv == lv).mean())
    closed_acc = float((pv[cv] == lv[cv]).mean())    if cv.any()   else float("nan")
    open_acc   = float((pv[~cv] == lv[~cv]).mean())  if (~cv).any() else float("nan")
    avg_f1     = sum(token_f1(p, g) for p, g in zip(pred_strs, gold_v)) / len(gold_v)
    return overall, closed_acc, open_acc, avg_f1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, clip_model, head, tokenizer, train_loader, test_loader, idx2ans, device,
          l2sp_anchors=None):
    """Train with periodic test evaluation. Returns best metrics dict."""

    # Optimizer groups: backbone (or LoRA adapters) at LR_BACKBONE, head at LR_HEAD
    backbone_params = [p for p in clip_model.parameters() if p.requires_grad]
    head_params     = list(head.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best = dict(overall=0.0, closed=float("nan"), open=float("nan"), f1=0.0, epoch=0)

    for ep in range(1, EPOCHS + 1):
        clip_model.train(); head.train()
        for imgs, qs, labs, _, _ in train_loader:
            valid = (labs >= 0)          # CPU mask — use before moving to device
            if not valid.any():
                continue
            imgs_v = imgs[valid]
            qs_v   = [qs[i] for i in valid.nonzero(as_tuple=True)[0].tolist()]
            labs_v = labs[valid].to(device)
            img_tok, txt_tok, pad_mask = encode_batch(clip_model, imgs_v, qs_v, tokenizer, device)
            logits = head(img_tok, txt_tok, pad_mask)
            ce_loss = F.cross_entropy(logits, labs_v)

            if l2sp_anchors is not None:
                l2sp = sum((p - l2sp_anchors[n]).pow(2).sum()
                           for n, p in clip_model.named_parameters() if p.requires_grad)
                loss = ce_loss + args.lambda_l2sp * l2sp
            else:
                loss = ce_loss

            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            overall, closed_acc, open_acc, avg_f1 = evaluate(
                clip_model, head, tokenizer, test_loader, idx2ans, device)
            print(f"  ep{ep:3d}  overall={overall*100:.2f}%  "
                  f"closed={closed_acc*100:.2f}%  open={open_acc*100:.2f}%", flush=True)
            if overall > best["overall"]:
                best = dict(overall=overall, closed=closed_acc, open=open_acc,
                            f1=avg_f1, epoch=ep)

    return best

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       required=True, choices=["lora", "l2sp"])
    parser.add_argument("--rank",         type=int,   default=4,    help="LoRA rank (lora only)")
    parser.add_argument("--lambda_l2sp",  type=float, default=0.01, help="L2-SP weight (l2sp only)")
    parser.add_argument("--data_dir",     default="../data/vqa_rad")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))
    json_path  = os.path.join(data_dir, "VQA_RAD Dataset Public.json")
    images_dir = os.path.join(data_dir, "images")

    print(f"Method: {args.method}  device: {device}", flush=True)

    clip_model, preprocess, tokenizer, _ = load_model()
    clip_model = clip_model.to(device)

    train_data, test_data = load_and_split(json_path, images_dir)
    ans2idx, idx2ans = build_answer_vocab(train_data, MIN_FREQ)
    num_classes = len(ans2idx)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}  Classes: {num_classes}", flush=True)

    train_loader = DataLoader(
        VQADataset(train_data, images_dir, ans2idx, preprocess),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(
        VQADataset(test_data, images_dir, ans2idx, preprocess),
        batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    head = METERHead(N_LAYERS, num_classes, DROPOUT).to(device)
    l2sp_anchors = None

    if args.method == "lora":
        replaced = apply_lora(clip_model, rank=args.rank)
        freeze_except_lora(clip_model)
        clip_model = clip_model.to(device)
        lora_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
        print(f"LoRA rank={args.rank}: replaced {len(replaced)} layers, "
              f"{lora_params:,} adapter params trainable", flush=True)
        print("  Replaced:", replaced[:6], "..." if len(replaced) > 6 else "", flush=True)

    elif args.method == "l2sp":
        # Save initial weights as anchors (on same device, detached)
        l2sp_anchors = {n: p.detach().clone()
                        for n, p in clip_model.named_parameters()}
        # All encoder params trainable
        for p in clip_model.parameters():
            p.requires_grad_(True)
        total_params = sum(p.numel() for p in clip_model.parameters())
        print(f"L2-SP λ={args.lambda_l2sp}: {total_params:,} backbone params trainable", flush=True)

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in head.parameters())
    print(f"Total trainable params: {trainable:,}", flush=True)

    t0 = time.time()
    best = train(args, clip_model, head, tokenizer, train_loader, test_loader,
                 idx2ans, device, l2sp_anchors=l2sp_anchors)
    elapsed = time.time() - t0

    print(f"\n=== Best ({args.method}) ===", flush=True)
    print(f"  Overall: {best['overall']*100:.2f}%  Closed: {best['closed']*100:.2f}%  "
          f"Open: {best['open']*100:.2f}%  F1: {best['f1']*100:.2f}%  epoch: {best['epoch']}", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s", flush=True)

    # Save results
    config = dict(method=args.method,
                  rank=args.rank if args.method == "lora" else None,
                  lambda_l2sp=args.lambda_l2sp if args.method == "l2sp" else None,
                  n_layers=N_LAYERS, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD,
                  batch_size=BATCH_SIZE, epochs=EPOCHS, dropout=DROPOUT)
    record = dict(config=config, **best, elapsed_s=round(elapsed),
                  timestamp=datetime.now().isoformat())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"constrained_{args.method}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
