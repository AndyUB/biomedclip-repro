"""Iterative hyperparameter search for frozen-encoder co-attention approach.

Strategy: coarse pass → identify promising regions → fine-grained follow-up rounds.

This script runs one round. Edit GRID and ROUND before each run.
Results stream live into notes/vqa_rad_tuning_r{ROUND}.md.

Multi-GPU: 2 workers run in parallel, each setting CUDA_VISIBLE_DEVICES and
precomputing BiomedCLIP features independently on their assigned GPU.

Collapse detection: if after epoch COLLAPSE_CHECK_START the model predicts fewer
than COLLAPSE_MIN_UNIQUE distinct classes, training stops early.

Usage:
    cd biomedclip-repro/src
    python tune_vqa_rad.py --data_dir ../data/vqa_rad
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_biomedclip import load_model, CONTEXT_LENGTH

# ---------------------------------------------------------------------------
# Round config — edit these between rounds
# ---------------------------------------------------------------------------

ROUND = 3   # increment each round

# Coarse pass: 3×3×2 = 18 runs — wide spread, one dropout value
# After this: narrow lr/n_layers around winners, add dropout as axis
GRID = list(product(
    [2, 4, 6],          # n_layers
    [5e-5, 1e-4, 2e-4], # lr
    [16, 32],           # batch_size  (key variable from round 2)
))
# Fixed this round
DROPOUT  = 0.1
EPOCHS   = 100
N_HEADS  = 8
MIN_FREQ = 2

# Collapse detection
COLLAPSE_CHECK_START = 8
COLLAPSE_CHECK_EVERY = 4
COLLAPSE_MIN_UNIQUE  = 5   # out of ~216 classes

NOTES_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"../notes/vqa_rad_tuning_r{ROUND}.md")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"../results/tuning_results_r{ROUND}.json")

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


class RawDataset(Dataset):
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


def _collate_raw(batch):
    imgs, qs, labs, closed, golds = zip(*batch)
    return (torch.stack(imgs), list(qs),
            torch.tensor(labs, dtype=torch.long),
            torch.tensor(closed, dtype=torch.bool),
            list(golds))


@torch.no_grad()
def precompute(clip_model, raw_ds, tokenizer, device, batch_size=32):
    loader = DataLoader(raw_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=_collate_raw)
    all_img, all_txt, all_mask, all_labs, all_closed, all_gold = [], [], [], [], [], []
    clip_model.eval()
    for imgs, qs, labs, closed, golds in tqdm(loader, desc="precompute", leave=False):
        toks = tokenizer(list(qs), context_length=CONTEXT_LENGTH).to(device)
        attn = (toks != 0).long()
        img_t = clip_model.visual.trunk.forward_features(imgs.to(device))
        txt_t = clip_model.text.transformer(input_ids=toks, attention_mask=attn).last_hidden_state
        all_img.append(img_t.cpu()); all_txt.append(txt_t.cpu())
        all_mask.append((toks == 0).cpu()); all_labs.append(labs)
        all_closed.append(closed); all_gold.extend(golds)
    return (torch.cat(all_img), torch.cat(all_txt), torch.cat(all_mask),
            torch.cat(all_labs), torch.cat(all_closed), all_gold)

# ---------------------------------------------------------------------------
# Model
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
        img = self.in1(img + self.drop(i2))
        img = self.in2(img + self.drop(self.iffn(img)))
        t2, _ = self.tc(txt, img, img)
        txt = self.tn1(txt + self.drop(t2))
        txt = self.tn2(txt + self.drop(self.tffn(txt)))
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

_gpt2_tok = None

def token_f1(pred, gold):
    global _gpt2_tok
    if _gpt2_tok is None:
        from transformers import GPT2Tokenizer
        _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    p = _gpt2_tok.tokenize(pred.lower())
    g = _gpt2_tok.tokenize(gold.lower())
    if not p or not g:
        return float(pred.lower() == gold.lower())
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)

# ---------------------------------------------------------------------------
# Train + evaluate one config
# ---------------------------------------------------------------------------

def train_one_config(n_layers, lr, batch_size, num_classes,
                     tr_img, tr_txt, tr_mask, tr_labels,
                     te_img, te_txt, te_mask, te_labels, te_closed, te_gold,
                     idx2ans, device):
    dropout = DROPOUT
    head = METERHead(n_layers, num_classes, dropout).to(device)
    opt  = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    tr_ds = TensorDataset(tr_img, tr_txt, tr_mask, tr_labels)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    probe_img  = tr_img[:128]
    probe_txt  = tr_txt[:128]
    probe_mask = tr_mask[:128]

    collapsed, ep = False, 0
    for ep in range(1, EPOCHS + 1):
        head.train()
        for img, txt, mask, lab in tr_loader:
            img, txt, mask, lab = img.to(device), txt.to(device), mask.to(device), lab.to(device)
            valid = lab >= 0
            if not valid.any():
                continue
            loss = F.cross_entropy(head(img[valid], txt[valid], mask[valid]), lab[valid])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep >= COLLAPSE_CHECK_START and
                (ep - COLLAPSE_CHECK_START) % COLLAPSE_CHECK_EVERY == 0):
            head.eval()
            with torch.no_grad():
                preds = head(probe_img.to(device), probe_txt.to(device),
                             probe_mask.to(device)).argmax(-1)
            head.train()
            if len(preds.unique()) < COLLAPSE_MIN_UNIQUE:
                collapsed = True
                break

    # Evaluate
    head.eval()
    te_loader = DataLoader(TensorDataset(te_img, te_txt, te_mask, te_labels, te_closed),
                           batch_size=64, shuffle=False, num_workers=0)
    all_preds, all_labs, all_closed_out = [], [], []
    with torch.no_grad():
        for img, txt, mask, lab, clo in te_loader:
            all_preds.append(head(img.to(device), txt.to(device), mask.to(device)).argmax(-1).cpu())
            all_labs.append(lab); all_closed_out.append(clo)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labs).numpy()
    closed = torch.cat(all_closed_out).numpy().astype(bool)
    valid  = labels >= 0
    pv, lv, cv = preds[valid], labels[valid], closed[valid]
    gold_v    = [te_gold[i] for i in range(len(te_gold)) if valid[i]]
    pred_strs = [idx2ans.get(int(p), "") for p in pv]

    overall    = float((pv == lv).mean())
    closed_acc = float((pv[cv] == lv[cv]).mean())   if cv.any()  else float("nan")
    open_acc   = float((pv[~cv] == lv[~cv]).mean()) if (~cv).any() else float("nan")
    avg_f1     = sum(token_f1(p, g) for p, g in zip(pred_strs, gold_v)) / len(gold_v)

    return dict(overall=overall, closed=closed_acc, open=open_acc, f1=avg_f1,
                collapsed=collapsed, epochs_run=ep)

# ---------------------------------------------------------------------------
# Worker (one process per GPU)
# ---------------------------------------------------------------------------

def worker(gpu_id, config_list, data_dir, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")  # maps to physical gpu_id via env var

    print(f"[GPU{gpu_id}] Precomputing features...", flush=True)
    clip_model, preprocess, tokenizer, _ = load_model()

    json_path  = os.path.join(data_dir, "VQA_RAD Dataset Public.json")
    images_dir = os.path.join(data_dir, "images")
    train_data, test_data = load_and_split(json_path, images_dir)
    ans2idx, idx2ans = build_answer_vocab(train_data, MIN_FREQ)
    num_classes = len(ans2idx)

    tr_ds = RawDataset(train_data, images_dir, ans2idx, preprocess)
    te_ds = RawDataset(test_data,  images_dir, ans2idx, preprocess)
    tr_img, tr_txt, tr_mask, tr_labels, _, _          = precompute(clip_model, tr_ds, tokenizer, device)
    te_img, te_txt, te_mask, te_labels, te_closed, te_gold = precompute(clip_model, te_ds, tokenizer, device)
    del clip_model; torch.cuda.empty_cache()
    print(f"[GPU{gpu_id}] Features ready. Running {len(config_list)} configs.", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").close()  # clear

    for run_idx, (n_layers, lr, batch_size) in config_list:
        print(f"[GPU{gpu_id}] [{run_idx}/{len(GRID)}] "
              f"n_layers={n_layers} lr={lr:.0e} bs={batch_size}", flush=True)
        t0 = time.time()
        torch.manual_seed(42)

        m = train_one_config(
            n_layers=n_layers, lr=lr, batch_size=batch_size, num_classes=num_classes,
            tr_img=tr_img, tr_txt=tr_txt, tr_mask=tr_mask, tr_labels=tr_labels,
            te_img=te_img, te_txt=te_txt, te_mask=te_mask,
            te_labels=te_labels, te_closed=te_closed, te_gold=te_gold,
            idx2ans=idx2ans, device=device,
        )
        elapsed = round(time.time() - t0)
        tag = "COLLAPSED" if m["collapsed"] else f"ep{m['epochs_run']}"
        print(f"[GPU{gpu_id}]   {m['overall']*100:.2f}% overall  "
              f"{m['open']*100:.2f}% open  [{tag}]  ({elapsed}s)", flush=True)

        record = dict(run=run_idx, gpu=gpu_id, n_layers=n_layers, lr=lr,
                      batch_size=batch_size, dropout=DROPOUT, elapsed_s=elapsed, **m)
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def write_markdown(results):
    results_s = sorted(results, key=lambda r: r["run"])
    hdr = (
        f"# VQA-RAD Tuning — Round {ROUND}\n\n"
        f"Frozen BiomedCLIP + METER co-attention head.\n"
        f"Paper: 72.70% overall / 67.00% open. Baseline: 59.24%.\n\n"
        f"Grid: n_layers∈[2,4,6] × lr∈[5e-5,1e-4,2e-4] × bs∈[16,32] = {len(GRID)} runs.\n"
        f"Fixed: dropout={DROPOUT}, epochs={EPOCHS}. "
        f"Collapse stops if <{COLLAPSE_MIN_UNIQUE} unique classes predicted.\n"
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"## Results\n\n"
        f"| # | n_layers | lr | bs | Overall | Closed | Open | F1 | ep | GPU |\n"
        f"|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results_s:
        flag = " ⚠" if r.get("collapsed") else ""
        rows.append(
            f"| {r['run']} | {r['n_layers']} | {r['lr']:.0e} | {r['batch_size']} "
            f"| {r['overall']*100:.2f}% | {r['closed']*100:.2f}% | {r['open']*100:.2f}% "
            f"| {r['f1']*100:.2f}% | {r['epochs_run']}{flag} | {r['gpu']} |\n"
        )
    footer = ""
    if results:
        best = max(results, key=lambda r: r["overall"])
        footer = (
            f"\n## Best so far\n\n"
            f"n_layers={best['n_layers']} lr={best['lr']:.0e} bs={best['batch_size']} "
            f"→ **{best['overall']*100:.2f}%** overall "
            f"/ {best['closed']*100:.2f}% closed / {best['open']*100:.2f}% open\n"
        )
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    with open(NOTES_PATH, "w") as f:
        f.write(hdr + "".join(rows) + footer)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/vqa_rad")
    args = parser.parse_args()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))

    # Interleave configs across GPUs for balanced load
    gpu0 = [(2*i + 1, GRID[2*i])     for i in range(len(GRID) // 2 + len(GRID) % 2)]
    gpu1 = [(2*i + 2, GRID[2*i + 1]) for i in range(len(GRID) // 2)]

    tmp0, tmp1 = "/tmp/tune_r3_gpu0.jsonl", "/tmp/tune_r3_gpu1.jsonl"
    write_markdown([])

    print(f"Round {ROUND}: {len(GRID)} configs across 2 GPUs ({len(gpu0)} + {len(gpu1)}).")
    print(f"dropout={DROPOUT} epochs={EPOCHS} collapse_check starts ep{COLLAPSE_CHECK_START}")

    p0 = mp.Process(target=worker, args=(0, gpu0, data_dir, tmp0))
    p1 = mp.Process(target=worker, args=(1, gpu1, data_dir, tmp1))
    p0.start(); p1.start()

    seen, all_results = set(), []
    while p0.is_alive() or p1.is_alive():
        time.sleep(20)
        for path in [tmp0, tmp1]:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r["run"] not in seen:
                            seen.add(r["run"]); all_results.append(r)
                    except (json.JSONDecodeError, KeyError):
                        pass
        if all_results:
            write_markdown(all_results)

    p0.join(); p1.join()

    # Final sweep
    for path in [tmp0, tmp1]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r["run"] not in seen:
                        seen.add(r["run"]); all_results.append(r)
                except (json.JSONDecodeError, KeyError):
                    pass

    write_markdown(all_results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(sorted(all_results, key=lambda r: r["run"]), f, indent=2)

    if all_results:
        best = max(all_results, key=lambda r: r["overall"])
        print(f"\n=== Best: n_layers={best['n_layers']} lr={best['lr']:.0e} "
              f"bs={best['batch_size']} → {best['overall']*100:.2f}% overall ===")
    print(f"Notes:   {NOTES_PATH}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
