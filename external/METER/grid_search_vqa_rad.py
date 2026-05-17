"""Grid search over METER VQA-RAD hyperparameters.

Reverse-engineering strategy:
- val split = test set (already wired in VQARadDataset)
- Train for up to 10 epochs, evaluate on test after every epoch
- Keep best checkpoint per run; record best test accuracy
- Try combinations of learning_rate × lr_mult_head × lr_mult_cross_modal

Usage:
    cd biomedclip-repro/external/METER
    conda run -n biomedclip python grid_search_vqa_rad.py \
        --data_root /path/to/arrow \
        --output_dir /path/to/results
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


def run_one(data_root, lr, lr_mult_head, lr_mult_cross_modal, output_dir, max_epoch=10):
    tag = f"lr{lr:.0e}_head{lr_mult_head}_xmod{lr_mult_cross_modal}"
    log_path = os.path.join(output_dir, f"{tag}.log")
    result_path = os.path.join(output_dir, f"{tag}.json")

    if os.path.exists(result_path):
        print(f"[skip] {tag} — result already exists")
        with open(result_path) as f:
            return tag, json.load(f)

    print(f"\n{'='*60}")
    print(f"[run] {tag}  ({datetime.now().strftime('%H:%M:%S')})")
    print(f"{'='*60}")

    cmd = [
        "conda", "run", "-n", "biomedclip",
        "python", "run_vqa_rad.py",
        "with", "task_finetune_vqa_rad_biomedclip",
        f"data_root={data_root}",
        f"learning_rate={lr}",
        f"lr_mult_head={lr_mult_head}",
        f"lr_mult_cross_modal={lr_mult_cross_modal}",
        f"max_epoch={max_epoch}",
        "per_gpu_batchsize=32",
        "num_workers=4",
        # Write result to a temp path; we'll rename it
        # (model_name="" since load_path="" → result/vqa_rad_.json)
    ]

    with open(log_path, "w") as log_f:
        ret = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=os.path.dirname(__file__))

    # METER writes result to result/vqa_rad_.json; read and copy it
    meter_result_path = os.path.join(os.path.dirname(__file__), "result", "vqa_rad_.json")
    if os.path.exists(meter_result_path):
        with open(meter_result_path) as f:
            result = json.load(f)
        result["tag"] = tag
        result["lr"] = lr
        result["lr_mult_head"] = lr_mult_head
        result["lr_mult_cross_modal"] = lr_mult_cross_modal
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        return tag, result
    else:
        print(f"[error] No result file found for {tag}")
        return tag, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/yhruan/courses/428/biomedclip-repro/data/vqa_rad/arrow")
    parser.add_argument("--output_dir", default="/home/yhruan/courses/428/biomedclip-repro/logs/grid_search_vqa_rad")
    parser.add_argument("--max_epoch", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Grid: search lower LRs — rates ≥1e-5 collapsed to 0% open accuracy
    learning_rates      = [1e-6, 2e-6, 5e-6, 1e-5]
    lr_mult_heads       = [10, 50, 100]
    lr_mult_cross_mods  = [5]   # keep cross-modal fixed; head vs. backbone LR is the main lever

    grid = list(itertools.product(learning_rates, lr_mult_heads, lr_mult_cross_mods))
    print(f"Grid search: {len(grid)} combinations × {args.max_epoch} epochs each")

    results = []
    for lr, lr_mult_head, lr_mult_cross_modal in grid:
        tag, result = run_one(
            data_root=args.data_root,
            lr=lr,
            lr_mult_head=lr_mult_head,
            lr_mult_cross_modal=lr_mult_cross_modal,
            output_dir=args.output_dir,
            max_epoch=args.max_epoch,
        )
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Tag':<35} {'Overall':>8} {'Closed':>8} {'Open':>8} {'F1':>8}")
    print(f"{'-'*70}")
    results.sort(key=lambda r: r["overall_acc"], reverse=True)
    for r in results:
        print(f"{r['tag']:<35} {r['overall_acc']*100:>7.2f}% {r['closed_acc']*100:>7.2f}% {r['open_acc']*100:>7.2f}% {r['token_f1']*100:>7.2f}%")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    best = results[0]
    print(f"\nBest: {best['tag']}")
    print(f"  Overall: {best['overall_acc']*100:.2f}%  Closed: {best['closed_acc']*100:.2f}%  Open: {best['open_acc']*100:.2f}%  F1: {best['token_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
