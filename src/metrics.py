import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report


def compute_accuracy(preds, labels):
    return accuracy_score(labels.numpy(), preds.numpy())


def report(name, preds, labels, class_names, results_dir="results"):
    acc = compute_accuracy(preds, labels)
    report_str = classification_report(
        labels.numpy(), preds.numpy(), target_names=class_names
    )
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(report_str)

    out = {
        "dataset": name,
        "accuracy": acc,
        "classification_report": report_str,
    }
    Path(results_dir).mkdir(exist_ok=True)
    out_path = Path(results_dir) / f"{name.lower().replace(' ', '_')}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")
    return acc
