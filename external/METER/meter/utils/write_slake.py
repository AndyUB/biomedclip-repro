"""Convert SLAKE (English only) to Arrow format for METER.

Usage:
    python -m meter.utils.write_slake \
        --slake_dir data/slake \
        --output_dir data/slake/arrow
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def make_arrow(slake_dir: str, output_dir: str):
    split_files = {
        "train": "train.json",
        "val":   "validation.json",
        "test":  "test.json",
    }

    # Build answer vocabulary from English training data only
    with open(os.path.join(slake_dir, split_files["train"])) as f:
        train_data = json.load(f)
    train_en = [d for d in train_data if d.get("q_lang") == "en"]
    unique_answers = sorted(set(str(d["answer"]).lower().strip() for d in train_en))
    ans2label = {a: i for i, a in enumerate(unique_answers)}
    label2ans = unique_answers
    print(f"Answer vocabulary size: {len(label2ans)}")

    os.makedirs(output_dir, exist_ok=True)

    for split, fname in split_files.items():
        with open(os.path.join(slake_dir, fname)) as f:
            raw = json.load(f)
        data = [d for d in raw if d.get("q_lang") == "en"]
        print(f"{split}: {len(data)} English QA pairs")

        # Group by image
        image_groups = defaultdict(list)
        for d in data:
            image_groups[d["img_name"]].append(d)

        rows = []
        oov_count = 0
        for img_name, qas in tqdm(image_groups.items(), desc=f"Processing {split}"):
            img_path = os.path.join(slake_dir, "imgs", img_name)
            if not os.path.isfile(img_path):
                continue
            with open(img_path, "rb") as fp:
                binary = fp.read()

            questions = [d["question"] for d in qas]
            answers_str = [[str(d["answer"]).lower().strip()] for d in qas]

            answer_labels, answer_scores = [], []
            for d in qas:
                ans = str(d["answer"]).lower().strip()
                if ans in ans2label:
                    answer_labels.append([ans2label[ans]])
                    answer_scores.append([1.0])
                else:
                    oov_count += 1
                    answer_labels.append([])
                    answer_scores.append([])

            qids = [int(d["qid"]) for d in qas]
            answer_types = [d.get("answer_type", "OPEN").strip().upper() for d in qas]
            img_id = int(qas[0]["img_id"])

            rows.append([
                binary, questions, answers_str, answer_labels, answer_scores,
                img_id, qids, split, answer_types,
            ])

        if oov_count:
            print(f"  OOV answers (counted wrong): {oov_count}")

        dataframe = pd.DataFrame(
            rows,
            columns=[
                "image", "questions", "answers", "answer_labels", "answer_scores",
                "image_id", "question_id", "split", "answer_types",
            ],
        )
        table = pa.Table.from_pandas(dataframe)
        arrow_path = os.path.join(output_dir, f"slake_{split}.arrow")
        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        n_qa = sum(len(r[1]) for r in rows)
        print(f"Wrote {split}: {len(rows)} images, {n_qa} QA pairs -> {arrow_path}")

    import json as _json
    with open(os.path.join(output_dir, "slake_ans2label.json"), "w") as fp:
        _json.dump(ans2label, fp, indent=2)
    with open(os.path.join(output_dir, "slake_label2ans.json"), "w") as fp:
        _json.dump(label2ans, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slake_dir", default="data/slake")
    parser.add_argument("--output_dir", default="data/slake/arrow")
    args = parser.parse_args()
    make_arrow(args.slake_dir, args.output_dir)
