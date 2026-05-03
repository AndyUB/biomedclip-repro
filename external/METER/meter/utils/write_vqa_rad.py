import json
import os
import io
import pandas as pd
import pyarrow as pa
from collections import defaultdict
from tqdm import tqdm


def get_image_id(image_name):
    # "synpic54610.jpg" -> 54610
    return int(image_name.replace("synpic", "").replace(".jpg", ""))


def make_arrow(json_path, images_dir, dataset_root):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    train_data = [d for d in data if not d["phrase_type"].startswith("test_")]
    test_data = [d for d in data if d["phrase_type"].startswith("test_")]

    # Answer vocabulary from training answers only (sorted for reproducibility)
    all_answers = [str(d["answer"]).lower().strip() for d in train_data]
    unique_answers = sorted(set(all_answers))
    ans2label = {a: i for i, a in enumerate(unique_answers)}
    label2ans = unique_answers

    print(f"Answer vocabulary size: {len(label2ans)}")

    for split, split_data in [("train", train_data), ("test", test_data)]:
        image_groups = defaultdict(list)
        for d in split_data:
            image_name = d["image_name"]
            if os.path.isfile(os.path.join(images_dir, image_name)):
                image_groups[image_name].append(d)

        rows = []
        for image_name, qas in tqdm(image_groups.items(), desc=f"Processing {split}"):
            with open(os.path.join(images_dir, image_name), "rb") as fp:
                binary = fp.read()

            image_id = get_image_id(image_name)
            questions = [d["question"] for d in qas]
            answers_str = [[str(d["answer"]).lower().strip()] for d in qas]

            answer_labels = []
            answer_scores = []
            for d in qas:
                ans = str(d["answer"]).lower().strip()
                if ans in ans2label:
                    answer_labels.append([ans2label[ans]])
                    answer_scores.append([1.0])
                else:
                    # OOV answer (only occurs in test set)
                    answer_labels.append([])
                    answer_scores.append([])

            qids = [int(d["qid"]) for d in qas]
            answer_types = [d.get("answer_type", "OPEN").strip().upper() for d in qas]

            rows.append([
                binary, questions, answers_str, answer_labels, answer_scores,
                image_id, qids, split, answer_types,
            ])

        dataframe = pd.DataFrame(
            rows,
            columns=[
                "image", "questions", "answers", "answer_labels", "answer_scores",
                "image_id", "question_id", "split", "answer_types",
            ],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        arrow_path = f"{dataset_root}/vqa_rad_{split}.arrow"
        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        n_qa = sum(len(r[1]) for r in rows)
        print(f"Wrote {split}: {len(rows)} images, {n_qa} QA pairs -> {arrow_path}")

    # Save vocabulary for reference
    with open(f"{dataset_root}/vqa_rad_ans2label.json", "w") as fp:
        json.dump(ans2label, fp, indent=2)
    with open(f"{dataset_root}/vqa_rad_label2ans.json", "w") as fp:
        json.dump(label2ans, fp, indent=2)
