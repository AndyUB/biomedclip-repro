from .base_dataset import BaseDataset


class VQARadDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_rad_train"]
        elif split == "val":
            # VQA-RAD has no separate val set; use test for checkpoint selection
            # (standard practice in VQA-RAD papers)
            names = ["vqa_rad_test"]
        elif split == "test":
            names = ["vqa_rad_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        answer_type = self.table["answer_types"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
            "answer_type": answer_type,
        }
