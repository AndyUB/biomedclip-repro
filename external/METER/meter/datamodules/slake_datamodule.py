from collections import defaultdict

from ..datasets import SlakeDataset
from .datamodule_base import BaseDataModule


class SlakeDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SlakeDataset

    @property
    def dataset_name(self):
        return "slake"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()

        all_answers = [c for c in train_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1

        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k
