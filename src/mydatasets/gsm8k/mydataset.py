import os
import re
from datasets import load_from_disk
from src.mydatasets.dataset_base import DatasetBase


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        super().__init__(size=size, use_fewshot=use_fewshot)
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))["test"]

    def evaluate(self, preds):
        preds = [p.split("\n\nQuestion:")[0] for p in preds]
        correct = 0
        for i, pred in enumerate(preds):
            pred_ans = self.find_answer(pred)
            ground_ans = self.find_answer(self.dataset[i]["answer"])
            if pred_ans == ground_ans:
                correct += 1
        result = float(correct) / len(preds)
        return result

    def find_answer(self, text):
        match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
        if match:
            return round(float(match.group(1)))
        else:
            all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
            if all_m:
                return round(float(all_m[-1]))
        return "No answer found"


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
