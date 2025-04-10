import os
import re
from datasets import load_from_disk
from src.mydatasets.dataset_base import DatasetBase
from src.myutils.file import load_json


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        super().__init__(size=size, use_fewshot=use_fewshot)
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))["test"]
        self.prefixs = load_json(
            os.path.join(self.cls_abspath, "mmlu-cot-claude-single.json")
        )

    @property
    def prompts(self):
        prompts = []
        for data in self.dataset:
            prompt = self.prompt_template.render(**self.modified_data(data))
            if self.use_fewshot:
                prefix = self.prefixs[data["subject"]]
                prompt = f"{prefix}\n\n{prompt}"
            prompts.append(prompt)
        return prompts

    def evaluate(self, preds):
        preds = [p.split("\n\nQ:")[0] for p in preds]
        choices = ["A", "B", "C", "D"]
        correct = 0
        for i, pred in enumerate(preds):
            ans_pred = self.find_answer(pred)
            gold = choices[self.dataset[i]["answer"]]
            if ans_pred == gold:
                correct += 1
        result = float(correct) / len(preds)
        return result

    def find_answer(self, text):
        patterns = [
            r"\b([A-D])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return "Z"

    def modified_data(self, data):
        choices = ["A", "B", "C", "D"]
        data["answer"] = choices[data["answer"]]
        choices_str = ""
        for char, value in zip(["A", "B", "C", "D"], data["choices"]):
            choices_str += f"({char}) {value}\n"
        data["choices_str"] = choices_str.strip()
        return data


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
