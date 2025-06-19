import os
import re
import json
from datasets import load_from_disk
from src.mydatasets.dataset_base import DatasetBase


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        super().__init__(size=size, use_fewshot=use_fewshot)
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))["test"]

    def evaluate(self, preds):
        self.preds = [p.split("\n\nQuestion:")[0] for p in preds]
        self.pred_ans = []
        self.ground_ans = []
        correct = 0
        for i, pred in enumerate(preds):
            # print("-"*100)
            # print("Pred {}".format(pred))
            pred_ans = self.find_answer(pred)
            ground_ans = self.find_answer(self.dataset[i]["answer"])
            self.pred_ans.append(pred_ans)
            self.ground_ans.append(ground_ans)
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

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, "results.jsonl")
        results = []
        for pred, data, pred_ans, ground_ans in zip(self.preds, self.dataset, self.pred_ans, self.ground_ans):
            result = {
                "prompt": data["question"],
                "generation": pred,
                "reference": data["answer"],
                "pred_ans": pred_ans,
                "ground_ans": ground_ans,
                "correct": pred_ans==ground_ans
            }
            results.append(result)

        with open(file_path, "w") as f:
            for r in results:
                json.dump(r,f)
                f.write("\n")


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
