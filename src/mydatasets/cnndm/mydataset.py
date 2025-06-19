import os
import re
import json
from datasets import load_from_disk
import evaluate
from src.mydatasets.dataset_base import DatasetBase


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))
        super().__init__(size=size, use_fewshot=use_fewshot)
        self.references = [data["highlights"] for data in self.dataset]

        print("Debug CNNDM: ", len(self.dataset), len(self.references))
        print("Len Prompts: ", len(self.prompts))

    def evaluate(self, predictions):
        self.predictions = [re.split("\n\nArticle:", p)[0] for p in predictions]
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=self.predictions, references=self.references)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, "results.jsonl")
        results = []
        for article, pred, ref in zip(self.dataset['article'], self.predictions, self.references):
            result = {
                "prompt": article,
                "generation": pred,
                "reference": ref
            }
            results.append(result)

        with open(file_path, "w") as f:
            for r in results:
                json.dump(r,f)
                f.write("\n")
            
        


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
