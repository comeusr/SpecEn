import os
import subprocess
import warnings
from datasets import load_from_disk
from src.mydatasets.dataset_base import DatasetBase


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        super().__init__(size=size, use_fewshot=use_fewshot)
        if use_fewshot:
            warnings.warn("Few-shot learning is not supported for humaneval dataset")
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))["test"]

    @property
    def prompts(self):
        return self.dataset["prompt"]

    def evaluate(self, preds):
        for i, p in enumerate(preds):
            preds[i] = self.clean_pred(p)
        n_correct = 0
        for i, pred in enumerate(preds):
            n_correct += self.check_correctness(self.dataset[i], pred, timeout=5)
        return n_correct / len(preds)

    def check_correctness(self, problem, completion, timeout):
        func = problem["prompt"] + "    " + completion.strip()
        code = (
            func
            + "\n\n"
            + problem["test"]
            + "\n\n"
            + f'check({problem["entry_point"]})'
        )
        return not self.check_python_program_error(code, timeout=timeout)

    def check_python_program_error(self, python_program, timeout):
        try:
            result = subprocess.run(
                ["python", "-c", python_program],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode != 0
        except:
            return True

    def clean_pred(self, pred):
        clean_list = ["</s><s>", "\n\n\n", "\n\ndef"]
        for cl in clean_list:
            if cl in pred:
                pred = pred.split(cl)[0]
        return pred


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
