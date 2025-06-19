from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
import os
from typing import Union
from jinja2 import Template
from datasets import Dataset
from src.myutils.file import read_txt


class DatasetSize(Enum):
    """Enum to represent the size of the dataset."""

    Tiny = 5
    Small = 200
    Full = -1


class PostInitMeta(ABCMeta):
    """Post init metaclass to check some conditions after the class is initialized."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not hasattr(instance, "dataset") or not isinstance(
            instance.dataset, Dataset
        ):
            raise ValueError("Dataset must be initialized with a hf Dataset object.")
        if instance.size == "tiny":
            instance.size = DatasetSize.Tiny
        elif instance.size == "small":
            instance.size = DatasetSize.Small
        elif instance.size == "full":
            instance.size = DatasetSize.Full

        # Compute the sample size
        sample_size = len(instance.dataset) if instance.size == DatasetSize.Full else instance.size.value

        print("[Debug PostInitmeta]: Before instance size {}".format(len(instance.dataset)))


        instance.dataset = instance.dataset.shuffle(seed=42).select(
            range(min(sample_size, len(instance.dataset)))
        )
        
        if hasattr(instance, "references") and instance.references is not None:
            instance.references = [data["highlights"] for data in instance.dataset]

        # print("-"*50+"Debugging the dataset base"+"-"*50)
        # print("="*5+"Prompt"+"="*5)
        # print([instance.prompt_template.render(**data) for data in instance.dataset][0])
        # print("="*5+"Reference"+"="*5)
        # print([data["highlights"] for data in instance.dataset][0])

        # for data in instance.dataset:
        #     print("="*5+"Prompt"+"="*5)
        #     print()
        #     print("="*5+"Reference"+"="*5)
        #     print(data["highlights"])


        return instance


class DatasetBase(ABC, metaclass=PostInitMeta):

    def __init__(
        self,
        size: Union[str, int] = "tiny",
        use_fewshot=False,
    ):
        self.size = size
        self.use_fewshot = use_fewshot and os.path.isfile(
            os.path.join(self.cls_abspath, "prompt_fewshot.txt")
        )
        self.dataset: Dataset
        if not isinstance(size, (str, int)):
            raise ValueError("Size must be a string or an integer.")
        if isinstance(size, str) and size not in ["tiny", "small", "full"]:
            raise ValueError(
                "Size must be one of the following: 'tiny', 'small', 'full' if string."
            )

    @property
    def cls_abspath(self):
        """
        Get the absolute path of the subclass
        """
        return os.path.abspath(
            os.path.sep.join(self.__class__.__module__.split(".")[:-1])
        )

    @property
    def prompt_template(self) -> Template:
        """
        Get the prompt template for the dataset.
        """
        return Template(
            read_txt(
                os.path.join(
                    self.cls_abspath,
                    "prompt_fewshot.txt" if self.use_fewshot else "prompt_zeroshot.txt",
                )
            )
        )

    @property
    def prompts(self):
        """
        Generate prompts for the dataset
        """
        return [self.prompt_template.render(**data) for data in self.dataset]

    @abstractmethod
    def evaluate(self, preds):
        """
        Evaluate the model output
        """
        raise NotImplementedError("evaluate method must be implemented in the subclass")
