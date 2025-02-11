<h1 align="center">Speculative Ensemble</h1>

<p align="center">
<a href="https://arxiv.org/abs/2502.01662v1">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2502.01662v1-red"></a>
</p>



**Speculative Ensemble** is a novel framework that accelerates the ensemble of any number of LLMs without sacrificing performance. It could reach 1.11x-2.23x over standard ensemble techniques on two-model or three-model pairs.


## Setup

```bash
conda create -n vllm_specdec python=3.11 -y
conda activate vllm_specdec

cd vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/c8/f4/e108a902ccad131d8978a9376343a6e95d78d0e12f152a796794647073ec/vllm-0.6.5-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .

cd ..
pip install -r requirements.txt
```

## How to run

1. create a `.env` file to create (automatically) environment variable to root path of your models.
    ```bash
    MODEL_PATH=xxx
    ```

2. run an example:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python ./main_dataset.py \
        dataset.name=humaneval \
        dataset.size=tiny \
        method=sd \
        method.model="llama-2-7b" \
        method.draft_model="llama-2-7b-68m" \
        method.gamma=5 \
        method.generate.temperature=0 \
    ```

## Code reading guides
**Chef** is the internal codename for the speculative ensemble implementation. The code can be found at [`vllm/vllm/chef`](./vllm/vllm/chef/). For the baseline ensemble implementation, see [vllm/vllm/ensemble_decode](./vllm/vllm/ensemble_decode/).

We have implemented multiple model inference methods. The configuration files are located in [`configs/method`](./configs/method/), and the desired method can be specified via `method={method_name}` (see Step 2 of [How to run](#how-to-run)). Annotations are as follows:

| Method | Description | Args Note |
| :-----: | :-----: | :----: |
| `large_model` | Infers a single model in an autoregressive manner | - |
| `cd` | Contrastive decoding with two models | Requires specifying the amateur model (`method.amateur_model`) and contrastive decoding hyperparameter (`method.alpha`) |
| `cd_sd` | Accelerates contrastive decoding directly using speculative decoding | Inherits from `cd`, with additional speculative decoding hyperparameter `method.gamma` |
| `cd_chef` | Accelerates contrastive decoding via speculative ensemble | Inherits from `cd_sd` |
| `ensemble_*` | Integrates models using `method.extra_model` (`str` or `list of str`) | Similar usage to `cd_*`. Modify `llm.ensemble_fn` and `llm.ensemble_target` in the YAML file to adjust the integration approach and objective. |

## Citations

```bib
@article{fu2025speculative,
  title={Speculative Ensemble: Fast Large Language Model Ensemble via Speculation},
  author={Fu, Jiale and Jiang, Yuchu and Chen, Junkai and Fan, Jiaming and Geng, Xin and Yang, Xu},
  journal={arXiv preprint arXiv:2502.01662},
  year={2025}
}
```