<h1 align="center">Speculative Ensemble</h1>

<p align="center">
<a href="https://arxiv.org/abs/2502.01662v1">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2502.01662v1-red"></a>
</p>

**Speculative Ensemble** is a novel framework that accelerates the ensemble of multiple LLMs without sacrificing performance. It achieves a performance boost of 1.11x to 2.23x over standard ensemble methods in two- or three-model configurations.

## Setup

```bash
# Create and activate the environment
conda create -n specens python=3.11 -y
conda activate specens

# Install vllm
cd vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/c8/f4/e108a902ccad131d8978a9376343a6e95d78d0e12f152a796794647073ec/vllm-0.6.5-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .

# Install the remaining dependencies
cd ..
pip install -r requirements.txt
```

## How to Run

1. Create a `.env` file to specify the root path of your models.
    ```text
    MODEL_PATH=xxx
    ```

2. Run an example:
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

## Code Reading Guides

**Chef** is the internal name for the speculative ensemble implementation. The code is located in [`vllm/vllm/chef`](./vllm/vllm/chef/), while the baseline ensemble implementation can be found at [`vllm/vllm/ensemble_decode`](./vllm/vllm/ensemble_decode).

We have implemented multiple model inference methods. The configuration files are located in [`configs/method`](./configs/method/), and the desired method can be specified via `method={method_name}` (see Step 2 of [How to run](#how-to-run)). Annotations are as follows:

| Method | Description | Args Note |
| :-----: | :-----: | :----: |
| `large_model` | Inference using a single model in an autoregressive manner | - |
| `cd` | Contrastive decoding with two models | Requires (`method.amateur_model`) and (`method.alpha`) |
| `we` | Weighted ensemble with two models | Requires (`method.extra_model`) and (`method.lambda`) |
| `*_sd` | Accelerates the ensemble directly using speculative decoding | Inherits from specific ensemble methods, with `method.gamma` as an additional hyperparameter |
| `*_chef` | Accelerates the ensemble using speculative ensemble | Inherits from specific ensemble methods. For ensembles with more than two models, `method.gamma` should be a list of integers|

## Customization

### Customizing an Ensemble Method

1. Create a YAML file, such as `configs/method/your_ens_method.yaml`, referencing [`configs/method/we.yaml`](./configs/method/we.yaml).
2. Customize the `method.extra_model` parameter (as a string or a list of strings) and any additional parameters if needed.
3. Modify the `llm.ensemble_fn` and `llm.ensemble_target` to define the ensemble function.
4. Finally, use `method=your_ens_method` to run your custom ensemble method.

### Customizing a Dataset

1. Create a directory `src/mydatasets/your_dataset/` and a file `src/mydatasets/your_dataset/mydataset.py` that follows the template in [`src/mydatasets/template.py`](./src/mydatasets//template.py).
2. Use `dataset=your_dataset` to run your custom dataset.

## Citation

```bib
@article{fu2025speculative,
  title={Speculative Ensemble: Fast Large Language Model Ensemble via Speculation},
  author={Fu, Jiale and Jiang, Yuchu and Chen, Junkai and Fan, Jiaming and Geng, Xin and Yang, Xu},
  journal={arXiv preprint arXiv:2502.01662},
  year={2025}
}
```

## Acknowledgements

We built our implementation upon the [VLLM](https://github.com/vllm-project/vllm) project and would like to thank the authors for their outstanding contributions.