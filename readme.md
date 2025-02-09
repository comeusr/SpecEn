# Setup

```bash
conda create -n vllm_specdec python=3.11 -y
conda activate vllm_specdec

# git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/c8/f4/e108a902ccad131d8978a9376343a6e95d78d0e12f152a796794647073ec/vllm-0.6.5-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .

cd ..
pip install -r requirements.txt
```

# How to use

1. create a .env file like
    ```bash
    MODEL_PATH=xxx
    ```

2. run
    ```bash
    export PYTHONPATH=./vllm:$PYTHONPATH
    ```

3. run a example:
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