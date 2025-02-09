CUDA_VISIBLE_DEVICES=6 \
python ./main_case.py \
    input="input here" \
    method=cd_CHEF \
    method.model="llama-3.1-8b-instruct" \
    method.amateur_model="llama-3.2-1b" \
    method.generate.temperature=0 \
    method.llm.max_model_len=4096 \
    method.gamma=1 \
    method.alpha=0.3