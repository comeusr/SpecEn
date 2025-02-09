read -p "Enter the run name (default to be 'test'): " run_name
run_name="${run_name:=test}"
run_dir=results/$run_name/$(date "+%Y-%m-%d/%H-%M-%S")
mkdir -p $run_dir

python -u ./pipeline.py \
    run_name=test \
	visible_devices=[0,1,2,3,5,6,7] \
	\
    methods=[ensemble_sd] \
    +dataset.name=[cnndm,mmlu] \
    +dataset.size=1000 \
    \
    +cd_CHEF.model=llama-3.1-8b-instruct \
    +cd_CHEF.amateur_model=llama-3.2-1b \
    +cd_CHEF.alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] \
    +cd_CHEF.generate.temperature=1 \
    +cd_CHEF.gamma=5 \
    +cd_CHEF.llm.max_model_len=4096 \
    \
    +chef.model=qwen2.5-1.5b-instruct \
    +chef.extra_model=[[qwen2.5-coder-1.5b-instruct,qwen2.5-math-1.5b-instruct]] \
    +chef.generate.temperature=1 \
    +chef.gamma=[[1,1,1]] \
    +chef.llm.max_model_len=4096 \
    \
    +ensemble_sd.model=qwen2.5-1.5b-instruct \
    +ensemble_sd.extra_model=[[qwen2.5-coder-1.5b-instruct,qwen2.5-math-1.5b-instruct]] \
    +ensemble_sd.generate.temperature=1 \
    +ensemble_sd.gamma=5 \
    +ensemble_sd.llm.max_model_len=4096 \
    \
    hydra.run.dir=$run_dir