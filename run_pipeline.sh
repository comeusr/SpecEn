read -p "Enter the run name (default to be 'test'): " run_name
run_name="${run_name:=test}"
run_dir=results/$run_name/$(date "+%Y-%m-%d/%H-%M-%S")
mkdir -p $run_dir

nohup python -u ./pipeline.py \
    run_name=test \
	visible_devices=[0,1,2,3,4,5,6,7] \
	\
    methods=[cd_chef,cd] \
    +dataset.name=[humaneval,gsm8k] \
    +dataset.size=100 \
    \
    +cd_chef.model=llama-3.1-8b-instruct \
    +cd_chef.amateur_model=llama-3.2-1b \
    +cd_chef.alpha=[0.1,0.2,0.3] \
    +cd_chef.generate.temperature=0 \
    +cd_chef.gamma=5 \
    +cd_chef.llm.max_model_len=4096 \
    \
    +cd.model=llama-3.1-8b-instruct \
    +cd.amateur_model=llama-3.2-1b \
    +cd.alpha=[0.1,0.2,0.3] \
    +cd.generate.temperature=0 \
    +cd.llm.max_model_len=4096 \
    \
    hydra.run.dir=$run_dir > $run_dir/nohup.out 2>&1 &