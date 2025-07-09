import os
from omegaconf import OmegaConf
from omegaconf import DictConfig
from dotenv import load_dotenv
from typing import Optional, Set

import torch

import wandb
import hydra
import json
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from train import trainers
from train import dataloader
from train.utils import disable_dropout
from train.trainers import SFTTrainer
from train.models import EnsembleWrapper


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


@hydra.main(version_base=None, config_path="train_config", config_name="config")
def main(config: DictConfig):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=config.local_run_dir,
        gradient_accumulation_steps=config.model.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    # Calculate microbatch sizes
    if config.model.batch_size % accelerator.num_processes == 0:
        config.model.microbatch_size = config.model.batch_size / accelerator.num_processes
    else:
        raise ValueError(f"{config.model.batch_size} needs to be divisible by the number of processes")

    if config.model.eval_batch_size % accelerator.num_processes == 0:
        config.model.eval_microbatch_size = config.model.eval_batch_size / accelerator.num_processes
    else:
        raise ValueError(f"{config.model.eval_batch_size} needs to be divisible by the number of processes")

    if config.eval_every % config.model.batch_size != 0:
        accelerator.print('WARNING: eval_every must be divisible by batch_size')
        accelerator.print('Setting eval_every to', config.eval_every - config.eval_every % config.model.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.model.batch_size

    accelerator.print(OmegaConf.to_yaml(config))

    if accelerator.is_main_process:
        os.makedirs(config.local_run_dir, exist_ok=True)
        accelerator.print("Making experiment directory", config.local_run_dir)

        if config.wandb.enabled:
            os.environ['WANDB_CACHE_DIR'] = config.cache_dir
            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                config=OmegaConf.to_container(config),
                dir=config.cache_dir,
                name=config.exp_name,
            )
        
        config_path = os.path.join(config.local_run_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        accelerator.print('=' * 80)
        accelerator.print(f'Writing to {config.local_run_dir}')
        accelerator.print('=' * 80)

   # Prepare tokenizer
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    accelerator.print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens = []
    # Check if the tokenizer has a chat template and set a default one if it doesn't
    if not tokenizer.chat_template:
        with open("config/template.jinja") as f:
            tokenizer.chat_template = f.read()

        print("Default chat template set.")

    control_tokens = list(config.loss.get("control_tokens", {}).values())
    special_tokens.extend(control_tokens)

    num_tokens_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Create data loaders
    accelerator.print(f'Loading data')
    data_loader_class = getattr(dataloader, config.loss.dataloader)
    data_iterator_kwargs = dict(
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        seed=config.seed,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        control_tokens=config.loss.get("control_tokens", {}),
    )
    
    train_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='train',
        microbatch_size=config.model.microbatch_size,
        n_epochs=config.n_epochs,
        n_examples=config.n_examples,
        **data_iterator_kwargs
    )
    
    eval_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='test',
        microbatch_size=config.model.eval_microbatch_size,
        n_examples=config.n_eval_examples, 
        n_epochs=(1 if config.n_eval_examples is None else None),
        **data_iterator_kwargs
    )
        
    # 4. Load custom model
    target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", 
                                                  torch_dtype=torch.bfloat16, 
                                                  trust_remote_code=True, 
                                                  attn_implementation="flash_attention_2")

    draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", 
                                                  torch_dtype=torch.bfloat16, 
                                                  trust_remote_code=True, 
                                                  attn_implementation="flash_attention_2")


    policy = EnsembleWrapper(target_model, draft_model, config)

    policy.load_ensemble_head("/home/sagemaker-user/data/model/Qwen3-8B_Qwen-06B_sft_5e-4/FINAL")
    
    for name, param in policy.target_model.named_parameters():
        param.requires_grad=False

    for name, param in policy.draft_model.named_parameters():
        param.requires_grad=False

    accelerator.print("Creating optimizer and scheduler")

    num_params = count_trainable_parameters(policy)
    accelerator.print(f"Trainable parameters: {num_params:,}")
    
    optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=train_iterator.num_training_steps - config.warmup_steps, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_steps])

    if config.model.from_checkpoint:
        optimizer_state = optimizer.state_dict()
        optimizer_state.update(torch.load(os.path.join(config.model.from_checkpoint, "optimizer.pt"), map_location='cpu'))
        optimizer.load_state_dict(optimizer_state)

        scheduler_state = torch.load(os.path.join(config.model.from_checkpoint, "scheduler.pt"))
        scheduler.load_state_dict(scheduler_state)

        metrics = json.load(open(os.path.join(config.model.from_checkpoint, 'metrics.json')))
        num_skip_batches = int(metrics.get('counter', 0) / config.model.batch_size)
    else:
        num_skip_batches = 0

    accelerator.print("Initial Trainer")
    trainer = SFTTrainer(
        tokenizer, 
        config, 
        train_iterator, 
        eval_iterator,
        accelerator, 
        optimizer,
        scheduler,
        policy, 
        reference_model=None,
        num_skip_batches=num_skip_batches,
    )
    accelerator.print("Initialed Trainer")
    
    # 7. Train
    trainer.train()
    
    # 8. Save
    trainer.save(
        os.path.join(config.local_run_dir, 'FINAL'), 
        metrics={'counter': trainer.example_counter}
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    main()

    