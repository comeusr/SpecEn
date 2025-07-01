import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import gc
from .models import AutoModelForCausalLM, AutoModelForCausalLMWithValueHead
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
from accelerate import Accelerator

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

from . import dataloader
from .utils import (
    formatted_dict,
    pad_to_length,
    masked_mean,
    masked_var,
    entropy_from_logits,
    delete_dicts,
    rowwise_product,
    get_base_model_state_dict_from_peft
)
import numpy as np
import wandb
from tqdm import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

from torch.autograd import Variable # Needed for custom backprop
torch.autograd.set_detect_anomaly(True)
import math

class BasicTrainer(object):
    policy_hf_model_class = AutoModelForCausalLM
    reference_hf_model_class = AutoModelForCausalLM
    use_reference_model = True

    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 config: DictConfig, 
                 train_iterator: dataloader.DataLoader, 
                 eval_iterator: dataloader.DataLoader, 
                 accelerator: Accelerator,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 policy: nn.Module, 
                 reference_model: Optional[nn.Module] = None,
                 num_skip_batches=0):
        """A trainer for a language model, supporting either SFT, HALO, or offline PPO training."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.accelerator = accelerator
        
        self.config = config
        self.run_dir = config.local_run_dir

        self.tokenizer = tokenizer
        self.example_counter = 0
        self.batch_counter = 0

        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)

        self.reference_model = reference_model
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_skip_batches = num_skip_batches # when loading from checkpoint
        self.prepare_accelerator()

    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy, self.reference_model, self.train_iterator, self.eval_iterator, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy,
            self.reference_model,
            self.train_iterator, 
            self.eval_iterator, 
            self.optimizer, 
            self.scheduler
        )

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        distribution_logps = logits.float().log_softmax(-1)
        per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps * loss_mask
        
    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy."""
        with self.accelerator.autocast():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.model.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
            )
        
        policy_output = pad_to_length(policy_output, self.config.model.max_length, self.tokenizer.pad_token_id)
        policy_output = self.accelerator.gather(policy_output)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'

        Returns:
            A tuple of a scalar loss and a dict of metrics.
        """
        raise NotImplementedError

    def eval(self) -> Dict[str, Dict]:
        """
        Run evaluation on all the examples in the test data and return the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset. 
        It does not compare two models to eacch other.

        Returns:
            A dict of form:
            {
                'metadata': the Hydra config
                'results': a dict of batch metrics (averaged across all of the test data)
            }
        """
        self.accelerator.print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        if self.reference_model is not None:
            self.reference_model.eval()

        all_eval_metrics = defaultdict(list)
    
        # Wrap the eval_iterator with accelerator.prepare
        eval_dataloader = self.accelerator.prepare(self.eval_iterator)

        for eval_batch in (tqdm(eval_dataloader, desc='Computing eval metrics') if self.accelerator.is_main_process else eval_dataloader):
            eval_batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(eval_batch, mode='eval')

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        delete_dicts(eval_batch, eval_metrics, all_eval_metrics)
        self.free_memory()

        if self.accelerator.is_main_process and self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        else:
            results = None

        results = {
            'metadata': OmegaConf.to_container(self.config),
            'results': formatted_dict(mean_eval_metrics),
        }
        
        return results

    def train(self):

        self.accelerator.print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')
        
        if self.reference_model is not None:
            self.reference_model.eval()
        
        for epoch in range(self.config.global_epochs):
            self.accelerator.print(f'====== Running epoch {epoch} =========')

            last_log = None
            batch_metrics = defaultdict(list)

            for batch in self.train_iterator:
                if self.batch_counter < self.num_skip_batches:
                    self.batch_counter += 1
                    self.example_counter += self.config.model.batch_size
                    continue

                # EVALUATION
                if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                    results = self.eval()

                    if self.example_counter > 0:
                        if self.config.debug:
                            self.accelerator.print('skipping save in debug mode')
                        elif self.config.intermediate_checkpoints:
                            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                            self.accelerator.print(f'creating checkpoint to write to {output_dir}...')
                            self.save(output_dir, results['results'], final_save=False)

                    self.accelerator.print(results['results'])
                    delete_dicts(results)

                # TRAINING
                self.policy.train()
                accumulated = 0
                start_time = time.time()
                
                with self.accelerator.accumulate(self.policy):
                    batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    loss, metrics = self.get_batch_metrics(batch)
                    self.accelerator.backward(loss)

                    for k, v in metrics.items():
                        batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

                    grad_norm = self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.model.max_grad_norm)
                    batch_metrics['grad_norm'].extend(torch.as_tensor(grad_norm).reshape(-1).float().cpu().numpy().tolist())
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated += 1

                step_time = time.time() - start_time
                examples_per_second = self.config.model.batch_size / step_time
                batch_metrics['examples_per_second'].append(examples_per_second)
                
                self.batch_counter += 1
                self.example_counter += self.config.model.batch_size

                if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                    mean_train_metrics = {}
                    for k, v in batch_metrics.items():
                        if len(v) > 0:
                            mean_train_metrics[k] = sum(v) / len(v)

                    mean_train_metrics['counters/examples'] = self.example_counter
                    mean_train_metrics['counters/updates'] = self.batch_counter
                    self.accelerator.print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                    if self.config.wandb.enabled and self.accelerator.is_main_process:
                        wandb.log(mean_train_metrics, step=self.example_counter)

                    last_log = time.time()
                    batch_metrics = defaultdict(list)
                else:
                    self.accelerator.print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

                delete_dicts(batch, metrics, batch_metrics, mean_train_metrics)

                if accumulated >= self.config.model.gradient_accumulation_steps:
                    self.free_memory()
                    accumulated = 0

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = {}, final_save=True):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        self.accelerator.print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            self.accelerator.print(f"Saving tokenizer...")
            self.tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                metrics['counter'] = self.example_counter
                json.dump(metrics, f)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving state...")
        optimizer = self.accelerator.unwrap_model(self.optimizer)
        scheduler = self.accelerator.unwrap_model(self.scheduler)
        if self.accelerator.is_main_process:
            optimizer_state = {
                'state_dict': optimizer.state_dict(),
                'class': optimizer.__class__.__name__,
            }
            torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Saving model...")

        if False:#self.config.model.use_peft and final_save:
            state_dict = get_base_model_state_dict_from_peft(
                self.accelerator.get_state_dict(self.policy),
                self.config.model.peft.lora_alpha,
                self.config.model.peft.lora_r,
            )
            unwrapped_model = self.accelerator.unwrap_model(self.policy).base_model
        else:
            state_dict = self.accelerator.get_state_dict(self.policy)
            unwrapped_model = self.accelerator.unwrap_model(self.policy)

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
            
        self.accelerator.wait_for_everyone()

    def free_memory(self):
        torch.cuda.empty_cache()
        self.accelerator.free_memory()
        gc.collect()


class SFTTrainer(BasicTrainer):
    use_reference_model = False

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        
        with self.accelerator.autocast():
            policy_chosen_logits = self.policy(
                batch['target_combined_input_ids'], 
                attention_mask=batch['target_combined_attention_mask'],
            ).logits.to(self.policy_dtype)
            
            policy_chosen_logps = self.get_batch_logps(policy_chosen_logits, batch['target_labels'])
            policy_chosen_logps = policy_chosen_logps.view(-1)
            losses = -policy_chosen_logps

        # Gather losses and logps from all processes
        total_nonzero_elements = self.accelerator.gather((policy_chosen_logps != 0).sum().detach()).sum()
        metrics[f'logps_{mode}/chosen'] = self.accelerator.gather(policy_chosen_logps.detach()).sum() / total_nonzero_elements
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.sum().detach()).sum() / total_nonzero_elements

        del policy_chosen_logits, policy_chosen_logps

        return losses.sum(), metrics

