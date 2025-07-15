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
from .models import EnsembleWrapper
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

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


EPS = 1e-8


def print_fsdp_param_info(model):
    print(f"{'Module':<40} {'# Params':>12} {'Memory (MB)':>15}")
    print("=" * 70)

    for name, module in model.named_modules():
        if isinstance(module, FSDP):
            total_params = sum(p.numel() for p in module.parameters())
            total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            print(f"{name:<40} {total_params:>12,} {total_bytes / (1024 ** 2):>15.2f}")

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
        self.accelerator.print("="*10+"Accelerator Preparing Models"+"="*10)
        self.policy, self.reference_model, self.train_iterator, self.eval_iterator, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy,
            self.reference_model,
            self.train_iterator, 
            self.eval_iterator, 
            self.optimizer, 
            self.scheduler
        )
        self.accelerator.print("="*10+"Accelerator Finish Preparing"+"="*10)

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

                # # EVALUATION
                # if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                #     results = self.eval()

                #     if self.example_counter > 0:
                #         if self.config.debug:
                #             self.accelerator.print('skipping save in debug mode')
                #         elif self.config.intermediate_checkpoints:
                #             output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                #             self.accelerator.print(f'creating checkpoint to write to {output_dir}...')
                #             self.save(output_dir, results['results'], final_save=False)

                #     self.accelerator.print(results['results'])
                #     delete_dicts(results)

                # TRAINING
                self.policy.train()
                accumulated = 0
                start_time = time.time()

                # print_fsdp_param_info(self.policy)
                
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
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.accelerator.print(f"Learning Rate: {current_lr}")
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
        # optimizer = self.accelerator.unwrap_model(self.optimizer)
        # scheduler = self.accelerator.unwrap_model(self.scheduler)
        # if self.accelerator.is_main_process:
        #     optimizer_state = {
        #         'state_dict': optimizer.state_dict(),
        #         'class': optimizer.__class__.__name__,
        #     }
        #     torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
        #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
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

        self.accelerator.print(output_dir)

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
                use_cache=False
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


class PPOTrainer(BasicTrainer):
    policy_hf_model_class = AutoModelForCausalLMWithValueHead
    use_reference_model = True
            
    def prepare_accelerator(self):
        """Prepare the Accelerator."""
        self.policy.pretrained_model, self.policy.v_head, self.reference_model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.pretrained_model,
            self.policy.v_head,
            self.reference_model,
            self.optimizer, 
            self.scheduler
        )

        if self.reward_model:
            self.reward_model = self.accelerator.prepare(self.reward_model)

    def forward(self, model: AutoModelForCausalLMWithValueHead, batch: Dict[str, Union[List, torch.LongTensor]], is_policy: bool=True, use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.

        Args:
            model: model to run forward pass on
            batch: input batch (forward pass will be run on keys with prefix 'chosen')
            masks: binary-valued tensor shape (batch size, sequence length)
            is_policy: whether the model is the policy or reference
            use_cache: if true, expecte to get cached logprobs from the model

        Returns: 
            all_logps: batch log probabilities at the token level of shape (batch size, sequence length)
            all_logits: corresponding logits of shape (batch size, sequence length)
            all_values: values predicted for each token, of shape (batch size, sequence length)
        """
        if is_policy:
            # here the prefix 'chosen' is a misnomer, since it can refer to the dispreferred generations
            # the 'status' field contains the actual status of the generations
            all_logits, _, all_values = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'])
            all_values = all_values[:, :-1].contiguous()
        else: # if reference
            if use_cache:
                all_logps = model(batch['target_combined_input_ids']).to(self.policy_dtype).to(self.accelerator.device)
            else:
                all_logits = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask']).logits.to(self.policy_dtype)
                all_values = None

        all_logps = self.get_batch_logps(all_logits.to(self.policy_dtype), batch['target_labels'])
        # Returned tensors will have sequence length that is one less than the inputs (to account for label shifting).
        all_logits = all_logits[:, :-1].contiguous()
        all_logps = all_logps.contiguous()

        return all_logps, all_logits, all_values

    def get_reward_scores(self, batch: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        """Get reward scores either from reward model or binary labels.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            torch.FloatTensor of shape (microbatch_size,) containing reward scores
        """
        if self.reward_model is not None:
            # Decode the sequences using policy tokenizer
            sequences = self.tokenizer.batch_decode(batch['target_combined_input_ids'], skip_special_tokens=True)
            # Encode with reward model tokenizer
            reward_inputs = self.reward_tokenizer(
                sequences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.model.max_length
            ).to(self.accelerator.device)

            with torch.no_grad():
                # Get reward model scores
                outputs = self.reward_model(reward_inputs['input_ids'], attention_mask=reward_inputs['attention_mask'])
                # Use the positive class logit as the reward score
                reward_scores = outputs.logits[:, 1]
        else:
            # Use binary labels (1 for chosen, -1 for rejected)
            reward_scores = torch.tensor([(1 if batch['status'][i] == 'chosen' else -1) for i in range(len(batch['status']))])

        return reward_scores
    
    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Estimate the advantages and rewards for every token taken.

        Args:
            values: the estimated values of the tokens. Should already be detached from graph.
            rewards: signal from the environment as to whether the generation is good or bad.
                In the basic implementation, this is only one nonzero reward, on the last unpadded token of each sequence.
                torch tensor of shape (batch size, sequence length)
            masks: torch tensor of shape (batch size, sequence length); 1 if token should be considered and 0 otherwise

        Returns:
            advantages: torch tensor of shape (batch size, sequence length)
            returns: Also called 'rewards-to-go'.
                Only tokens after the current token are used to calculate this: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
                torch tensor of shape (batch size, sequence length)
        """
        values = values * masks
        rewards = rewards * masks
        gae = 0 # generalized advantage estimation
        seq_len = rewards.shape[-1]
        advantages_reversed = []
        
        discounted_future_reward = torch.zeros_like(rewards[:,0])
        discounted_future_rewards_reversed = []

        for t in reversed(range(seq_len)):
            # see https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
            delta = rewards[:, t] + self.config.loss.gamma * (values[:, t + 1] if t < seq_len - 1 else 0.0) - values[:, t]
            gae = delta + self.config.loss.gamma * self.config.loss.lam * gae
            advantages_reversed.append(gae)
            
            discounted_future_rewards_reversed.append(discounted_future_reward)
            discounted_future_reward = rewards[:, t] + self.config.loss.gamma * discounted_future_reward

        advantages = (torch.stack(advantages_reversed[::-1]).transpose(0, 1) * masks)
        returns = (advantages + values).contiguous()
        discounted_future_rewards = (torch.stack(discounted_future_rewards_reversed[::-1]).transpose(0, 1) * masks).contiguous()

        # normalizing advantages leads to more stable learning
        mean_adv, var_adv = masked_mean(advantages, masks), masked_var(advantages, masks)
        normalized_advantages = (advantages - mean_adv) * torch.rsqrt(var_adv + 1e-8)
        normalized_advantages = (normalized_advantages * masks).detach().contiguous()

        return normalized_advantages, returns, discounted_future_rewards

    def loss(self, batch: Dict, episode: Dict) -> Tuple[torch.FloatTensor, Dict]:
        """
        Given the batch statistics and the current episode's values, calculate the loss and return some loss statistics.

        Args:
            batch: dictionary containing batch data (shoud have keys 'values', 'returns', 'advantages', 'logprobs', 'masks')
            episode: dictionary containing the episode data (should have keys 'logits', 'values', 'logprobs')

        Returns:
            loss: combined policy and critic loss of shape (1,)
            loss_stats: dictionary of episode/batch statistics
        """
        value_losses = (episode['values'] - batch['discounted_future_rewards'].detach()) ** 2
        critic_loss = 0.5 * masked_mean(value_losses, batch['masks'])
        
        ratio = torch.exp(episode['logprobs'] - batch['logprobs'])
        policy_losses = -batch['advantages'] * ratio
        policy_losses_clipped = -batch['advantages'] * torch.clamp(ratio, self.config.loss.cliprange, 1 / self.config.loss.cliprange)
        policy_loss = masked_mean(torch.max(policy_losses, policy_losses_clipped), batch['masks'])

        KL_penalty = masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks'])

        loss = policy_loss + self.config.loss.critic_coef * critic_loss + self.config.loss.KL_coef * KL_penalty

        loss_stats = {
            'loss/total': loss.detach(),
            'loss/critic': critic_loss.detach(),
            'loss/policy': policy_loss.detach(),
            'clipfrac/policy': masked_mean(torch.gt(policy_losses_clipped, policy_losses).float(), batch['masks']).detach(),
            'loss/entropy': entropy_from_logits(episode['logits'], batch['masks']).detach(),
            'loss/policykl': masked_mean(batch['logprobs'] - episode['logprobs'], batch['masks']).detach(),
            'loss/seqratio': rowwise_product(ratio, batch['masks']).mean().detach(),
        }

        return loss, loss_stats
    
    def get_global_batch_dict(self, batch):
        """
        Get the processed dict for the entire batch.

        Args:
            batch: dictionary containing batch data (shoud have keys 'values', 'returns', 'advantages', 'logprobs', 'masks')

        Returns:
            global_batch_dict: dictionary containing processed batch data
        """
        batch_size = len(batch['prompt_text'])
        batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            masks = (batch['target_labels'][:, 1:] != -100).clone().to(self.policy_dtype)
            logprobs, _, _ = self.forward(self.reference_model, batch, is_policy=False)
            _, _, values = self.forward(self.policy, batch)
            # Get reward scores from either reward model or binary labels
            scores = self.get_reward_scores(batch)
            rewards = torch.zeros_like(masks) 
            for row in range(rewards.shape[0]):
                rewards[row, masks[row].nonzero()[-1]] += scores[row]

            rewards = rewards * masks
            advantages, returns, discounted_future_rewards = self.compute_advantages(values, rewards, masks)
            
        global_batch_dict = {
            "target_combined_input_ids": batch['target_combined_input_ids'],
            "target_labels": batch['target_labels'],
            "target_combined_attention_mask": batch['target_combined_attention_mask'],
            "logprobs": logprobs,
            "rewards": scores,
            "values": values,
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            "discounted_future_rewards": discounted_future_rewards,
        }
        global_batch_dict = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in global_batch_dict.items()}

        return global_batch_dict

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

        for eval_batch in (tqdm(self.eval_iterator, desc='Computing eval metrics') if self.accelerator.is_main_process else self.eval_iterator):
            eval_batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            global_batch_dict = self.get_global_batch_dict(eval_batch)

            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(global_batch_dict, mode='eval')

            delete_dicts(eval_batch)

        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)

        if self.accelerator.is_main_process and self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        else:
            results = None

        results = {
            'metadata': OmegaConf.to_container(self.config),
            'results': formatted_dict(mean_eval_metrics),
        }

        delete_dicts(eval_metrics, mean_eval_metrics)
        self.accelerator.free_memory()
        torch.cuda.empty_cache()
        
        return results

    def train(self):
        """Train with PPO."""
        self.accelerator.print(f'Using {self.config.optimizer} optimizer with learning rate {self.config.lr}')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        self.policy.train()
        self.reference_model.eval()
        
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
            start_time = time.time()

            microbatch_size = len(batch['prompt_text'])
            global_batch_dict = self.get_global_batch_dict(batch)

            for ppo_epoch in range(self.config.loss.ppo_epochs):
                with self.accelerator.accumulate(self.policy):
                    loss, local_batch_metrics = self.get_batch_metrics(global_batch_dict, microbatch_size, mode='train')

                    for k, v in local_batch_metrics.items():
                        batch_metrics[k].extend(v)

                    self.accelerator.backward(loss)
                    v_head_norm = self.accelerator.clip_grad_norm_(self.policy.pretrained_model.parameters(), self.config.model.max_grad_norm)
                    pretrained_norm = self.accelerator.clip_grad_norm_(self.policy.v_head.parameters(), self.config.model.v_head_max_grad_norm)
                    batch_metrics['grad_norm'].extend(torch.as_tensor(v_head_norm + pretrained_norm).reshape(-1).float().cpu().numpy().tolist())
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            self.batch_counter += 1
            self.example_counter += microbatch_size * self.accelerator.num_processes

            step_time = time.time() - start_time
            examples_per_second = (microbatch_size * self.accelerator.num_processes) / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)

            delete_dicts(global_batch_dict, batch, local_batch_metrics)

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

                delete_dicts(batch_metrics, mean_train_metrics)
                batch_metrics = defaultdict(list)    
            else:
                self.accelerator.print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

    def get_batch_metrics(self, global_batch_dict: Dict, microbatch_size: int=0, mode:str='train'):
        """
        Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the loss.
        """
        # for train
        if microbatch_size:
            indices = torch.randperm(microbatch_size).tolist()
            shuffled_global_batch = {k: v[indices] if isinstance(v, torch.Tensor) else [v[i] for i in indices] for k, v in global_batch_dict.items()}
        # for eval
        else:
            shuffled_global_batch = global_batch_dict

        episode_logprobs, episode_logits, episode_values = self.forward(self.policy, shuffled_global_batch)
        episode = {
            'logprobs': episode_logprobs,
            'logits': episode_logits,
            'values': episode_values,
        }
        loss, metrics = self.loss(shuffled_global_batch, episode)

        metrics['rewards'] = shuffled_global_batch['rewards'].detach()
        metrics['returns/mean'] = masked_mean(shuffled_global_batch['returns'], shuffled_global_batch['masks']).detach()
        metrics['returns/var'] = masked_var(shuffled_global_batch['returns'], shuffled_global_batch['masks']).detach()
        metrics['val/mean'] = masked_mean(shuffled_global_batch['values'], shuffled_global_batch['masks']).detach()
        metrics['val/var'] = masked_var(shuffled_global_batch['values'], shuffled_global_batch['masks']).detach()

        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = self.accelerator.gather(v).flatten()
            batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())

        delete_dicts(metrics, episode, global_batch_dict, shuffled_global_batch)
        del episode_logprobs, episode_logits, episode_values

        return loss, batch_metrics

    def save(self, output_dir=None, metrics={}, final_save=True):
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

        if self.config.model.use_peft and final_save:
            state_dict = get_base_model_state_dict_from_peft(
                self.accelerator.get_state_dict(self.policy.pretrained_model),
                self.config.model.peft.lora_alpha,
                self.config.model.peft.lora_r,
            )
            unwrapped_model = self.accelerator.unwrap_model(self.policy.pretrained_model).base_model
        else:
            state_dict = self.accelerator.get_state_dict(self.policy.pretrained_model)
            unwrapped_model = self.accelerator.unwrap_model(self.policy.pretrained_model)

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
            
        self.accelerator.wait_for_everyone()

        unwrapped_v_head = self.accelerator.unwrap_model(self.policy.v_head)
        torch.save(unwrapped_v_head.state_dict(), os.path.join(output_dir, "v_head.pt"))
        self.accelerator.wait_for_everyone()


class ReinforceTrainer:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 reward_fn, 
                 optimizer,
                 scheduler,
                 train_iterator, 
                 eval_iterator, 
                 config,
                 seed=42,
                 reg_scaler=0.5,
                 log_ensemble_weight=True,
                 gamma=1.0
                ):

        self.seed=seed

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.batch_counter=0
        self.example_counter=0

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.reward_fn = reward_fn
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.reg_scaler = reg_scaler
        self.gamma = gamma
        self.config = config
        self.log_ensemble_weights=log_ensemble_weight


    def _get_batch_metric(self, batch):

        input_ids = batch['prompt_input_ids'].to(self.model.device)
        attn_mask = batch['prompt_attention_mask']
        target = [answer[0]['content'] for answer in batch['target']]

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.config.model.max_tokens,
                do_sample=False,
                use_cache=True,
                temperature=0.0,
                num_beams=1,
            )
            if isinstance(self.model, EnsembleWrapper):
                self.model.reset_cache()

        reply_ids = output_ids[:, input_ids.shape[-1]:]
        generations = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        rewards = torch.Tensor(self.reward_fn(generations, target))


        # 3. Get logits from model to compute log-probs
        self.model.train()
        input_ids = output_ids[:, :-1]
        target_ids = output_ids[:, 1:]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()


        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, log_ensemble_weights=False)
        if isinstance(self.model, EnsembleWrapper):
            self.model.reset_cache()

        # 1. Slice out generation logits and target ids
        gen_start = batch['prompt_input_ids'].shape[-1] - 1
        logits_gen = outputs.logits[:, gen_start:, :]         # [B, T_gen, V]
        target_ids_gen = target_ids[:, gen_start:]            # [B, T_gen]
        w_draft = outputs.w_draft[:, gen_start:, :]
        w_target = outputs.w_target[:, gen_start:, :]

        # 2. Create attention mask to ignore pad tokens
        gen_attention_mask = (target_ids_gen != self.tokenizer.pad_token_id).float()  # [B, T_gen]
        
        # 3. Compute token-level log probs
        log_probs = F.log_softmax(logits_gen, dim=-1)                                   # [B, T_gen, V]
        token_log_probs = torch.gather(log_probs, 2, target_ids_gen.unsqueeze(-1)).squeeze(-1)  # [B, T_gen]
        
        # 4. Apply padding mask and sum
        masked_log_probs = token_log_probs * gen_attention_mask                        # [B, T_gen]
        sequence_log_probs = masked_log_probs.sum(dim=1)                               # [B]

        # 5. Compute the regularization

        if isinstance(self.model, EnsembleWrapper) and self.log_ensemble_weights:
            w_draft_mean = (w_draft.detach().squeeze(-1) * gen_attention_mask).sum() / gen_attention_mask.sum()
            w_target_mean = (w_target.detach().squeeze(-1) * gen_attention_mask).sum() / gen_attention_mask.sum()
            
            print({
                "Draft_weight": w_draft_mean.item(),
                "Target_weight": w_target_mean.item()
            })
            wandb.log({
                "Draft_weight": w_draft_mean.item(),
                "Target_weight": w_target_mean.item()
            }, commit=False)
      
        w = torch.cat([w_draft, w_target], dim=-1)  # [B, T_gen, 2]
        entropy = - (w + EPS) * torch.log(w + EPS)  # [B, T_gen, 2]
        entropy = entropy.sum(dim=-1)               # [B, T_gen]
        
        # 5.3 Mask out padding
        entropy = entropy * gen_attention_mask      # [B, T_gen]
        
        # 5.4 Sum and average over batch
        entropy_reg = entropy.mean()     # scalar

        print("Entropy Mean: ", entropy_reg)

        
        # 6. Compute REINFORCE loss
        rewards = torch.tensor(rewards, dtype=torch.float32, device=sequence_log_probs.device)
        rewards = rewards-1
        loss = -(sequence_log_probs * rewards).mean()-self.config.model.reg_scale*entropy_reg

        # Logging
        metric = {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "log_prob_mean": sequence_log_probs.mean().item(),
        }

        return loss, metric

        

    def train(self):
        grad_accum_steps = self.config.model.gradient_accumulation_steps

        for epoch in range(self.config.global_epochs):

            last_log = None
            batch_metrics = defaultdict(list)

            for batch_idx, batch in enumerate(self.train_iterator):
                
                start_time = time.time()

                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, metrics = self._get_batch_metric(batch)
                
                loss = loss / grad_accum_steps
                loss.backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(torch.as_tensor(v).reshape(-1).float().cpu().numpy().tolist())


                if (batch_idx+1) % grad_accum_steps == 0:

                    grad_norm=nn.utils.clip_grad_norm_(self.model.parameters(), self.config.model.max_grad_norm)
                    batch_metrics['grad_norm'].extend(torch.as_tensor(grad_norm).reshape(-1).float().cpu().numpy().tolist())
                    self.optimizer.step()
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Learning Rate: {current_lr}")
                    self.optimizer.zero_grad()
    
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
                        print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')
    
                        if self.config.wandb.enabled:
                            wandb.log(mean_train_metrics, step=self.example_counter)
    
                        last_log = time.time()
                        batch_metrics = defaultdict(list)
                    else:
                        print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

                    delete_dicts(batch_metrics, mean_train_metrics)
                    

                if (batch_idx+1) % self.config.model.save_freqs == 0:
                    self.save(
                            os.path.join(self.config.local_run_dir, str(batch_idx+1)), 
                            metrics={'counter': self.example_counter}
                        )

                delete_dicts(batch, metrics)


    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = {}, final_save=True):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            metrics['counter'] = self.example_counter
            json.dump(metrics, f)
        
        print(f"Saving model...")
        print(output_dir)

        self.model.save_pretrained(
            output_dir,
        )
                

                

        