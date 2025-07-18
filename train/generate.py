import argparse
import json
import sys
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DistributedDataParallelKwargs
from .dataloader import SFTDataLoader
from .utils import set_offline_if_needed
from .models import EnsembleWrapper
import torch

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

def extract_first_answer_block(text):
    split_marker = "Question:"
    if split_marker in text:
        return text.split(split_marker, 1)[0].strip()
    return text.strip()

def find_answer(text):
    match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
    if match:
        return round(float(match.group(1)))
    else:
        all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
        if all_m:
            return round(float(all_m[-1]))
    return "No answer found"

def reward_func(completions, ground_truth, **kwargs):
    contents = [find_answer(c) for c in completions]
    ground_truth = [find_answer(gt) for gt in ground_truth]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def main(args):

    print(f"Loading Ensemblemodel and tokenizer from {args.model_path}")

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    model = EnsembleWrapper(target_model, draft_model, True)
    model.load_ensemble_head(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    tokenizer.chat_template = open('train_config/template.jinja').read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_iterator_kwargs = dict(
        process_index=0,
        num_processes=1,
        max_length=1536,
        max_prompt_length=1152,
        seed=42,
        frac_unique_desirable=1.0,
        frac_unique_undesirable=1.0,
        control_tokens={},
    )

    dataloader = SFTDataLoader(
        [args.dataset], 
        tokenizer,
        split=args.split,
        microbatch_size=args.batch_size,
        n_examples=args.n_examples, 
        n_epochs=1,
        **data_iterator_kwargs
    )

    os.makedirs(args.model_path, exist_ok=True)    
    output_path = os.path.join(args.model_path, "en_generations.json")
    metrics_path = os.path.join(args.model_path, "en_metrics.json")
    
    all_completions, all_labels = [], []
    all_results = []

    for batch in dataloader:
        # print(batch)
        
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # input_ids = batch['prompt_input_ids']
        # attn_mask = batch['prompt_attention_mask']

        input_ids = batch['original_prompt_input_ids']
        attn_mask = batch['original_prompt_attention_mask']
        labels = [answer[0]['content'] for answer in batch['target']]
        prompts = batch['original_prompt']


        if isinstance(model, EnsembleWrapper):
            model.reset_cache()
    
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                use_cache=True,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                num_beams=1,
            )
    
        generations = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        all_completions.extend(generations)
        all_labels.extend(labels)

        for p, g, t in zip(prompts, generations, labels):
            g = extract_first_answer_block(g)
            pred = find_answer(g)
            truth = find_answer(t)
            label = (pred==truth)
            all_results.append({
                "prompt": p[0],
                "generation": g,
                "pred": pred,
                "answer": t,
                "ground_truth": truth,
                "label": label,
                "model_path": args.model_path,
                "seed": args.seed
            })

    # Dump all results at once as a JSON array
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    acc = sum(reward_func(all_completions, all_labels)) / len(all_labels)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)

    print(f"Saved generations to {output_path}")
    print(f"Saved accuracy: {acc:.4f} to {metrics_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("--model_path", type=str, help="Path to the local model folder or the Huggingface repo")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="Path to save the output JSON file")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum length of prompt (in tokens)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samples to generate per input")
    parser.add_argument("--stop_token", type=str, default='<|im_end|>', help="Stop token")
    parser.add_argument("--dataset", type=str, default='gsm8k')
    parser.add_argument("--local_run_dir", type=str, default='.cache/gsm8k/generation')
    parser.add_argument("--n_examples", type=int, default=8)
    parser.add_argument("--draft_model", type=str)
    parser.add_argument("--target_model", type=str)
  
    args = parser.parse_args()
    main(args)
        
        

    