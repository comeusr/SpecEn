import argparse
import json
import sys
import re
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DistributedDataParallelKwargs
from .dataloader import SFTDataLoader
from .utils import set_offline_if_needed
from .models import EnsembleWrapper
import torch
import evaluate
import sacrebleu

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

def cnndm_find_answer(text):
    return re.split("\n\nArticle:", text)[0].strip()

def xsum_find_answer(text):
    return re.split("\n\nDocument:", text)[0].strip()

def wmt_find_answer(text):
    return re.split("\n\nEnglish:", text)[0].strip()

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


def wmt_reward_func(completions, ground_truth):
    contents = [wmt_find_answer(complete) for complete in completions]
    reward=[]
    for i, (content, gt) in enumerate(zip(contents, ground_truth)):
        reward.append(sacrebleu.sentence_bleu(content, [gt]))
    return reward



def main(args):

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device0)

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device1)

    print("Target Model Device: ", target_model.device)
    print("Draft Model Device: ", draft_model.device)

    if args.method == "static":
        static_draft_weights = args.static_draft_weights
    else:
        static_draft_weights = None

    do_sample = (args.do_sample=="True")

    model = EnsembleWrapper(target_model, draft_model, True, static_draft_weights=static_draft_weights)
    if args.method == 'dynamic':
        print(f"Loading Ensemblemodel and tokenizer from {args.model_path}")
        model.load_ensemble_head(args.model_path)
    # model = target_model - this only for testing purposes. The actual naive speculative decoding is happening in speculative_decoding.py

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    tokenizer.chat_template = open('train_config/template.jinja').read() #kasasiva changed to gemma for gemma models
    # tokenizer.chat_template = open('train_config/template_gemma.jinja').read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_iterator_kwargs = dict(
        process_index=0,
        num_processes=1,
        max_length=1350, #changed by kasasiva from 640 to 1350
        max_prompt_length=1200,#changed by kasasiva from 384 to 1200
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
    output_path = os.path.join(args.model_path, "{}_{}_generations.json".format(args.method, args.static_draft_weights))
    metrics_path = os.path.join(args.model_path, "{}_{}_metrics.json".format(args.method, args.static_draft_weights))
    
    all_completions, all_labels = [], []
    all_results = []
    all_bleu = []

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
                do_sample=do_sample,
                use_cache=True,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                num_beams=1,
            )
    
        generations = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        all_completions.extend(generations)
        all_labels.extend(labels)

        if args.dataset == 'cnndm':
            rouge = evaluate.load('rouge')
            

        for p, g, t in zip(prompts, generations, labels):
            if args.dataset == "gsm8k":
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
            elif args.dataset == 'wmt':
                g = wmt_find_answer(g)
                metric = sacrebleu.sentence_bleu(g, [t], tokenize="13a", lowercase=True).score
                all_bleu.append(metric)
                all_results.append({
                    "prompt": p[0],
                    "generation": g,
                    "reference": t,
                    "metric": metric,
                })

            elif args.dataset == 'cnndm':
                g = cnndm_find_answer(g)
                metric = rouge.compute(predictions=[g], references=[t])
                all_results.append({
                    "prompt": p[0],
                    "generation": g,
                    "reference": t,
                    "metric": metric,
                })
            elif args.dataset == 'xsum':
                g = xsum_find_answer(g)
                metric = rouge.compute(predictions=[g], references=[t])
                all_results.append({
                    "prompt": p[0],
                    "generation": g,
                    "reference": t,
                    "metric": metric,
                })


    # Dump all results at once as a JSON array
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if args.dataset == 'gsm8k':
        acc = sum(reward_func(all_completions, all_labels)) / len(all_labels)
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f, indent=2)
    elif args.dataset == "cnndm":

        metrics = rouge.compute(predictions=all_completions, references=all_labels)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    elif args.dataset == 'wmt':
        # bleu = sacrebleu.corpus_bleu(all_completions, [all_labels], tokenize="13a", lowercase=True).score
        bleu = np.array(all_bleu).mean()

        print("THE FINAL BLUE SCORE", bleu)
        with open(metrics_path, "w") as f:
            json.dump(bleu, f, indent=2)


    print(f"Saved generations to {output_path}")
    # print(f"Saved accuracy: {acc:.4f} to {metrics_path}")

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
    parser.add_argument("--method", type=str, choices=['static', 'dynamic'])
    parser.add_argument("--do_sample", type=str, choices=['True', "False"])
    parser.add_argument("--static_draft_weights", type=float, help="Useful only when method is static.")
    parser.add_argument("--draft_model", type=str)
    parser.add_argument("--target_model", type=str)
  
    args = parser.parse_args()
    main(args)
        
        

    