import argparse
import json
import sys
import re
import traceback
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from train.dataloader import SFTDataLoader
from train.models import EnsembleWrapper, EnsembleHead
import torch
import time
import numpy as np
import evaluate
import sacrebleu

import sys
import os

import transformers, inspect

print("Transformers:", transformers.__version__)
print("Transformers path:", transformers.__file__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_hidden_size(cfg):
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    # Common HF models (Llama, Qwen, etc.)
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size


def cnndm_find_answer(text):
    return re.split("\n\nArticle:", text)[0].strip()

def xsum_find_answer(text):
    return re.split("\n\nDocument:", text)[0].strip()

def wmt_find_answer(text):
    return re.split("\n\nEnglish:", text)[0].strip()

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
    try:
        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        
        # Get local rank for device placement
        local_rank = accelerator.local_process_index
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Process {accelerator.process_index} using device: {device}")
    
        # Set seed for reproducibility
        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        
        do_sample = (args.do_sample=="True")

        # Load target model with device_map="auto" for better memory management
        logger.info(f"Process {accelerator.process_index} loading target model: {args.target_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2", # Modified by Ziyi
            # attn_implementation="flash_attention_2"
            device_map={"": device},  # Map all modules to the specified device
        )

        if args.method == "sd":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2", # Modified by Ziyi
                # attn_implementation="flash_attention_2"
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True

            print("Debuging the num assitant tokens: ", args.num_assistant_tokens)
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
            print("Debuging the num assitant tokens: ", draft_model.generation_config.num_assistant_tokens)


            #wzyi made a change
            if args.assistant_schedule != 'dynamic':
                print("Debuging the assitant schedule: ", args.assistant_schedule)
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            ensemble_head = None
            draft_ensemble_weights = None
        elif args.method == "sd_en":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2", # Modified by Ziyi
                # attn_implementation="flash_attention_2"
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
            

            if args.assistant_schedule != 'dynamic':
                print("Debuging the assitant schedule: ", args.assistant_schedule)
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            target_hidden_size = get_hidden_size(model.config)
            draft_hidden_size = get_hidden_size(draft_model.config)

            ensemble_head = EnsembleHead(target_hidden_size=target_hidden_size, draft_hidden_size=draft_hidden_size)
            head_path = os.path.join(args.model_path, "ensemble_head.bin")
            print(f"Loading Ensemblemodel and tokenizer from {args.model_path}")
            ensemble_head.load_state_dict(torch.load(head_path))
            ensemble_head = ensemble_head.to(device)
            draft_ensemble_weights = None
        elif args.method == "static_en":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2", # Modified by Ziyi
                # attn_implementation="flash_attention_2"   
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens

            #wzyi made a change
            if args.assistant_schedule != 'dynamic':
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            draft_ensemble_weights = args.draft_ensemble_weights
            ensemble_head = None
        else:
            draft_model = None
            ensemble_head = None
            draft_ensemble_weights = None

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible paths for the template file
        possible_paths = [
            os.path.join(script_dir, 'train_config/template.jinja'),
            os.path.join(script_dir, '../train_config/template.jinja'),
            os.path.join(os.path.dirname(script_dir), 'train_config/template.jinja'),
            os.path.join(script_dir, '..', 'backups/train_config/template.jinja'),
        ]

        # Added by Ziyi
        if 'gemma' in args.target_model:
            for m in (model, draft_model):
                # m.generation_config = GenerationConfig.from_model_config(m.config)
                m.generation_config.cache_implementation = "dynamic"
                m.config.cache_implementation = "dynamic"
            

        template_path = None
        for path in possible_paths:
            if os.path.exists(path):
                template_path = path
                logger.info(f"Process {accelerator.process_index} found template at {template_path}")
                break
        
        if template_path is None:
            # If template file is not found, use a default template
            logger.warning(f"Process {accelerator.process_index} could not find template file, using default template")
            template_content = """{% for message in messages %}
                                {% if message['role'] == 'user' %}
                                {{ message['content'] }}
                                {% elif message['role'] == 'assistant' %}
                                {{ message['content'] }}
                                {% endif %}
                                {% endfor %}"""
        else:
            with open(template_path, 'r') as f:
                template_content = f.read()
        tokenizer.chat_template = template_content
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            max_length=2506,
            max_prompt_length=2048,
            seed=args.seed if args.seed is not None else 42,
            frac_unique_desirable=1.0,
            frac_unique_undesirable=1.0,
            control_tokens={},
        )

        dataloader = SFTDataLoader(
            [args.dataset], 
            tokenizer,
            split=args.split,
            # microbatch_size=args.batch_size,
            microbatch_size=1, #added by kasasiva
            batch_size=args.gpu_count,
            n_examples=args.n_examples, 
            n_epochs=1,
            **data_iterator_kwargs
        )

        # Only create directories on the main process
        if accelerator.is_main_process:
            os.makedirs(args.model_path, exist_ok=True)    
        
        output_path = os.path.join(args.model_path, "{}_{}_generations.json".format(args.method, args.draft_ensemble_weights))
        metrics_path = os.path.join(args.model_path, "{}_{}_metrics.json".format(args.method, args.draft_ensemble_weights))
        
        all_completions, all_labels = [], []
        all_results = []

        all_metrics = {
            "generated": [],
            "total_time": [],
            "num_tokens": [],
            "num_tokens_per_sec": [],
        }

        # We don't need to wrap the models in DDP since we're just doing inference
        # Instead, we'll just distribute the data across processes
        dataloader = accelerator.prepare(dataloader)
        
        for idx, batch in enumerate(dataloader):
            try:
                logger.info(f"Process {accelerator.process_index} processing batch {idx}")
                
                # Move batch to the correct device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                input_ids = batch['original_prompt_input_ids']
                attn_mask = batch['original_prompt_attention_mask']
                labels = [answer[0]['content'] for answer in batch['target']]
                prompts = batch['original_prompt']

                # Clear CUDA cache to free up memory
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    print("Debug the generation schedule: ", )
                    start_time = time.time()
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=args.max_tokens,
                        do_sample=do_sample,
                        use_cache=True,
                        assistant_model=draft_model,
                        output_hidden_states=True, # Ziyi changed here
                        temperature=args.temperature if args.temperature > 0 else 1.0,
                        num_beams=1,
                        ensemble_head=ensemble_head,
                        static_ensemble_draft_weight=draft_ensemble_weights,
                    )
                    end_time = time.time()
                    
                logger.info(f"Process {accelerator.process_index} generated {output_ids.shape[1] - input_ids.shape[1]} tokens in {end_time - start_time:.2f}s")

                all_metrics["total_time"].append(end_time-start_time)
                all_metrics["num_tokens"].append(output_ids[:, input_ids.shape[1]:].shape[-1])
                all_metrics["num_tokens_per_sec"].append(
                        all_metrics["num_tokens"][-1] / all_metrics["total_time"][-1]
                    )
                    
                generations = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

                all_completions.extend(generations)
                all_labels.extend(labels)

                if args.dataset == 'cnndm' or args.dataset == 'xsum':
                    rouge = evaluate.load('rouge')
                elif args.dataset == 'wmt':
                    all_bleu = []
                            

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
                    elif args.dataset == "cnndm":
                        g = cnndm_find_answer(g)
                        metric = rouge.compute(predictions=[g], references=[t])
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "reference": t,
                            "metric": metric,
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
                    elif args.dataset == "xsum":
                        g = xsum_find_answer(g)
                        metric = rouge.compute(predictions=[g], references=[t])
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "reference": t,
                            "metric": metric,
                        })
            except Exception as e:
                logger.error(f"Process {accelerator.process_index} encountered error in batch {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Wait for all processes to finish processing batches
        accelerator.wait_for_everyone()
        logger.info(f"Process {accelerator.process_index} finished processing all batches")
        
        # Gather results from all processes
        try:
            logger.info(f"Process {accelerator.process_index} gathering results")
            all_completions = gather_object(all_completions)
            all_labels = gather_object(all_labels)
            all_results = gather_object(all_results)
            for key in all_metrics:
                if key != "generated":
                    all_metrics[key] = accelerator.gather(torch.tensor(all_metrics[key], device=device)).cpu().numpy().tolist()
            logger.info(f"Process {accelerator.process_index} gathered results successfully")
        except Exception as e:
            logger.error(f"Process {accelerator.process_index} encountered error during gathering: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty results if gathering fails
            if accelerator.is_main_process:
                all_completions = []
                all_labels = []
                all_results = []
        
        # Only the main process should write to files
        if accelerator.is_main_process:
            # Dump all results at once as a JSON array
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

            if args.dataset == 'gsm8k':
                acc = sum(reward_func(all_completions, all_labels)) / len(all_labels)
                metrics = {"accuracy": acc}
                with open(metrics_path, "w") as f:
                    json.dump({"accuracy": acc}, f, indent=2)
            elif args.dataset == "cnndm":
                metrics = rouge.compute(predictions=all_completions, references=all_labels)
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            elif args.dataset == "xsum":
                metrics = rouge.compute(predictions=all_completions, references=all_labels)
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            elif args.dataset == "wmt":
                metrics = np.mean(all_bleu)
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

            result_stats = {
                "performance": metrics,
                "num_tokens_per_sec": np.mean(all_metrics["num_tokens_per_sec"]),
                "total_time": np.mean(all_metrics["total_time"]),
                "num_tokens": np.mean(all_metrics["num_tokens"]),
            }
            
            with open(metrics_path, "w") as f:
                json.dump(result_stats, f, indent=2)

            print(f"Saved generations to {output_path}")
            print(result_stats)
            print(f"Saved Metrics to {metrics_path}")

    except Exception as e:
        logger.error(f"Process {accelerator.process_index} encountered error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Initialize distributed environment variables if not already set
    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

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
    parser.add_argument("--method", type=str, default="sd", choices=['sd', 'sd_en', 'auto', 'static_en'])
    parser.add_argument("--draft_ensemble_weights", type=float, default=0.5, help="The static ensemble weights for draft model, only useful when method is static_en.")
    parser.add_argument("--num_assistant_tokens", type=int, default=10)
    parser.add_argument('--do_sample', type=str, choices=["True", "False"], help="Do Sample for the ensemble task.")
    parser.add_argument('--assistant_schedule', type=str, choices=['constant', 'heuristic', 'dynamic'], default='dynamic', help="Num of draft length schedule.")
    parser.add_argument('--assistant_confidence_threshold', type=float, default=0, help="used only when schedule is not dynamic")

        
    args = parser.parse_args()
    main(args)
