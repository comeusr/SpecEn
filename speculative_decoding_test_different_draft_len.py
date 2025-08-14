import os, torch, random
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  
# Seeds
seed = 0
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Determinism switches (may hurt speed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False  # avoid TF32 drift

import argparse
import json
import sys
import re
from jinja2 import Template
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DistributedDataParallelKwargs
from train.dataloader import SFTDataLoader
from train.models import EnsembleWrapper, EnsembleHead
import time
import evaluate

import sys

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {path}')

def cnndm_find_answer(text):
    return re.split("\n\nArticle:", text)[0].strip()

def xsum_find_answer(text):
    return re.split("\n\nDocument:", text)[0].strip()

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

    device1 = torch.device("cuda:0")

    do_sample = (args.do_sample=="True")

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device1)

    if args.method == "sd":
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device1)
        draft_model.generation_config.do_sample = do_sample
        draft_model.generation_config.temperature = args.temperature
        draft_model.generation_config.is_assistant=True
        draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
        ensemble_head = None
        draft_ensemble_weights = None
        
        # Added by wzyi
        draft_model.generation_config.num_assistant_tokens_schedule='constant'
        draft_model.generation_config.assistant_confidence_threshold=0.0

    elif args.method == "sd_en":
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device1)
        draft_model.generation_config.do_sample = do_sample
        draft_model.generation_config.temperature = args.temperature
        draft_model.generation_config.is_assistant=True
        draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
        ensemble_head = EnsembleHead(target_hidden_size=model.config.hidden_size, draft_hidden_size=draft_model.config.hidden_size)
        head_path = os.path.join(args.model_path, "ensemble_head.bin")
        print(f"Loading Ensemblemodel and tokenizer from {args.model_path}")
        ensemble_head.load_state_dict(torch.load(head_path))
        ensemble_head = ensemble_head.to(model.device)
        draft_ensemble_weights = None
    elif args.method == "static_en":
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device1)
        draft_model.generation_config.do_sample = do_sample
        draft_model.generation_config.temperature = args.temperature
        draft_model.generation_config.is_assistant=True
        draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
        draft_ensemble_weights = args.draft_ensemble_weights
        ensemble_head = None
    else:
        draft_model = None
        ensemble_head = None
        draft_ensemble_weights = None

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    tokenizer.chat_template = open('train_config/template.jinja').read()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    test_data = {
        "article":  "(CNN)For superhero fans, the cup runneth over. Most of us know the members of the Avengers by now: Iron Man, Captain America, Hulk and the rest, and the fact that a few more like Quicksilver are joining the cast in the \"Avengers: Age of Ultron\" sequel. But there was one character who remained a mystery: the Vision, to be played by Paul Bettany. Thus far, we've only seen his eyes in a trailer. With less than a month to go before the movie hits theaters, Marvel Studios put all the speculation to rest with a poster featuring Bettany as the heroic android, who was a member of the superhero group for many years in the comics. Meanwhile, as many Marvel fans know, Thursday was the eve of the new Netflix series \"Daredevil,\" and after a photoshopped first look at Charlie Cox's iconic red Daredevil suit went out, Marvel put out a video of the real one. Not to be outdone, director Bryan Singer announced a new character for next year's sequel \"X-Men: Apocalypse,\" by telling Empire magazine that Ben Hardy would be playing the role of the winged mutant Angel. He even had a photo to share. And Thursday's new super images weren't quite done, because the questions over how Jamie Bell's rocky character The Thing in the rebooted \"Fantastic Four\" movie (out August 7) might look were also finally answered. And he looks ... pretty much like The Thing we already knew (but reportedly, CGI this time). Within 24 hours, we got yet another indication that the superhero trend isn't going anywhere anytime soon (and we didn't even talk about the new photo of Ryan Reynolds' \"Deadpool\")."
        # 'article': "SS Sergeant Oskar Groening - known as\u00a0'the bookkeeper of Auschwitz' -\u00a0is set to go on trial charged with complicity in the killing of 300,000 Jews at the Nazi extermination camp . An SS sergeant known as 'the bookkeeper of Auschwitz' is set to go on trial charged with complicity in the killing of 300,000 Jews at the Nazi extermination camp. Auschwitz survivors and the relatives of those murdered there filed into court today for the trial of 93-year-old Oskar Groening - who may well have met their loved ones shortly before they were gassed. They spoke of their pain, pride and duty in confronting this 'cog' in the machinery of genocide. Groening, known as 'The Bookkeeper' for his role in the camp in Nazi-occupied Poland, was tasked with meeting the trains bringing victims there and robbing those aboard of their possessions. Between May 16 and July 11, 1944 he was on duty when 450,000 Hungarian Jews were transported there, with 300,000 being gassed just after arrival. Now those who lost loved ones have travelled thousands of miles  to bear witness as co-plaintiffs against Groening in what may prove to be the last Nazi trial of its kind in Germany. 'I lost 49 members of my family in the Holocaust,' said Eva Pusztai-Fahidi, 89, from Budapest. 'He must have been there, on the ramp, witnessing the suffering. Now I want to look into his eyes and see him recognise his guilt. 'The Holocaust was made of small men like him, little cogs in the machine. It wasn't just big fish, it was people like Oskar Groening. It doesn't matter what his punishment is, but the verdict. The Holocaust deniers can always say a little old Jewish woman told lies. But they will not be able to deny the words of a single SS man who admits he was there.' Hedy Bohm, 87, travelled from Toronto, Canada, with her daughter to bear witness for her lost family. She too lost numerous family members, her father and his sister, together with her small baby, on the day they arrived. 'I am so grateful to have been given this opportunity to come here and testify,' she said. 'I don't know if I ever saw him. But he was there. And there can be no statute of limitations on people who served in such a place.' Groening, a sergeant in the dreaded SS, was in Auschwitz as a guard for two and-a-half years, but prosecutors are charging him with complicity in the murders of 300,000 people who arrived on 137 trains during the 48 day period of that summer in 1944 because of his intact service records. Auschwitz (pictured) survivors and the relatives of those murdered there filed into court today for the trial of 93-year-old Groening . Hedy Bohm (left) and Eva Pusztai-Fahidi (right), survivors of the Auschwitz concentration camp, take part in a news conference ahead of Groening's trial . Groening never denied being at Auschwitz and has been haunted by it ever since. He once admitted: 'I never really left Auschwitz - and it never left me.' But he denies a single instance of killing or cruelty - even though he witnessed plenty. Such a defence worked in Germany before 2011 and the trial of Sobibor death camp guard John Demjanjuk, but no longer. Demjanjuk was tried and convicted for being part of the machinery of mass murder at the camp where 250,000 Jews were liquidated without a single shred of evidence linking him to a crime. There was no-one left alive to testify at his trial in Munich for his role in the extermination of 28,000 Dutch Jews. No-one could say whether he slaughtered with his bare hands, but he was convicted, for the first time in history, simply because he was there - and that is why Groening now has his appointment with justice. Groening, who lives near the Lueneburg Heath - ironically the place where his boss, SS chief Heinrich Himmler, was buried in an unmarked grave after committing suicide when he fell into British hands at the end of the war - lived a comfortable life after the war. He married, had two children and worked as a wages accountant in a glass factory after being released from a POW camp in Britain. Groening was in Auschwitz as a guard for two and-a-half years, but prosecutors are charging him with complicity in the murders of 300,000 people over a two-month period in 1944 . He spoke at trials after the war of the operations of the gas chambers and crematoria but denied any involvement. Now a frail widower, he bears little resemblance to the young soldier with thin glasses shown in a black and white wartime photograph, except for the shadow of a military tattoo on his left arm bearing his 'O' blood type. Groening has said he volunteered for the Waffen SS in 1941 at age 20, drawn by wartime fervour and 'the elegance of the uniform'. But he testified to his nightmares in interviews before he was charged. 'Every night and every day I remember it for the nightmare it was,' he said. 'It was in 1942 that my SS chiefs in Berlin ordered me there. 'I was an official in the prisoners' possessions administration which basically involved removing the money, jewels and other valuables from the inmates, registering it and sending it back to Berlin. 'They had diamonds and gold worth millions and it was my duty to make sure all of it got to Berlin. 'It was completely understood by all that the majority were going straight to the gas chamber, although some believed they were only going to be showered before going to work. Many Jews knew they were going to die. 'One time a drunken SS man discovered a crying baby on the platform. He grabbed the waif by its legs and smashed its head against the side of a truck. My blood froze when I saw it. 'When I saw this I went to my superior officers and made an application for a transfer to the front, to anywhere. But he refused. Down the years I have heard the cries of the dead in my dreams and in every waking moment. I will never be free of them. 'It was becoming harder and harder to suppress everything I saw. On one night in January 1943 I saw for the first time how the Jews were actually gassed. It was in a half-built farmyard near to the Auschwitz-Birkenau camp. A gas chamber was built there. We were searching the wood nearby for prisoners who had escaped. Loved ones (pictured being interviewed before the trial) have travelled thousands of miles to bear witness as co-plaintiffs against Groening . Between May 16 and July 11, 1944 he was on duty when 450,000 Hungarian Jews were transported there, with 300,000 being gassed just after arrival. Pictured, Auschwitz survivors . 'There were more than 100 prisoners and soon there were panic-filled cries as they were herded into the chamber and the door was shut. 'Then a sergeant with a gas mask went to a hole in the wall and from a tin shook Zyklon B gas pellets inside. In that moment the cries of the people inside rose to a crescendo, a choir of madness. These cries I have ringing in my ears to this day. 'I again made an application for a transfer and at the end of October 1944 I was shipped to the Belgian Ardennes where I served with a fighting unit until capture. 'But you can imagine that down the years I have heard the cries of the dead in my dreams and in every waking moment. I will never be free of them. 'I have never been back there because of my shame. This guilt will never leave me. I can only plead for forgiveness and pray for atonement.' Kurt Schrimm, who heads Germany's sole Nazi hunting agency, prepared the case against Groening. Three more are pending against former Auschwitz personnel but age and infirmity seem likely to derail those proceedings before they begin. Judith Kalman, 61, also from Toronto, has travelled to represent her sister Evike,who was six when she was gassed upon arrival at Auschwitz during one of those days that Groening was on duty. She said: 'She never grew up. She will be a six-year-old child forever. A talented, cute little girl who taught herself to read. 'There is not only a shadow on my family, there is a shadow on my whole life. She was gassed 75 days after her sixth birthday on June 3 1944. What could have become of this talented, gifted child? 'I don't feel resentement or hatred when I think of Groening. I believe him when he says he never laid a hand on a Jew. He is unsure himself of his guilt. He wants to have peace of mind, to justify himself, but he knows there can be no justification. 'I will be there for Evika and for all the others who were murdered. I want to understand. He chose to participate in this crime and participated with conviction. And now, finally, he must answer for it.'"
    }

    template_path = './src/mydatasets/cnndm/prompt_fewshot.txt'

    template = Template(read_txt(template_path))

    prompt = template.render(**test_data)

    print("Debug the prompt: ", prompt)

    input_ids = tokenizer.encode(prompt)
    attn_mask = [1]*len(input_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)
    attn_mask = torch.tensor([attn_mask], dtype=torch.long).to(model.device)

    print("Debug input_ids shape: ", input_ids.shape)
    ### The first draft length:
    draft_model.generation_config.num_assistant_tokens = args.draft_len

    model.eval()
    draft_model.eval()

    output_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                use_cache=True,
                assistant_model=draft_model,
                output_hidden_states=True,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                num_beams=1,
                ensemble_head=ensemble_head,
                static_ensemble_draft_weight=draft_ensemble_weights,
            )

    print(tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True))


    ### The second draft length:
    # draft_model.generation_config.num_assistant_tokens = 7

    # output_ids = model.generate(
    #         input_ids,
    #         attention_mask=attn_mask,
    #         max_new_tokens=args.max_tokens,
    #         do_sample=do_sample,
    #         use_cache=True,
    #         assistant_model=draft_model,
    #         output_hidden_states=True,
    #         temperature=args.temperature if args.temperature > 0 else 1.0,
    #         num_beams=1,
    #         ensemble_head=ensemble_head,
    #         static_ensemble_draft_weight=draft_ensemble_weights,
    #     )

    # print(tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sample from a local model using vllm for AlpacaEval")
    parser.add_argument("--model_path", type=str, default="./test_draft_lens", help="Path to the local model folder or the Huggingface repo")
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
    parser.add_argument("--draft_len", type=int, default=5)
    parser.add_argument("--draft_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--method", type=str, default="sd", choices=['sd', 'sd_en', 'auto', 'static_en'])
    parser.add_argument("--draft_ensemble_weights", type=float, default=0.5, help="The static ensemble weights for draft model, only useful when method is static_en.")
    parser.add_argument("--num_assistant_tokens", type=int, default=5)
    parser.add_argument('--do_sample', type=str, default='False', choices=["True", "False"], help="Do Sample for the ensemble task.")
        
    args = parser.parse_args()
    main(args)
        
        

    