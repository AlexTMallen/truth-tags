import torch
from itertools import islice
from tqdm import tqdm
import uuid
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser
import torch.multiprocessing as mp
import random
import numpy as np
from prompt_processing import templatize_mmlu
from utils import load_accelerated_model_and_tokenizer

seed = 633
random.seed(seed)
np.random.seed(seed)

SAVE_FORMAT = "data/generated-responses/rank_{rank}_{i}_transcripts_mmlu_{model_str}.json"
VICUNA_PREFIX = "A chat between a curious human and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the human's questions."
LLAMA_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully " \
                      "as possible, while being safe.  Your answers should not include any harmful, " \
                      "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure " \
                      "that your responses are socially unbiased and positive in nature.\n" \
                      "If a question does not make any sense, or is not factually coherent, explain " \
                      "why instead of answering something not correct. If you don't know the answer "\
                      "to a question, please don't share false information."

def main_worker(rank, model_name, devices_by_rank, dfs_by_rank, resume_from):
    df = dfs_by_rank[rank]
    del dfs_by_rank
    
    devices = devices_by_rank[rank]
    # load model
    model, tokenizer = load_accelerated_model_and_tokenizer(model_name, GPU_map={i: "70GiB" for i in devices})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
    # tokenizer.eos_token = "### End"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    try:
        max_length = model.config.max_position_embeddings
    except AttributeError:
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 1024
    print("Max length:", max_length)
    
    results = []
    with torch.no_grad():
        for i, row in tqdm(enumerate(df.iloc), total=len(df), position=rank):
            # print(f"\n\n\n####################################  {i}  ##############################################\n\n")
            model_family = model_name.split("/")[0]
            prefix = {"WizardLM": VICUNA_PREFIX + " ",
                      "meta-llama": "<s>[INST] <<SYS>>\n" + LLAMA_SYSTEM_PROMPT + "\n<</SYS>>\n\n"}[model_family]
            prompter_template = {"WizardLM": "USER: {} ",
                                 "meta-llama": "{}"}[model_family]
            assistant_prefix = {"WizardLM": "ASSISTANT:",
                                "meta-llama": " [/INST]"}[model_family]
            assistant_template = {"WizardLM": "ASSISTANT {}</s>",
                                  "meta-llama": " [/INST] {} </s><s>[INST] "}[model_family]
                
            prev_messages = row["prev_messages"]
            msgs = []
            for i, msg in list(enumerate(prev_messages)):  # 0 is least recent
                template = prompter_template if i % 2 == 0 else assistant_template
                msgs.append(template.format(msg))
            transcript = "".join(msgs)
            prompt = prefix + transcript + assistant_prefix

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if len(inputs["input_ids"][0]) > max_length - 256:
                print("Skipping because too long")
                continue
            
            generate_kwargs = {"max_length": max_length, "temperature": 1.0, "repetition_penalty": 1.2, "do_sample": True, "top_p": 0.95}
            gen_kwargs_str = ";".join([f"{k}={v}" for k, v in generate_kwargs.items()])

            model_output = model.generate(
                inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                eos_token_id=tokenizer.eos_token_id,
                **generate_kwargs
            )
            transcript = tokenizer.decode(model_output[0])
            response = transcript.split(assistant_prefix)[-1].lstrip().removesuffix(tokenizer.eos_token)
            # print()
            # print(prev_messages)
            # print(response)
            results.append({
                "message_id": str(uuid.uuid4()),
                "prompt": prompt,
                "assistant_text": response,
                "parent_id": row["parent_id"],
                "prev_messages": prev_messages,
                "role": "assistant",
                "model": model_name,
                "template_name": row["template_name"],
                "gen_kwargs": gen_kwargs_str,
                "subject": row["subject"],
                "answer_index": row["answer"],
                "correct_answer": row["correct_answer"],
                "random_incorrect_answer": row["random_incorrect_answer"],
                "random_other_subject": row["random_other_subject"],
            })

            step = 50
            if i % step == 0 or i == len(df) - 1:
                results_df = pd.DataFrame.from_dict(results)
                model_str = model_name.replace("/", "_")
                new_file = SAVE_FORMAT.format(rank=rank, i=i, model_str=model_str)
                new_file = new_file + f"_resumed_from_{resume_from}" if resume_from > 0 else new_file
                results_df.to_json(new_file, orient="records")
                prev_file = SAVE_FORMAT.format(rank=rank, i=i - step, model_str=model_str)
                prev_file = prev_file + f"_resumed_from_{resume_from}" if resume_from > 0 else prev_file    
                if os.path.exists(prev_file):
                    os.remove(prev_file)


def main(model_family, num_parameters, devices, n_examples, resume_from):
    # load dataset from local jsonl
    df = templatize_mmlu(n_examples=n_examples, subj_balanced=False)
    gpus_needed_per_model = {"7b": 1, "13b": 2, "70b": 6}[num_parameters]
    if len(devices) % gpus_needed_per_model != 0:
        raise ValueError(f"Number of devices ({len(devices)}) is not a multiple of the number of GPUs needed per model ({gpus_needed_per_model})")
    devices_by_rank = np.array_split(devices, len(devices) // gpus_needed_per_model)
    shuffled_df = df.sample(frac=1, random_state=seed).iloc[resume_from:]
    dfs_by_rank = np.array_split(shuffled_df, len(devices_by_rank))

    if model_family == "meta-llama":
        model_name = f"meta-llama/Llama-2-{num_parameters}-chat-hf"
    else:
        raise NotImplementedError

    mp.spawn(main_worker, args=(model_name, devices_by_rank, dfs_by_rank, resume_from), nprocs=len(devices_by_rank), join=True)

    # join results
    results = []
    for rank in range(len(devices_by_rank)):
        name = model_name.replace("/", "_")
        fname = SAVE_FORMAT.format(rank=rank, i=len(dfs_by_rank[rank]), model_str=name)
        fname = fname + f"_resumed_from_{resume_from}" if resume_from > 0 else fname
        results.append(pd.read_json(fname))
    
    # merge with previous results if resuming
    if resume_from > 0:
        results.append(SAVE_FORMAT.format(rank="all", i=resume_from, model_str=name))
    results = pd.concat(results)
    results.to_json(SAVE_FORMAT.format(rank="all", i=n_examples, model_str=name), orient="records")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-family", type=str, default="meta-llama")
    parser.add_argument("--n-parameters", type=str, default="7b")
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7], required=False)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    args = parser.parse_args()
    main(args.model_family, args.n_parameters, args.devices, args.n_examples, args.resume_from)
