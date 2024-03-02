# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import sys
from pathlib import Path
import datasets
import pandas as pd
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
from chat_utils import format_conv, format_tokens
import random

top = -1
nproc = 2

LOADED_INSTRUCTIONS = None

def _load_instructions():
    global LOADED_INSTRUCTIONS
    if LOADED_INSTRUCTIONS == None:
        full_instructions = []
        data_path = Path("ft_datasets/ro_camel/biology.json")
        instructions = json.load(open(data_path, encoding="utf-8"))
        full_instructions.extend(instructions)

        data_path = Path("ft_datasets/ro_camel/chemistry.json")
        instructions = json.load(open(data_path, encoding="utf-8"))
        full_instructions.extend(instructions)

        data_path = Path("ft_datasets/ro_camel/math.json")
        instructions = json.load(open(data_path, encoding="utf-8"))
        full_instructions.extend(instructions)

        data_path = Path("ft_datasets/ro_camel/physics.json")
        instructions = json.load(open(data_path, encoding="utf-8"))
        full_instructions.extend(instructions)
        LOADED_INSTRUCTIONS = full_instructions

        # sys.exit()
    return LOADED_INSTRUCTIONS


def get_split(convs, split):

    if split == "full":
        return convs
    split_convs = []
    # set random seed
    random.seed(1238)

    for conv in convs:
        rs = random.random()
        if split == "train" and rs < 0.85:
            split_convs.append(conv)
        elif split == "dev" and rs > 0.85 and rs < 0.9: 
            split_convs.append(conv)
        elif split == "test" and rs > 0.9:
            split_convs.append(conv)
        elif split == "train+dev" and rs < 0.9:
            split_convs.append(conv)
    random.seed()
    return split_convs


def get_preprocessed_rocamel_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 256
    else:
        max_words = dataset_config.max_words

    print("RoCamel max words:", max_words)

    def get_text(sample):
        
        question = sample["message_1"]
        answer = sample["message_2"]

        x = [{"role": "user", "content": question}]
        prompt = format_conv(x)
        return {"prompt": prompt, "text": prompt + " " + answer}  

    
    def encode_texts(sample, tokenizer):
        return tokenizer(sample["text"])
 
    def find_sub_list(sl,l):
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll-1
        return (-1, -1)
    
    def prepare_input(sample, tokenizer, max_tokens):
        sample["input_ids"].append(tokenizer.eos_token_id)
        sample["attention_mask"].append(1)
        end = sample["input_ids"].index(tokenizer.encode("[/INST]")[1])

        # trim
        # print(len(sample["input_ids"]), len(sample["attention_mask"]))
        sample["input_ids"] = sample["input_ids"][:max_tokens]
        sample["attention_mask"] = sample["attention_mask"][:max_tokens]
        # print(len(sample["input_ids"]), len(sample["attention_mask"]))

        # build labels
        sample["labels"] = copy.deepcopy(sample["input_ids"])
        sample["labels"][:end+1] = [-100] * (end+1)
        sample["labels"] = sample["labels"][:max_tokens]
    
        return {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"], "labels": sample["labels"]}


    instructions = _load_instructions()
    instructions = get_split(instructions, split)
    if top != -1:
        instructions = instructions[:top]

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=instructions))
    dataset = dataset.map(get_text, num_proc=nproc, remove_columns=["role_1", "topic;", "id", "message_1", "message_2", "sub_topic"], desc="Extract texts")
    dataset = dataset.map(lambda sample: encode_texts(sample, tokenizer), batched=True, num_proc=nproc, remove_columns=["prompt"], desc="Tokenize texts")

    if compute_stats == True:
        import numpy as np
        lens = np.array(list(map(lambda x: len(x["input_ids"])+1, dataset)))
        print(len(lens))
        print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
        for i in [256, 512, 1024, 2048, 4096]:
            print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
        print("########################################################################################")
        print()

    dataset = dataset.map(lambda sample: prepare_input(sample, tokenizer, max_words), remove_columns=["text"], num_proc=nproc, desc="Prepare inputs")
    dataset = dataset.shuffle(seed=42)

    return dataset



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"]})

    get_preprocessed_rocamel_dataset(None, tokenizer, "full", compute_stats=True)