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
nproc = 1

LOADED_INSTRUCTIONS = None

msgid_position = {}


def _load_instructions():
    global LOADED_INSTRUCTIONS
    if LOADED_INSTRUCTIONS == None:
        full_instructions = []
        data_path = Path("ft_datasets/ro_oasst/2023-04-12_oasst_all.trees_ro.json")
        instructions = json.load(open(data_path, encoding="utf-8"))
        full_instructions.extend(instructions)
        LOADED_INSTRUCTIONS = full_instructions

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


def build_msgid_dict(msgs):
    d = {}
    for im, msg in enumerate(msgs):
        d[msg["message_id"]] = im
    return d


def find_msg_by_id(msgs, id):
    # global msgid_position
    return msgs[msgid_position[id]]


def extract_conversation(root, level, msg_list, msgs, full_conv_list, fully_translated, fully_validated):

    fmsg = find_msg_by_id(msgs, root["message_id"])
    text_key = "text"
    root_text = fmsg[text_key]
    if root_text not in msg_list:
        msg_list.append(root_text)
        if len(root["replies"]) == 0:
            crt = find_msg_by_id(msgs, root["message_id"])
            crt_list = []
            while crt.get("parent_id", None) != None:
                crt_list.append([crt["role"], crt["text"]]) 
                
                crt = find_msg_by_id(msgs, crt["parent_id"])
           
            crt_list.append([crt["role"], crt["text"]])
            full_conv_list.append(crt_list)
        for child in root["replies"]:
            extract_conversation(child, level+1, msg_list, msgs, full_conv_list, fully_translated, fully_validated)
        
    return full_conv_list, fully_translated, fully_validated


def load_dataset_oasst(filepath):
    global msgid_position
    # random.seed(1238)

    msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages-ro-translated-mbart.jsonl")
    # msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages.jsonl")
    tree_file = Path("ft_datasets/ro_oasst/2023-04-12_oasst_all.trees_ro.json")

    ro = 0
    with msg_file.open('r', encoding="utf-8") as f:
        msgs = f.readlines()
        msgs = list(map(lambda x: json.loads(x), msgs))
        for msg in msgs:
            if msg["lang"] == "ro":# or "text_translated_ro" in msg: 
                ro += 1
    msgid_position = build_msgid_dict(msgs)
   
    
    c = 0
    convs = []
    train_convs, dev_convs, test_convs = [], [], []
    trees = json.load(tree_file.open('r', encoding="utf-8"))
    for tree in trees:
        root = tree["prompt"]
        if (root["lang"] == "ro" or "text_translated_ro" in find_msg_by_id(msgs, root["message_id"])) and len(root["replies"]) > 0:
            x, translated, validated = extract_conversation(root, 0, [], msgs, [], [], [])
            if False in translated:
                translated = False
            else:
                translated = True
            
            if False in validated:
                validated = False
            else:
                validated = True
            
            if translated == False and validated == True:
                print("Translated false but validated?")
                sys.exit()
            
            if translated == True:
                dcs = {}
                dcs["source"] = "oasst"
                dcs["order"] = "reversed"
                dcs["conv"] = x
                convs.append(dcs)
                c += 1
    print(convs[0])
    sys.exit()
    return convs

def get_preprocessed_rooasst_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 256
    else:
        max_words = dataset_config.max_words

    print("RoOASST max words:", max_words)

    load_dataset_oasst(None)
    sys.exit()

    def get_text(sample):
        
        x = []
        for id, d in enumerate(sample["data"]):
            if id % 2 == 0:
                user = "user"
            else:
                user = "assistant"
            x.append({"role": user, "content": d})
        prompt = format_conv(x)
        return {"text": prompt}  

    
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

    
    sys.exit()
    dataset = dataset.map(get_text, num_proc=nproc, remove_columns=["id", "data"], desc="Extract texts")
    dataset = dataset.map(lambda sample: encode_texts(sample, tokenizer), batched=True, num_proc=nproc,  desc="Tokenize texts")

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

    get_preprocessed_rooasst_dataset(None, tokenizer, "full", compute_stats=True)
