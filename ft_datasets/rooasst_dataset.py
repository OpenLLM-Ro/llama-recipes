# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import sys
import jsonlines
from pathlib import Path
import datasets
import pandas as pd
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
from chat_utils import format_conv, format_tokens
import random

top = -1
nproc = 2

LOADED_convs = None

msgid_position = {}


def _load_convs():
    global LOADED_convs
    if LOADED_convs == None:
        LOADED_convs = load_dataset_oasst()
    return LOADED_convs


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


def extract_conversation(root, level, msg_list, msgs, full_conv_list):

    fmsg = find_msg_by_id(msgs, root["message_id"])
    text_key = "text"
    root_text = fmsg[text_key]
    if root_text not in msg_list:
        msg_list.append(root_text)
        if len(root["replies"]) == 0:
            crt = find_msg_by_id(msgs, root["message_id"])
            crt_list = []
            while crt.get("parent_id", None) != None:
                crt_list.append([crt["role"], crt["text"], crt["message_id"]]) 
                
                crt = find_msg_by_id(msgs, crt["parent_id"])
           
            crt_list.append([crt["role"], crt["text"], crt["message_id"]])
            full_conv_list.append(crt_list)
        for child in root["replies"]:
            extract_conversation(child, level+1, msg_list, msgs, full_conv_list)
        
    return full_conv_list


def build_new_msgs_file():
    og_msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages-ro-translated-mbart.jsonl")
    tree_file = Path("ft_datasets/ro_oasst/2023-04-12_oasst_all.trees_ro.json")

    msg_dict = {}
    trees = json.load(tree_file.open('r', encoding="utf-8"))
    for tree in trees:
        root = tree["prompt"]
        links = [root] + root["replies"]
        for reply in links:
            # print(reply["message_id"])
            if reply["message_id"] not in msg_dict:
                msg_dict[reply["message_id"]] = reply["text"]
            if "replies" in reply:
                links.extend(reply["replies"])

    print(len(msg_dict))
    with og_msg_file.open('r', encoding="utf-8") as f:
        og_msgs = f.readlines()
        og_msgs = list(map(lambda x: json.loads(x), og_msgs))
    
    print(len(og_msgs))
    lines = []
    for og_msg in og_msgs:
        og_msg["text"] = msg_dict[og_msg["message_id"]]
        lines.append(og_msg)

    with jsonlines.Writer(open("ft_datasets/ro_oasst/2023-04-12_oasst_all.messages_ro.jsonl", "w", encoding="utf-8")) as writer:
        writer.write_all(lines)
    sys.exit()


def load_dataset_oasst():
    global msgid_position

    og_msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages.jsonl")
    with og_msg_file.open('r', encoding="utf-8") as f:
        og_msgs = f.readlines()
        og_msgs = list(map(lambda x: json.loads(x), og_msgs))


    msg_file = Path("ft_datasets/ro_oasst/2023-04-12_oasst_all.messages_ro.jsonl")
    tree_file = Path("ft_datasets/ro_oasst/2023-04-12_oasst_all.trees_ro.json")
    # build new msg file
    #build_new_msgs_file()

    with msg_file.open('r', encoding="utf-8") as f:
        msgs = f.readlines()
        msgs = list(map(lambda x: json.loads(x), msgs))

    msgid_position = build_msgid_dict(msgs)
    
    c = 0
    convs = []
    train_convs, dev_convs, test_convs = [], [], []
    trees = json.load(tree_file.open('r', encoding="utf-8"))
    for tree in trees:
        root = tree["prompt"]
        if tree["tree_state"] != "ready_for_export":
            continue
        if len(root["replies"]) > 0:
            x = extract_conversation(root, 0, [], msgs, [],)
            to_remove_idx = []
            for idx, conv in enumerate(x):
                found = False
                for turn in conv:
                    if msgs[msgid_position[turn[-1]]] == og_msgs[msgid_position[turn[-1]]]:
                        # we have un-translated reply
                        found = True
                        break
                if found == True:
                    to_remove_idx.append(idx)
            
            for idx in to_remove_idx[::-1]:
                del x[idx]


            if x == []:
                continue
            x = list(map(lambda z: list(map(lambda y: [y[0], y[1]], z)), x))
            dcs = {}
            dcs["source"] = "oasst"
            dcs["order"] = "reversed"
            dcs["conv"] = x
            convs.append(dcs)
            c += 1

    return convs


def expand_scraped_convs(convs):
    expanded_convs = []
    err = 0
    for conv in convs:
        if len(conv["conv"]) == 0:
            err += 1
            continue
        if type(conv["conv"][0][0]) == list:
            if "www.reddit.com/r" not in conv["source"] and "oasst" not in conv["source"]: 
                print("Need to expand source <{0}> that is not reddit!".format(conv["source"]))
                sys.exit()
            for inner_conv in conv["conv"]:
                new_conv_dict = copy.deepcopy(conv)
                new_conv_dict["conv"] = inner_conv
                expanded_convs.append(new_conv_dict)
        else:
            expanded_convs.append(conv)
    return expanded_convs


def get_preprocessed_rooasst_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 256
    else:
        max_words = dataset_config.max_words

    print("RoOASST max words:", max_words)

    
    def get_text(sample):
        if sample["order"] != "reversed":
            print("BAD ORDER")
            sys.exit()
        conv = sample["conv"][::-1]        
        x = []
        for id, d in enumerate(conv):
            if id % 2 == 0:
                user = "user"
            else:
                user = "assistant"
            x.append({"role": user, "content": d[1]})
        if x[-1]["role"] == "user":
            del x[-1]

        prompt = format_conv(x)
        prompt = prompt[:5] + prompt[5:].replace("[INST]", "</s><s>[INST]")
        return {"text": prompt}  
    
    def encode_texts(sample, tokenizer):
        return tokenizer(sample["text"])

    def prepare_input(sample, tokenizer, max_tokens):
        sample["input_ids"].append(tokenizer.eos_token_id)
        sample["attention_mask"].append(1)
      
        start_indexes = [i for i, x in enumerate(sample["input_ids"]) if x == tokenizer.encode("[INST]")[1]]
        end_indexes = [i for i, x in enumerate(sample["input_ids"]) if x == tokenizer.encode("[/INST]")[1]]
        if len(start_indexes) != len(end_indexes):
            print("missmatch count of [INST] and [/INST]")
            sys.exit()

        sample["labels"] = copy.deepcopy(sample["input_ids"])
        for i in range(len(start_indexes)):
            st = start_indexes[i]
            en = end_indexes[i]
            sample["labels"][st-1:en+1] = [-100] * (en-st+1+1)

        sample["input_ids"] = sample["input_ids"][:max_tokens]
        sample["attention_mask"] = sample["attention_mask"][:max_tokens]
        sample["labels"] = sample["labels"][:max_tokens]
        
        return {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"], "labels": sample["labels"]}

    convs = _load_convs()
    convs = get_split(convs, split)
    print("Threads:", len(convs))
    convs = expand_scraped_convs(convs)
    print("Convs (expanded):", len(convs))

    if top != -1:
        convs = convs[:top]

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=convs))
    dataset = dataset.map(get_text, num_proc=nproc, remove_columns=["source", "conv", "order"], desc="Extract texts")
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
