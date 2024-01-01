import json
import copy
from pathlib import Path
import torch
import sys
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
# from utils import ConcatDataset
from torch.utils.data import Dataset
from chat_utils import format_conv, format_tokens
import pandas as pd
import datasets
import os
import random
import nltk
nltk.download('punkt', quiet=True)
import pickle

top = -1
nproc = 6

LOADED_DATA = None
LOADED_CULTURAX = None

msgid_position = {}


def _load_culturaX_from_disk():
    global LOADED_CULTURAX
    if LOADED_CULTURAX == None:
        LOADED_CULTURAX = datasets.load_from_disk('ft_datasets/cultura')

    return LOADED_CULTURAX


def load_culturaX_dataset(max_words, split, tokenizer, compute_stats=False):
    
    def chunk_list(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_text(sample):
        return {"text": " ".join(x[1] for x in sample["conv"])}

    def encode_texts(sample, tokenizer):
        return tokenizer(sample["text"])
    
    def prepare_input(sample, prompt_enc, tokenizer, max_tokens):
        full_doc_enc = sample["input_ids"]
        full_doc_enc.append(tokenizer.eos_token_id)

        chunks = list(chunk_list(full_doc_enc, max_tokens-len(prompt_enc)))
        hf_dict_chunks = {}
        for chunk_id, chunk in enumerate(chunks):
            if chunk_id == 0:
                hf_dict_chunks["input_ids"] = [prompt_enc + chunk]
                hf_dict_chunks["attention_mask"] = [[1] * len(hf_dict_chunks["input_ids"][0])]
                hf_dict_chunks["labels"] = copy.deepcopy(hf_dict_chunks["input_ids"])
                hf_dict_chunks["labels"][0][:len(prompt_enc)] = [-100] * len(prompt_enc)
            else:
                hf_dict_chunks["input_ids"].append(prompt_enc + chunk)
                hf_dict_chunks["attention_mask"].append([1] * len(hf_dict_chunks["input_ids"][chunk_id]))
                hf_dict_chunks["labels"].append(copy.deepcopy(hf_dict_chunks["input_ids"][chunk_id]))
                hf_dict_chunks["labels"][chunk_id][:len(prompt_enc)] = [-100] * len(prompt_enc)
        return {"hf_dict_chunks": hf_dict_chunks}
    
    def flatten_chunks(chunks):
        
        dicts = []
        ii = []
        aa = []
        ll = []
        for chunk in chunks["hf_dict_chunks"]:
            for x in range(len(chunk["input_ids"])):
                # d = {}
                # d["input_ids"] = chunk["input_ids"][x]
                # d["attention_mask"] = chunk["attention_mask"][x]
                # d["labels"] = chunk["labels"][x]
                # dicts.append(d)
                ii.append(chunk["input_ids"][x])
                aa.append(chunk["attention_mask"][x])
                ll.append(chunk["labels"][x])
        return {"input_ids": ii, "attention_mask": aa, "labels": ll}
    

    culturaX_dataset = _load_culturaX_from_disk()
    print("Len of entire culturaX:", len(culturaX_dataset), flush=True)

    split_indexes = get_split(list(range(len(culturaX_dataset))), split)
    culturaX_dataset = culturaX_dataset.select(split_indexes)

    print("Len of {1} split culturaX: {0}".format(len(culturaX_dataset), split), flush=True)
    if top != -1:
        culturaX_dataset = culturaX_dataset.select(range(top))

    culturaX_dataset = culturaX_dataset.map(get_text, num_proc=nproc, remove_columns=["source", "order", "conv"], desc="Extract texts")
    culturaX_dataset = culturaX_dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=nproc, batched=True, remove_columns=["text"], desc="Tokenize texts")

    if compute_stats == True:
        import numpy as np
        lens = np.array(list(map(lambda x: len(x["input_ids"])+1, culturaX_dataset)))
        print(len(lens))
        print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
        for i in [256, 512, 1024, 2048, 4096]:
            print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
        print("########################################################################################")
        print()
    
    culturaX_dataset = culturaX_dataset.map(lambda sample: prepare_input(sample, [], tokenizer, max_words), num_proc=nproc, remove_columns=["attention_mask", "input_ids"], desc="Build chunks of size {0}".format(max_words))
    culturaX_dataset = culturaX_dataset.shuffle(seed=42)

    culturaX_dataset = culturaX_dataset.map(flatten_chunks, batched=True, remove_columns=["hf_dict_chunks"], num_proc=nproc, desc="Flatten chunks")
    # lens = list(map(lambda x: len(x["input_ids"]), culturaX_dataset))
    return culturaX_dataset


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


def get_preprocessed_foundational_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 512
    else:
        max_words = dataset_config.max_words

    dataset = load_culturaX_dataset(max_words, split, tokenizer, compute_stats)
    
    prompt_enc_size = 0
    print("Prompt enc size:", prompt_enc_size, flush=True)
    print("=======================-------------------CulturaX dataset: {0} | split: {1} | max_seq_len: {2}-------------------=======================".format(len(dataset), split, max_words))    
    return dataset
    
if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)
    get_preprocessed_foundational_dataset(None, tokenizer, "test")
    