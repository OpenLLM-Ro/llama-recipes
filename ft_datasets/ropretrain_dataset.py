import datasets
import numpy as np
import sys
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
from .utils import get_split
import json
import os
import random
import copy


top = -1
nproc = 2

LOADED_DATA = None

def _load_pretrain_from_disk():
    global LOADED_DATA
    if LOADED_DATA == None:
        LOADED_DATA = datasets.load_from_disk('ft_datasets/ccnet_cultura')

    return LOADED_DATA

def chunk_list(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

def encode_texts(sample, tokenizer):
    return tokenizer(sample["raw_text"])

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
        ii = []
        aa = []
        ll = []
        for chunk in chunks["hf_dict_chunks"]:
            for x in range(len(chunk["input_ids"])):
                ii.append(chunk["input_ids"][x])
                aa.append(chunk["attention_mask"][x])
                ll.append(chunk["labels"][x])
        return {"input_ids": ii, "attention_mask": aa, "labels": ll}

def get_preprocessed_ropretrain_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 512
    else:
        max_words = dataset_config.max_words

    dataset = _load_pretrain_from_disk()
    print("Len of entire pretraining dataset:", len(dataset), flush=True)

    split_indexes = get_split(list(range(len(dataset))), split)
    dataset = dataset.select(split_indexes)
    print("Len of {1} split pretraining dataset: {0}".format(len(dataset), split), flush=True)

    if top != -1:
        dataset = dataset.select(range(top))
    dataset = dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=nproc, batched=True, remove_columns=["raw_text"], desc="Tokenize texts")
    
    if compute_stats == True:
        import numpy as np
        lens = np.array(list(map(lambda x: len(x["input_ids"])+1, dataset)))
        print(len(lens))
        print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
        for i in [256, 512, 1024, 2048, 4096]:
            print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
        print("########################################################################################")
        print()

    dataset = dataset.map(lambda sample: prepare_input(sample, [], tokenizer, max_words), num_proc=nproc, remove_columns=["attention_mask", "input_ids"], desc="Build chunks of size {0}".format(max_words))
    dataset = dataset.map(flatten_chunks, batched=True, remove_columns=["hf_dict_chunks"], num_proc=nproc, desc="Flatten chunks")
    return dataset

if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)
    get_preprocessed_ropretrain_dataset(None, tokenizer, "test")
