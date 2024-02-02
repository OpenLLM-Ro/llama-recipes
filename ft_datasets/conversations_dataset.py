import json
import copy
from pathlib import Path
import sys
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
from chat_utils import format_conv, format_tokens
import pandas as pd
import datasets
import os
import random

top = -1
nproc = 4

dataset_scraped_path = "ft_datasets/scraped/scraped_convs.json"
LOADED_DATA = None

PROMPT = """\
Ești un asistent folositor, respectuos și onest. Încearcă să ajuți cât mai mult prin informațiile oferite, excluzând răspunsuri toxice, rasiste, sexiste, periculoase și ilegale."""

msgid_position = {}

def build_msgid_dict(msgs):
    d = {}
    for im, msg in enumerate(msgs):
        d[msg["message_id"]] = im
    return d


def find_msg_by_id(msgs, id):
    # global msgid_position
    return msgs[msgid_position[id]]


def get_node_text(node):
    if "text_translated_ro" in node:
        return node["text_translated_ro"]
    else:
        return node["text"]


def get_sources_stats(data, text_start=""):
    if len(data) == 0:
        avocat = 0
        softpedia = 0
        reddit = 0
        oasst = 0
    
    elif type(data[0]) == str:
        fr = "links"
        avocat = len(list(filter(lambda x: "avocatnet" in x, data)))
        softpedia = len(list(filter(lambda x: "forum.softpedia.com" in x, data)))
        reddit = len(list(filter(lambda x: "www.reddit.com/r/" in x, data)))
        oasst = len(list(filter(lambda x: x == "oasst", data)))
        
    elif type(data[0]) == dict:
        fr = "convs"
        avocat = len(list(filter(lambda x: "avocatnet" in x["source"], data)))
        softpedia = len(list(filter(lambda x: "forum.softpedia.com" in x["source"], data)))
        reddit = len(list(filter(lambda x: "www.reddit.com/r/" in x["source"], data)))
        oasst = len(list(filter(lambda x: x["source"] == "oasst", data)))

    
    print("{4}: Avocat: {0} | Softpedia: {1} | Reddit: {2} | OAsst: {6} | Total: {5}".format(avocat, softpedia, reddit, fr, text_start, avocat+softpedia+reddit+oasst, oasst), flush=True)
    return


def extract_conversation(root, level, msg_list, msgs, full_conv_list, fully_translated, fully_validated):

    fmsg = find_msg_by_id(msgs, root["message_id"])
    if "text_translated_ro" in fmsg:
        text_key = "text_translated_ro"
        if fmsg["human_validated"] == False:
            fully_validated.append(False)
    else:
        text_key = "text"
        if fmsg["lang"] != "ro":
            fully_translated.append(False)
            return None, fully_translated, [False]
        
    
    # print("--------------")
    # print(text_key)
    root_text = fmsg[text_key]
    # print(level, root["role"],  root_text[:120])
    # x = find_msg_by_id(msgs, root["message_id"])
    # print(root["message_id"], x.get("parent_id", None))
    if root_text not in msg_list:
        msg_list.append(root_text)
        if len(root["replies"]) == 0:
            crt = find_msg_by_id(msgs, root["message_id"])
            crt_list = []
            while crt.get("parent_id", None) != None:
                crt_list.append([crt["role"], get_node_text(crt)]) 
                crt = find_msg_by_id(msgs, crt["parent_id"])
           
            crt_list.append([crt["role"], get_node_text(crt)])
            full_conv_list.append(crt_list)
        for child in root["replies"]:
            extract_conversation(child, level+1, msg_list, msgs, full_conv_list, fully_translated, fully_validated)
        
    return full_conv_list, fully_translated, fully_validated
    

def get_conv_length(conv):
    ft = ""
    for ic, c in enumerate(conv):
        ft += " " + c["content"]

    return len(ft.split(" ")), len(conv)



def load_dataset_oasst(filepath):
    global msgid_position
    # random.seed(1238)

    msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages-ro-translated-mbart.jsonl")
    # msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages.jsonl")
    tree_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.trees.jsonl")

    ro = 0
    with msg_file.open('r', encoding="utf-8") as f:
        msgs = f.readlines()
        msgs = list(map(lambda x: json.loads(x), msgs))
        for msg in msgs:
            if msg["lang"] == "ro":# or "text_translated_ro" in msg: 
                ro += 1
    msgid_position = build_msgid_dict(msgs)
    # print("Total ro messages:", ro)
    
    
    c = 0
    convs = []
    train_convs, dev_convs, test_convs = [], [], []
    with tree_file.open('r', encoding="utf-8") as f:
        trees = f.readlines()
        trees = list(map(lambda x: json.loads(x), trees))
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


def load_scraped_convs(filepath):
   
    if os.path.isfile(filepath.split(".json")[0]+"_0.json"):
        crt_index = 1
        convs = json.load(open(filepath.split(".json")[0]+"_0.json", "r"))
        print(filepath.split(".json")[0]+"_0.json", len(convs), flush=True)

        while os.path.isfile(filepath.split(".json")[0]+"_{0}.json".format(crt_index)):
            crt_convs = json.load(open(filepath.split(".json")[0]+"_{0}.json".format(crt_index), "r"))
            convs.extend(crt_convs)
            print(filepath.split(".json")[0]+"_{0}.json".format(crt_index), len(crt_convs), flush=True)
            crt_index += 1
            # break

    elif os.path.isfile(filepath):
        convs = json.load(open(filepath, "r"))
        print(filepath, len(convs), flush=True)
    else:
        convs = []

    print("Total scrap loaded:", len(convs), flush=True)

    links = []
    for c in convs:
        links.append(c["source"])
    if len(links) != len(set(links)):
        print(len(links))
        print(len(set(links)))
        import collections
        print([item for item, count in collections.Counter(links).items() if count > 1])
        print("ERROR: Duplicates in links")
        sys.exit()
    
    return convs


def get_hf_dict(conv, tokenizer, prompt_enc_size, max_length):
    hf_dict = {}
    hf_dict = tokenizer(conv["text"], truncation=True, max_length=max_length)
    hf_dict["labels"] = copy.deepcopy(hf_dict["input_ids"])
    if type(hf_dict["labels"][0]) == list :
        for i in range(len(hf_dict["labels"])):
            hf_dict["labels"][i][:prompt_enc_size] = [-100] * prompt_enc_size
    else:
        hf_dict["labels"][:prompt_enc_size] = [-100] * prompt_enc_size
    return hf_dict


def load_datasets():
    global LOADED_DATA
    if LOADED_DATA == None:
        # scraped_convs = []
        scraped_convs = load_scraped_convs(dataset_scraped_path)
        oasst_convs = load_dataset_oasst(dataset_scraped_path)
        convs = scraped_convs
        convs.extend(oasst_convs)
        LOADED_DATA = convs
    
    return LOADED_DATA


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


def get_preprocessed_conversations_dataset(dataset_config, tokenizer, split, compute_stats=False):

    if dataset_config == None:
        max_words = 1024
    else:
        max_words = dataset_config.max_words

    def encode_texts(sample, tokenizer):
        return tokenizer(sample["conv"])
    
    def replace_occurences(old_value, new_value, ids, start_index=0):
        new_ids = copy.deepcopy(ids)
        for i in range(start_index, len(new_ids)):
            if new_ids[i] == old_value:
                new_ids[i] = new_value
        return new_ids

    def get_last_index(value, ids):
        return len(ids) - 1 - ids[::-1].index(value)

    def ensure_conv_format(conv, max_words, tokenizer, prompt_enc):
        start_inst_enc = tokenizer.encode("[INST]")[1]
        end_inst_enc = tokenizer.encode("[/INST]")[1]
        if conv[len(prompt_enc)] == start_inst_enc:
            conv = conv[:len(prompt_enc)] + conv[len(prompt_enc)+1:]

        first_start_index = conv.index(start_inst_enc)
        if start_inst_enc in conv[first_start_index+1:]:
            second_start_index = conv[first_start_index+1:].index(start_inst_enc) + first_start_index
        else:
            second_start_index = 99999999999

        # there might be a case here that we don't have end_inst_ec
        if end_inst_enc not in conv:
            conv.append(end_inst_enc)
        first_end_index = conv.index(end_inst_enc)
        
        # print("-----", first_start_index, second_start_index, first_end_index)
        if second_start_index < first_end_index:
            # print("NEED TO CHANGE INST HERE")
            # print(conv)
            # print(conv[len(prompt_enc):])
            # print(conv)
            # replace [INST] with -1
            conv = replace_occurences(start_inst_enc, -1, conv, first_start_index+1)
            # replace [/INST] with -2
            conv = replace_occurences(end_inst_enc, -2, conv)

            # replace -1 with [/INST]
            conv = replace_occurences(-1, end_inst_enc, conv)
            # replace -2 with [INST]
            conv = replace_occurences(-2, start_inst_enc, conv)

            # print(crt_ids)
            if conv[-1] == start_inst_enc:
                conv = conv[:-2]
            elif conv[-2] == start_inst_enc:
                conv = conv[:-2] + conv[-1:]
            
        last_start_index = get_last_index(start_inst_enc, conv)
        last_end_index = get_last_index(end_inst_enc, conv)

        # print(last_start_index, last_end_index)
            
        if last_end_index < last_start_index:
            if conv[-1] == 2: # this is </s>
                conv.insert(-2, 29871)
                conv.insert(-2, end_inst_enc)
            
            elif conv[-1] == 29871:
                conv.append(end_inst_enc)

            else:
                conv.append(29871)
                conv.append(end_inst_enc)
        


        while len(conv) > max_words:
            conv = conv[:-3] + conv[-2:]

        return conv

    def prepare_input(sample, prompt_enc, tokenizer, max_tokens):
        full_doc_enc = sample["input_ids"]
        full_doc_enc.append(tokenizer.eos_token_id)

        start_inst_enc = tokenizer.encode("[INST]")[1]
        end_inst_enc = tokenizer.encode("[/INST]")[1]
        
        extended = []
        full_enc = full_doc_enc[len(prompt_enc):]
        # print(full_doc_enc)
        i = 0
        crt_ids = []
        while True:
            if len(full_enc) <= 1:
                # print("Adding 594", len(crt_ids+full_enc), flush=True)
                extended.append(crt_ids + full_enc)
                break
            if i % 2 == 0:
                # search for the first [/INST]
                end_inst_index = full_enc.index(end_inst_enc)
                if i == 0:
                    inner_ids = prompt_enc + full_enc[:end_inst_index+1]
                    if len(inner_ids) > max_words:
                        break
                else:
                    inner_ids = full_enc[:end_inst_index+1]

            else:
                # print(full_enc)
                if start_inst_enc not in full_enc:
                    # print("Adding 610", len(crt_ids + full_enc), flush=True)
                    extended.append(crt_ids + full_enc)
                    break
                start_inst_index = full_enc.index(start_inst_enc)
                inner_ids = full_enc[:start_inst_index]


            if len(inner_ids) > max_words:
                # print(len(crt_ids))
                if len(crt_ids) != 0:
                    # print("AAAAAAAAAAAAAAA", len(crt_ids), len(inner_ids))
                    # print("Adding 621", len(crt_ids), flush=True)
                    extended.append(crt_ids)
                # print("BREAK HERE FOR SIZE?")
                # print(tokenizer.decode(inner_ids))
                break

            # print("i =", i, "| len(crt_ids + inner_ids) =", len(crt_ids) + len(inner_ids))
            # print("len(full_enc) =", len(full_enc), "| len(extended) =", len(extended), "| len(prompt_enc) =", len(prompt_enc))
            # print("inner_ids =", tokenizer.decode(inner_ids).replace("\n", ""))
            # print()

            if len(crt_ids) + len(inner_ids) > max_words:
                # print("STOP")
                # print(crt_ids)
                # print(len(crt_ids))
                # print(tokenizer.decode(crt_ids))
                if crt_ids[-1] == 29871:
                    crt_ids = crt_ids[:-1]
                
                # print()
                # print("FULL", i, tokenizer.decode(crt_ids))
                # print("Adding 642", len(crt_ids), len(inner_ids), crt_ids, flush=True)
                # print(tokenizer.decode(inner_ids))
                if crt_ids == prompt_enc:
                    break
                extended.append(crt_ids)
                if len(full_enc) + len(prompt_enc) < max_words:
                    # print("STOP HEREE")
                    # print("Adding 646", len(prompt_enc + full_enc), flush=True)
                    extended.append(prompt_enc + full_enc)
                    break
                
                if i % 2 == 0:
                    full_enc = full_enc[1:]
                # if len(extended) == 4:
                #     sys.exit()
                crt_ids = copy.deepcopy(prompt_enc)
            
            else:
                if i == 0:
                    full_enc = full_enc[len(inner_ids)-len(prompt_enc):]    
                else:
                    full_enc = full_enc[len(inner_ids):]
                i += 1
                crt_ids.extend(inner_ids)
        
        # extra safe step
        extended = list(filter(lambda x: x != prompt_enc, extended))

        for i, conv in enumerate(extended):
            # print(i, len(conv), tokenizer.decode(conv))
            nconv = ensure_conv_format(conv, max_words, tokenizer, prompt_enc)
            # print(i, len(nconv))#, tokenizer.decode(nconv))
            extended[i] = nconv
        del sample
        del full_enc
        del full_doc_enc
        chunks = extended
        hf_dict_chunks = {}
        for chunk_id, chunk in enumerate(chunks):
            if chunk_id == 0:
                hf_dict_chunks["input_ids"] = [chunk]
                hf_dict_chunks["attention_mask"] = [[1] * len(hf_dict_chunks["input_ids"][0])]
                hf_dict_chunks["labels"] = copy.deepcopy(hf_dict_chunks["input_ids"])
                hf_dict_chunks["labels"][0][:len(prompt_enc)] = [-100] * len(prompt_enc)
            else:
                hf_dict_chunks["input_ids"].append(chunk)
                hf_dict_chunks["attention_mask"].append([1] * len(hf_dict_chunks["input_ids"][chunk_id]))
                hf_dict_chunks["labels"].append(copy.deepcopy(hf_dict_chunks["input_ids"][chunk_id]))
                hf_dict_chunks["labels"][chunk_id][:len(prompt_enc)] = [-100] * len(prompt_enc)
        del chunks
        return {"hf_dict_chunks": hf_dict_chunks}
    
    def flatten_chunks(data):

        ii = []
        aa = []
        ll = []
        ss = []

        for chunk_id, chunk in enumerate(data["hf_dict_chunks"]):
            if chunk["input_ids"] == None:
                continue
            source = data["source"][chunk_id]
            for x in range(len(chunk["input_ids"])):
                ii.append(chunk["input_ids"][x])
                aa.append(chunk["attention_mask"][x])
                ll.append(chunk["labels"][x])
                ss.append(source)
        return {"input_ids": ii, "attention_mask": aa, "labels": ll, "sources": ss}
    
    def process_conv(conv, system_prompt):
        new_conv = {}
        new_conv["source"] = conv["source"]

        if conv["order"] == "reversed":
            chrono_conv = conv["conv"][::-1]
        elif conv["order"] == "normal":
            chrono_conv = conv["conv"]

        new_conv["conv"] = []
        for ic, c in enumerate(chrono_conv):       
            inner_d = {}
            if ic % 2 == 0:
                inner_d["role"] = "user"
            else:
                inner_d["role"] = "assistant"
            inner_d["content"] = c[1]
            new_conv["conv"].append(inner_d)
        new_conv["conv"] = format_conv(new_conv["conv"], system_prompt)
        return new_conv    

    unexpanded_full_convs = load_datasets()
    print("Split: {0} | Max seq len: {1}".format(split, max_words))
    split_convs = get_split(unexpanded_full_convs, split)
    if top != -1:
        split_convs = split_convs[:top]
    # split_convs = [split_convs[0], split_convs[2]]
    # 2, 5, 10
    # split_convs = [split_convs[2], split_convs[5], split_convs[10]]
    # split_convs = split_convs[352+62+33:]
    # print(split_convs)

    get_sources_stats(split_convs, "Split unexpanded convs")
    split_convs = expand_scraped_convs(split_convs)
    get_sources_stats(split_convs, "After expanding sources")
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=split_convs))
    dataset = dataset.remove_columns(["topic"])

    dataset = dataset.filter(lambda conv: len(conv["conv"]) > 1, num_proc=nproc, desc="Filter short conversations")
    # print(dataset)
    get_sources_stats(list(map(lambda x: x["source"], dataset)), "After filtering short conversations")

    dataset = dataset.map(lambda conv: process_conv(conv, PROMPT), num_proc=nproc, desc="Preprocess & format conversation")  
    dataset = dataset.remove_columns(["order"])
    # print(dataset)
    # get prompt here
    prompt = dataset[0]["conv"].split("\n<</SYS>>\n\n")[0] + "\n<</SYS>>\n\n"
    prompt_enc = tokenizer.encode(prompt)
    prompt_enc_size = len(prompt_enc)

    dataset = dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=nproc, batched=True, desc="Tokenize texts")
    dataset = dataset.remove_columns(["conv", "attention_mask"])
    # print(dataset)
    if compute_stats == True:
        import numpy as np
        lens = np.array(list(map(lambda x: len(x["input_ids"])+1, dataset)))
        print(len(lens))
        print(np.min(lens), np.mean(lens), np.median(lens), np.max(lens), np.quantile(lens, 0.75), np.quantile(lens, 0.85), np.quantile(lens, 0.90))
        for i in [256, 512, 1024, 2048, 4096]:
            print("{0}% over {1}".format(100.0*(lens>i).sum()/len(lens), i))
        print("########################################################################################")
        print()

    dataset = dataset.map(lambda sample: prepare_input(sample, prompt_enc, tokenizer, max_words), num_proc=nproc, remove_columns=["input_ids"], desc="Build chunks of size {0}".format(max_words))
    # print(dataset)
    dataset = dataset.shuffle(seed=42)
    columns_to_remove = dataset.column_names + ["hf_dict_chunks"]
    dataset = dataset.map(flatten_chunks, batched=True, num_proc=nproc, remove_columns=columns_to_remove, desc="Flatten chunks")#, keep_in_memory=False, cache_file_name="tmp.cache")
    # dataset = dataset.remove_columns(columns_to_remove)
    # print(dataset)
    # dataset = dataset.select_columns(["hf_dict_chunks", "source"])
    # dataset = dataset.map(flatten_chunks, batched=True, num_proc=nproc, desc="Flatten chunks")
    get_sources_stats(list(map(lambda x: x["sources"], dataset)), "After building conversation chunks")
    dataset = dataset.remove_columns(["sources"])

    return dataset

    

if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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", legacy=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"]})
    get_preprocessed_conversations_dataset(None, tokenizer, "train+dev")
    