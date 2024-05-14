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

# these flags are used to differentiate between foundational (CULTURA) and chat (SCRAP); TODO: get acces to trainconfig here somehow; maybe from main add to datasetconfig the train_config.type_of_model flag
LOAD_SCRAP = False
LOAD_CULTURA = True


dataset_scraped_path = "ft_datasets/scraped/scraped_convs.json"
LOADED_DATA = None
LOADED_CULTURAX = None

msgid_position = {}

def convert_conv_to_hf(conv):
    new_conv = []
    for ic, c in enumerate(conv):
        
        inner_d = {}
        # if c[0] == "prompter":
        if ic % 2 == 0:
            inner_d["role"] = "user"
        else:
            inner_d["role"] = "assistant"
        inner_d["content"] = c[1]

        new_conv.append(inner_d)

    # # this should only happend for scraped
    # if new_conv[-1]["role"] == "assistant" and new_conv[-2]["role"] == "assistant":
    #     if len(new_conv) % 2 == 0 : 
    #         for i in range(0, len(new_conv)):
    #             if i % 2 == 0:
    #                 new_conv[i]["role"] = "user"

    #     elif len(new_conv) == 2:
    #         new_conv = new_conv
    #     else:
    #         new = {}
    #         new["role"] = "user"
    #         new["content"] = new_conv[-3]["content"]+"\n"+new_conv[-2]["content"]
    #         del new_conv[-3]
    #         del new_conv[-2]
    #         new_conv.insert(-1, new)
    #         for i in range(0, len(new_conv)):
    #             if i % 2 == 0:
    #                 new_conv[i]["role"] = "user"

    return new_conv


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
    

def build_dataset_and_stats(convs, max_words):
    # all_messages = []
    # for conv in convs:
        # print(conv)
        # all_messages.extend(list(map(lambda x: x[1], conv["conv"])))
    # print("Total messages:", len(all_messages))
    # print("Total distinct messages:", len(set(all_messages)))
    # print("Before:", len(convs))

    
    full_convs = get_full_conv(convs, max_words)
    # enforce (if needed after max_words separation) user assistant

    for conv in full_convs:
        for ic, c in enumerate(conv):
            if ic % 2 == 0:
                c["role"] = "user"
            else:
                c["role"] = "assistant"
    # for conv in full_convs:
    #     if conv[0]["role"] != "user":
    #         print(conv)
    #         sys.exit()
    # print("Total conversations in ro:", len(convs), "Total messages:", len(all_messages), "Total disting messages:", len(set(all_messages)))

    # print("After:", len(full_convs))
    # sys.exit()
    # print("Total conversations to model:", len(full_convs))
    # print(convs[0])
    # print(full_convs[0])
    # sys.exit()
    return full_convs    


def get_conv_length(conv):
    ft = ""
    for ic, c in enumerate(conv):
        ft += " " + c["content"]

    return len(ft.split(" ")), len(conv)


def get_full_conv(convs, max_words, print_msgs=True):
    
    full_convs = []
    sources = []


    for iconv, conv in enumerate(convs):

        if conv["order"] == "reversed":
            chrono_conv = conv["conv"][::-1]
        elif conv["order"] == "normal":
            chrono_conv = conv["conv"]

        
        converted_conv = convert_conv_to_hf(chrono_conv)
        crt_conv = converted_conv
        while True:
            crt_words = 0
            # print(iconv, "Big_conv length:", get_conv_length(crt_conv))
            bigger = False
            add_special = True
            for msg_index, msg in enumerate(crt_conv):
                crt_words += len(msg["content"].split(" "))
                # print(msg_index, crt_words)
                if crt_words > max_words:
                    bigger = True
                    if msg_index == 0:
                        bigger = False
                        add_special = False
                        break

                    conv_so_far = crt_conv[:msg_index]
                    len_text, len_conv = get_conv_length(conv_so_far)
                    if len_text - 1 != crt_words - len(msg["content"].split(" ")):
                        print("BAD SPLIT")
                        sys.exit()
                    # print(iconv, "Adding big conv length:", get_conv_length(conv_so_far))
                    full_convs.append(conv_so_far)
                    sources.append(conv["source"])
                    crt_conv = crt_conv[msg_index:]
                    # print("Remaining:", get_conv_length(crt_conv))
                    # print("---------")
                    break
            if bigger == False:
                if add_special == False:
                    break
                # print(iconv, "Adding small conv length:", get_conv_length(crt_conv))
                full_convs.append(crt_conv)
                sources.append(conv["source"])
                break


        # print("len full convs:", len(full_convs))
        # print()
        # if iconv == 471:
            # sys.exit()

        # for msg_index, msg in enumerate(chrono_conv):
        #     if msg[0] == 'assistant':
        #         so_far = chrono_conv[:msg_index+1]
        #         so_far_converted = convert_conv_to_hf(so_far)
        #         # format_tokens([so_far_converted], tokenizer)
        #         # print("##################################################")
        #         # crt_msgs = list(map(lambda x: x[1][:50], so_far))
        #         # print(crt_msgs)
        #         # sys.exit()
        #         # full_convs.append(crt_msgs)
        #         full_convs.append(so_far_converted)
        
    # print(len(list(filter(lambda x: "avocatnet" in x, sources))), len(list(filter(lambda x: "forum.softpedia.com" in x, sources))))
    if print_msgs == True:
        get_sources_stats(sources, "After expanding maxwords")
    # get_sources_stats(full_convs, "After expanding maxwords. ")
    return full_convs


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
                # for y in x:
                #     print(y)
                #     print("------------------------")
                # # print(x)
                # print(len(x))
                # sys.exit()
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
                    # rs = random.random()
                    # dcs = []
                    # for xs in x:
                    #     dc = {}
                    #     dc["source"] = "oasst"
                    #     dc["conv"] = xs
                    #     dc["order"] = "reversed"
                    #     dcs.append(dc)
                    # dcs = x
                    dcs = {}
                    dcs["source"] = "oasst"
                    dcs["order"] = "reversed"
                    dcs["conv"] = x
                    convs.append(dcs)
                    # print(x)
                    # sys.exit()

                    # if rs < 0.85:
                    #     train_convs.extend(dcs)
                    # elif rs < 0.9:
                    #     dev_convs.extend(dcs)
                    # else:
                    #     test_convs.extend(dcs)
                    
                    # convs.extend(dcs)
                    c += 1

    # print("Len convs:", len(convs))
    # print("OASST: ro msgs: {0} | threads:{1}".format(ro, c))
    # random.seed()
    # print(train_convs[0])
    # print(train_convs[1])
    # sys.exit()
    return convs
    if split == "full":
        full_convs = build_dataset_and_stats(convs, max_words)
        return full_convs


    elif split == "train":
        train_convs = build_dataset_and_stats(train_convs, max_words)
        return train_convs

    elif split == "dev":
        dev_convs = build_dataset_and_stats(dev_convs, max_words)
        return dev_convs

    elif split == "test":
        test_convs = build_dataset_and_stats(test_convs, max_words)
        return test_convs

    elif split == "train+dev":
        train_convs.extend(dev_convs)
        train_convs = build_dataset_and_stats(train_convs, max_words)
        return train_convs

    else:
        print("Unrecognized {0} split.".format(split))
        sys.exit()


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

    # if split != "train" and split != "full":
    #     return []
    
    if os.path.isfile(filepath.split(".json")[0]+"_0.json"):
        crt_index = 1
        convs = json.load(open(filepath.split(".json")[0]+"_0.json", "r"))
        print(filepath.split(".json")[0]+"_0.json", len(convs), flush=True)

        while os.path.isfile(filepath.split(".json")[0]+"_{0}.json".format(crt_index)):
            crt_convs = json.load(open(filepath.split(".json")[0]+"_{0}.json".format(crt_index), "r"))
            convs.extend(crt_convs)
            print(filepath.split(".json")[0]+"_{0}.json".format(crt_index), len(crt_convs), flush=True)
            crt_index += 1

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
    # get_sources_stats(links, "Before expanding sources. ")
    # convs = expand_scraped_convs(convs)
    # get_sources_stats(convs, "After  expanding sources. ")
    # full_convs = build_dataset_and_stats(convs, max_words)
    return full_convs


def load_culturaX_convs():

    # ds = datasets.load_dataset('ft_datasets/cultura', streaming=True)
    # ds = ds.take(10)
    ds = datasets.load_from_disk('ft_datasets/cultura')
    return ds
    print(len(ds))
    convs=[]
    for x in ds:
        convs.append(x)
    print(len(convs))
    sys.exit()
    print("loading dataset", flush=True)
    ds = datasets.load_dataset("uonlp/CulturaX", "ro", split="train")
    # ds = datasets.load_dataset("uonlp/CulturaX", "ro", streaming=True, split="train")
    # ds = ds.take(200)
    print("dataset loaded", flush=True)
    def convert_to_format(entry):
        sentences = nltk.tokenize.sent_tokenize(entry["text"])
        sentences = list(map(lambda x: ["assistant", x], sentences))
        for i in range(len(sentences)):
            if i % 2 == 0:
                sentences[i][0] = "prompter"        
        entry["order"] = "normal"
        entry["source"] = "culturaX_:_"+entry["source"]+"_:_"+entry["url"]
        entry["conv"] = sentences
        return entry
    ds = ds.map(convert_to_format, remove_columns=["text", "timestamp", "url"], num_proc=nproc)
    ds.save_to_disk("ft_datasets/cultura/test", max_shard_size="1GB")
    sys.exit()


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


def _load_culturaX_from_disk():
    global LOADED_CULTURAX
    if LOADED_CULTURAX == None:
        LOADED_CULTURAX = datasets.load_from_disk('ft_datasets/cultura')

    return LOADED_CULTURAX


def load_culturaX_dataset(max_words, split, tokenizer):
    
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

    culturaX_dataset = culturaX_dataset.map(get_text, num_proc=nproc, remove_columns=["source", "order", "conv"])
    culturaX_dataset = culturaX_dataset.map(lambda sample: encode_texts(sample, tokenizer), num_proc=nproc, batched=True, remove_columns=["text"])
    culturaX_dataset = culturaX_dataset.map(lambda sample: prepare_input(sample, [], tokenizer, max_words), num_proc=nproc, remove_columns=["attention_mask", "input_ids"])
    culturaX_dataset = culturaX_dataset.map(flatten_chunks, batched=True, remove_columns=["hf_dict_chunks"], num_proc=nproc)
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


def get_preprocessed_chatdataset(dataset_config, tokenizer, split):

    if (LOAD_SCRAP == False and LOAD_CULTURA == False) or (LOAD_SCRAP == True and LOAD_CULTURA == True):
        print("both load scrap and load cultura are {0}".format(LOAD_SCRAP))
        sys.exit()


    def apply_prompt_template_conv(sample):
        return {"text": format_conv(sample)}

    def apply_prompt_template_foundational(sample):
        return {"text": "".join(list(map(lambda x: x["content"], sample)))}
    # TODO: solve this

    if LOAD_CULTURA == True and LOAD_SCRAP == False:
        apply_prompt_template = apply_prompt_template_foundational
    elif LOAD_CULTURA == False and LOAD_SCRAP == True:
        apply_prompt_template = apply_prompt_template_conv



    if dataset_config == None:
        max_words = 512
    else:
        max_words = dataset_config.max_words


    dataset = None
    len_culturaX = "NA"


    # load scraped
    if LOAD_SCRAP == True:

        unexpanded_full_convs = load_datasets()
        print("Split: {0} | Max seq len: {1}".format(split, max_words))
        split_convs = get_split(unexpanded_full_convs, split)
        if top != -1:
            split_convs = split_convs[:top]
        get_sources_stats(split_convs, "Split unexpanded convs")
        split_convs = expand_scraped_convs(split_convs)
        get_sources_stats(split_convs, "After expanding sources")
        split_convs = build_dataset_and_stats(split_convs, max_words)
        # random.shuffle(split_convs)
        dataset = list(map(lambda x: apply_prompt_template(x), split_convs))   
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
    
    if LOAD_CULTURA == True:
        culturaX_dataset = load_culturaX_dataset(max_words, split, tokenizer)
        len_culturaX = len(culturaX_dataset)

        if dataset != None:
            dataset = datasets.concatenate_datasets([dataset, culturaX_dataset])
        else:
            dataset = culturaX_dataset

    dataset = dataset.shuffle(seed=42)
    
    if LOAD_CULTURA == True and LOAD_SCRAP == False:
        prompt_enc_size = 0
    elif LOAD_CULTURA == False and LOAD_SCRAP == True:
        prompt = dataset[0]["text"].split("\n<</SYS>>\n\n")[0] + "\n<</SYS>>\n\n"
        prompt_enc = tokenizer.encode(prompt)
        prompt_enc_size = len(prompt_enc)

    print("Prompt enc size:", prompt_enc_size, flush=True)
    if LOAD_CULTURA == True:
        print("=======================-------------------CulturaX dataset: {0} | split: {1} | max_seq_len: {2}-------------------=======================".format(len_culturaX, split, max_words))    
        return dataset
    
    dataset = dataset.map(lambda sample: get_hf_dict(sample, tokenizer, prompt_enc_size, max_words), batched=True, remove_columns=list(dataset.features), num_proc=nproc)
    print("=======================-------------------CulturaX dataset: {0} | Final dataset: {1} | split: {2}-------------------=======================".format(len_culturaX, len(dataset), split))
    return dataset


def compute_stats():

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=os.getenv("HF_TOKEN"), legacy=False)

    max_words = 99999999999999

    print("max seq len:", max_words)
    # load scraped_conv
    scraped_convs = load_scraped_convs("ft_datasets/scraped/scraped_convs.json", "full")
    convs = scraped_convs     

    len_scrap = len(convs)
    print("Scrap:",len(convs), flush=True)
    # load oasst
    oasst_convs = load_dataset_oasst(split="full")
    print("OASST convs:", len(convs), flush=True) 
    convs.extend(oasst_convs)
    print("All:", len(convs), flush=True)

    def apply_prompt_template(sample):
        return {"text": format_conv(sample)}


    dataset = list(map(lambda x: apply_prompt_template(x), convs))
    prompt = dataset[0]["text"].split("\n<</SYS>>\n\n")[0] + "\n<</SYS>>\n\n"
    prompt_enc = tokenizer.encode(prompt)
    prompt_enc_size = len(prompt_enc)

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(lambda sample: get_hf_dict(sample, tokenizer, prompt_enc_size, max_words), batched=True, remove_columns=list(dataset.features))
    
    rs = []
    for x in dataset:
        rs.append(len(x["input_ids"]))
    
    import numpy as np
    for (text, (start_index, end_index)) in [("scrap",(0, len_scrap)), ("oasst", (len_scrap+1, -1)), ("all", (0, -1))]:
        print(text)
        nrs = np.array(rs[start_index:end_index])
        print("Size:", len(nrs))
        print(np.min(nrs), np.mean(nrs), np.median(nrs), np.max(nrs), np.quantile(nrs, 0.75), np.quantile(nrs, 0.85), np.quantile(nrs, 0.90))

        for i in [256, 512, 1024, 2048, 4096]:
            print("{0}% over {1}".format(100.0*(nrs>i).sum()/len(nrs), i))
        print("########################################################################################")
    return dataset




if __name__ == "__main__":

    # compute_stats()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=os.getenv("HF_TOKEN"), legacy=False)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # tokenizer.save_pretrained("tok_pad")
    
    # text = "Ești un asistent folositor"
    # ids = tokenizer(text)["input_ids"]
    # print(ids)
    # ids.append(2)
    # sys.exit()
    # print(tokenizer.convert_ids_to_tokens(ids))
    # sys.exit()

    # # sys.exit()

    # get_preprocessed_chatdataset(None, tokenizer, "train+dev")
    get_preprocessed_chatdataset(None, tokenizer, "test")
    