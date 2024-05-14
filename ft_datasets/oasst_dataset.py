import json
import copy
from pathlib import Path
import sys
sys.path.insert(1, 'inference/')
sys.path.insert(1, 'utils/')
from utils import Concatenator, ConcatDataset
from chat_utils import format_tokens
import pandas as pd
import datasets
import random
random.seed(1238)

msgid_position = {}

def convert_conv_to_hf(conv):
    new_conv = []
    for c in conv:
        inner_d = {}
        if c[0] == "prompter":
            inner_d["role"] = "user"
        else:
            inner_d["role"] = "assistant"
        inner_d["content"] = c[1]
        # print(inner_d)

        new_conv.append(inner_d)
    return new_conv


def build_msgid_dict(msgs):
    d = {}
    for im, msg in enumerate(msgs):
        d[msg["message_id"]] = im
    return d


def find_msg_by_id(msgs, id):
    # global msgid_position
    return msgs[msgid_position[id]]
    
    for im, msg in enumerate(msgs):
        if msg["message_id"] == id:
            return msg
    
    return None


def get_node_text(node):
    if "text_translated_ro" in node:
        return node["text_translated_ro"]
    else:
        return node["text"]


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
    

def build_dataset_and_stats(convs):
    all_messages = []
    for conv in convs:
        all_messages.extend(list(map(lambda x: x[1], conv["conv"])))

    print("Total conversations in ro:", len(convs))
    print("Total messages:", len(all_messages))
    print("Total distinct messages:", len(set(all_messages)))
    full_convs = get_full_conv(convs)

    print("Total conversations to model:", len(full_convs))
    return full_convs    

def get_full_conv(convs):
    
    full_convs = []
    for conv in convs:
        if conv["order"] == "reversed":
            chrono_conv = conv["conv"][::-1]
        elif conv["order"] == "normal":
            chrono_conv = conv["conv"]
        # print(chrono_conv)
        for msg_index, msg in enumerate(chrono_conv):
            if msg[0] == 'assistant':
                so_far = chrono_conv[:msg_index+1]
                so_far_converted = convert_conv_to_hf(so_far)
                
                # print("converted:", so_far_converted)
                # print(format_tokens([so_far_converted], tokenizer))
                # crt_msgs = list(map(lambda x: x[1][:50], so_far))
                # print(crt_msgs)
                # sys.exit()
                # full_convs.append(crt_msgs)
                full_convs.append(so_far_converted)

    return full_convs

def load_dataset(split = None):
    global msgid_position
    random.seed(1238)

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
    print("Total ro messages:", ro)
    # sys.exit()
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
                    rs = random.random()
                    dcs = []
                    for xs in x:
                        dc = {}
                        dc["source"] = "oasst"
                        dc["conv"] = xs
                        dc["order"] = "reversed"
                        dcs.append(dc)

                    if rs < 0.85:
                        train_convs.extend(dcs)
                    elif rs < 0.9:
                        dev_convs.extend(dcs)
                    else:
                        test_convs.extend(dcs)
                    
                    convs.extend(dcs)
                    c += 1

    print(len(train_convs), len(dev_convs), len(test_convs))
    print(len(convs))
    print("Total distinct threads in ro:", c)
    # sys.exit()
    
    if split == "full" or 1 == 1:

        ctrain = copy.deepcopy(train_convs)
        cdev = copy.deepcopy(dev_convs)
        print("FULL:")

        full_convs = build_dataset_and_stats(convs)
        sys.exit()


        print("TRAIN:")
        train_convs = build_dataset_and_stats(train_convs)

        print("DEV:")
        dev_convs = build_dataset_and_stats(dev_convs)

        print("TRAIN+DEV:")
        ctrain.extend(cdev)
        train_convs = build_dataset_and_stats(ctrain)


        print("TEST:")
        test_convs = build_dataset_and_stats(test_convs)

        return full_convs


    elif split == "train":
        print("TRAIN:")
        train_convs = build_dataset_and_stats(train_convs)
        return train_convs

    elif split == "dev":
        print("DEV:")
        dev_convs = build_dataset_and_stats(dev_convs)
        return dev_convs

    elif split == "test":
        print("TEST:")
        test_convs = build_dataset_and_stats(test_convs)
        return test_convs

    elif split == "train+dev":
        train_convs.extend(dev_convs)
        train_convs = build_dataset_and_stats(train_convs)
        return train_convs

    else:
        print("Unrecognized {0} split.".format(split))
        sys.exit()


def get_hf_dict(conv):
    hf_dict = {}
    hf_dict["input_ids"] = format_tokens([conv], tokenizer)[0]
    hf_dict["token_type_ids"] = [0] * len(hf_dict["input_ids"])
    hf_dict["attention_mask"] = [1] * len(hf_dict["input_ids"])
    hf_dict["labels"] = hf_dict["input_ids"].copy()
    return hf_dict

def get_preprocessed_oasst1(dataset_config, tokenizer, split):

    convs = load_dataset(split=split)
    def apply_prompt_template(sample):
        return {"text": get_hf_dict(sample)}
    dataset = list(map(lambda x: apply_prompt_template(x), convs))
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(lambda sample: sample["text"], batched=False, remove_columns=list(dataset.features))
    dataset = ConcatDataset(dataset, chunk_size=4096)
    return dataset




if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=os.getenv("HF_TOKEN"))

    get_preprocessed_oasst1(None, tokenizer, "test")
    # d = load_dataset("full")
    # d = load_dataset("dev")
    # print(d)
    # print(len(d))