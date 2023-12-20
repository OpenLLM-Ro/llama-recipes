import json
import copy
from pathlib import Path
import sys
from .utils import Concatenator
import pandas as pd
import datasets
import random
random.seed(1238)

msgid_position = {}

system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

def convert_conversation_to_format(conv):
    prompt = system_prompt

    for ix, x in enumerate(conv):
        if ix % 2 == 0:
            # user time
            prompt += conv[ix] + " [/INST] "
        else:
            # prompt time
            prompt += conv[ix] + " </s><s>[INST] "
    return prompt


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
        all_messages.extend(list(map(lambda x: x[1], conv)))

    print("Total conversations in ro:", len(convs))
    print("Total messages:", len(all_messages))
    print("Total distinct messages:", len(set(all_messages)))
    full_convs = get_fulls_convs(convs)

    print("Total conversations to model:", len(full_convs))
    return full_convs    

def get_fulls_convs(convs):
    
    full_convs = []
    for conv in convs:
        chrono_conv = conv[::-1]
        for msg_index, msg in enumerate(chrono_conv):
            if msg[0] == 'assistant':
                so_far = chrono_conv[:msg_index+1]
                crt_msgs = list(map(lambda x: x[1], so_far))
                full_convs.append(crt_msgs)

    return full_convs

def load_dataset(split = None):
    global msgid_position
    random.seed(1238)

    msg_file = Path("ft_datasets/oasst1/2023-04-12_oasst_all.messages-ro-translated-mbart-good.jsonl")
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
                    if rs < 0.85:
                        train_convs.extend(x)
                    elif rs < 0.9:
                        dev_convs.extend(x)
                    else:
                        test_convs.extend(x)
                    convs.extend(x)
                    c += 1

    print(len(train_convs), len(dev_convs), len(test_convs))
    print(len(convs))
    print("Total distinct threads in ro:", c)
    

    if split == "full":

        ctrain = copy.deepcopy(train_convs)
        cdev = copy.deepcopy(dev_convs)
        print("FULL:")
        full_convs = build_dataset_and_stats(convs)


        print("TRAIN:")
        train_convs = build_dataset_and_stats(train_convs)

        print("DEV:")
        dev_convs = build_dataset_and_stats(dev_convs)

        print("TRAIN+DEV:")
        ctrain.extend(cdev)
        train_convs = build_dataset_and_stats(ctrain)


        print("TEST:")
        test_convs = build_dataset_and_stats(test_convs)



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

    


def get_preprocessed_oasst1(dataset_config, tokenizer, split):

    convs = load_dataset(split=split)
    def apply_prompt_template(sample):
        return {
            "text": convert_conversation_to_format(sample)
        }
    dataset = list(map(lambda x: apply_prompt_template(x), convs))
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset




if __name__ == "__main__":
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token="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")

    # get_preprocessed_oasst1(None, tokenizer, None)
    d = load_dataset("full")
    d = load_dataset("dev")
    # print(d)
    # print(len(d))