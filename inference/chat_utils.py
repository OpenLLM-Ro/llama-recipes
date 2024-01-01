# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal, Optional, Tuple, TypedDict, Union
import json
import sys

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT_ENGLISH = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

DEFAULT_SYSTEM_PROMPT_RO = """\
Ești un asistent folositor, respectuos și onest. Încearcă să ajuți cât mai mult prin informațiile oferite, excluzând răspunsuri toxice, rasiste, sexiste, periculoase și ilegale."""


DEFAULT_SYSTEM_PROMPT_NONE = ""


DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_RO

def format_tokens(dialogs, tokenizer, model_name, peft_model):
    print(model_name)
    if "v1" in model_name:
        prompt = DEFAULT_SYSTEM_PROMPT_ENGLISH
    elif "v2" in model_name or "v4" in model_name or "v5" in model_name or "v6" in model_name:
        if "-chat" in model_name:
            prompt = DEFAULT_SYSTEM_PROMPT_RO
        elif "-full" in model_name:
            prompt = DEFAULT_SYSTEM_PROMPT_NONE
    elif "v3" in model_name:
        prompt = DEFAULT_SYSTEM_PROMPT_NONE
    elif "ndrei481" in model_name :
        prompt = DEFAULT_SYSTEM_PROMPT_NONE
    elif peft_model != None and "denis" in peft_model:
        prompt = DEFAULT_SYSTEM_PROMPT_NONE
    else:
        # this is for raw llama maybe
        prompt = DEFAULT_SYSTEM_PROMPT_RO
    print("Model name: {0} | Peft model: {2} | Prompt: {1}".format(model_name, prompt[:30], peft_model))
    prompt_tokens = []
    for dialog in dialogs:

        if prompt == DEFAULT_SYSTEM_PROMPT_NONE:
            dialog_text = [x['content'].strip() for x in dialog]
            dialog_text = "".join(dialog_text)
            dialog_tokens = tokenizer.encode(dialog_text) 
            prompt_tokens.append(dialog_tokens)
            continue
        if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ] + dialog
        #print(dialog)
        #print()
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        #print(dialog)
        #sys.exit()
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        # if dialog[-1]["role"] == "user":
             # this is for inference
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        # dialog_text += f"FINAL{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        # print(dialog_tokens)
        # print(tokenizer.decode(dialog_tokens))
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def format_conv(dialog, system_prompt=None):
    if system_prompt == None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # print("dialog:", dialog)
    dialog_text = ""
    # print("dialog:", dialog)
    # print(list(map(lambda x: x["role"], dialog)))
    if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ] + dialog
    # print(list(map(lambda x: x["role"], dialog)))    
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    # print(list(map(lambda x: x["role"], dialog)))
    # print(len(dialog))
    # print(dialog[0])
    # print()
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system','user' and 'assistant' roles, "
        "starting with user and alternating (u/a/u/a/u...)"
    )
    """
    Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
    Here, we are adding it manually.
    """
    dialog_text = [f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                   for prompt, answer in zip(dialog[::2], dialog[1::2])]

    if dialog[-1]["role"] == "user":
            # this is for inference
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_text += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
    dialog_text = "".join(dialog_text)
    return dialog_text


def read_dialogs_from_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        dialogs = json.load(file)
    return dialogs


if __name__ == "__main__":
    print("TEST")

    text = [{"role": "user", "content": "Pe ce fus orar se află Atena?"}]

    # x = format_tokens([text], None, "v4-chat", None)

    y = format_conv(text)
    print(y)