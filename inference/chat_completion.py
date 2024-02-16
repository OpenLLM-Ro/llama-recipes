# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
import warnings
from typing import List

from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model
from chat_utils import read_dialogs_from_file, format_tokens

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 1024, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    output: str = None,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        print("LOADING PEFT MODEL", flush=True)
        model = load_peft_model(model, peft_model)
    if os.path.exists(os.path.join(peft_model, "tokenizer")):
        print("Loading tokenizer from peft model: {0}".format(peft_model), flush=True)
        tokenizer = LlamaTokenizer.from_pretrained(peft_model, use_auth_token="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", legacy=False)
    else:
        print("Loading tokenizer from base model: {1}".format(model_name), flush=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token="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", legacy=False)
    print(len(tokenizer))
    print("Pad token id:", tokenizer.pad_token_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    # print("pad token id", tokenizer.pad_token_id)
    # tokenizer.add_special_tokens({"pad_token": "<pad>", "additional_special_tokens": ["[INST]", "[/INST]", "\n<</SYS>>\n\n", "<<SYS>>\n"]})
    # print(len(tokenizer), flush=True)
    # sys.exit()
    # print("pad token id:", tokenizer.pad_token_id)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    # sys.exit()

    chats = format_tokens(dialogs, tokenizer, model_name, peft_model)
    print("Dialogs formatted.", flush=True)
    # print(chats[0])
    # sys.exit()
    output_texts = []
    with torch.no_grad():
        for idx, chat in enumerate(chats):
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            # print("Max new tokens:", max_new_tokens, "| Do_sample:", do_sample, "| Top_p:", top_p, "| Temperature:", temperature, "| Top k:", top_k, flush=True)
            # print(dialogs[0])
            # print(tokens)
            # sys.exit()
            #input_ids=batch["input_ids"].to('cuda')
            # outputs = model.generate(
            #     input_ids=tokens,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=do_sample,
            #     top_p=top_p,
            #     temperature=temperature,
            #     use_cache=use_cache,
            #     top_k=top_k,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     **kwargs
            # )
            outputs = model.generate(input_ids=tokens, max_new_tokens=max_new_tokens, do_sample=do_sample)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"Model output:\n{output_text}")
            print("\n==================================\n", flush=True)
            output_texts.append(output_text+"\n==================================\n")
    
    with open("{0}.txt".format(output), "w", encoding="utf-8") as f:
        f.writelines(output_texts)

            
if __name__ == "__main__":
    fire.Fire(main)
