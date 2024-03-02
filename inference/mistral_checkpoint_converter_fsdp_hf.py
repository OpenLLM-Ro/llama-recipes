# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import yaml

from transformers import AutoTokenizer

from model_utils import  load_mistral_from_config

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu, load_model_checkpoint

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):
    
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name').replace("+", "/")

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        

    #loading the tokenizer form the  model_path
    #tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(fsdp_checkpoint_path, "tokenizer"))
    
    print(len(tokenizer))
    #tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n", "<UNK5>", "<UNK6>", "<UNK7>", "<UNK8>"]})
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    print(len(tokenizer))
    #tokenizer.save_pretrained(consolidated_model_path)

    #load the HF model definition from config
    model_def = load_mistral_from_config(HF_model_path_or_name)
    model_def.resize_token_embeddings(len(tokenizer))
    print("model is loaded from config")

    #load the FSDP sharded checkpoints into the model
    # model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    
    # foundational 000-008
    model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-mistralai+Mistral-7B-v0.1/mistralai+Mistral-7B-v0.1-0.pt"))
    # foundational 009-017
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024/models+v3+llama7b-full-1e-4_low-chunk1024-0.pt"))
    # foundational 018-035
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024-009-017/models+v3+llama7b-full-1e-4_low-chunk1024-009-017-0.pt"))

    # v4 on foundational 000-008
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk2048/models+v3+llama7b-full-1e-4_low-chunk2048-0.pt"))
    # v4 on foundational 009-017
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024-009-017/models+v3+llama7b-full-1e-4_low-chunk1024-009-017-0.pt"))
    # v4 on foundational 018-035
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024-018-035/models+v3+llama7b-full-1e-4_low-chunk1024-018-035-0.pt"))


    # roalpaca v5.v7	
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024/models+v3+llama7b-full-1e-4_low-chunk1024-0.pt"))
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024-009-017/models+v3+llama7b-full-1e-4_low-chunk1024-009-017-0.pt"))
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v3+llama7b-full-1e-4_low-chunk1024-018-035/models+v3+llama7b-full-1e-4_low-chunk1024-018-035-0.pt"))

    # roalpaca v6
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v4+llama7b-chat-5e-5_v3-low-chunk/models+v4+llama7b-chat-5e-5_v3-low-chunk-0.pt"))
    #model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v4+llama7b-chat-5e-5_v3-low-chunk1024-018-035-n/models+v4+llama7b-chat-5e-5_v3-low-chunk1024-018-035-n-0.pt"))
    # model = load_model_checkpoint(model_def, 0, os.path.join(fsdp_checkpoint_path, "fine-tuned-models+v4+llama7b-chat-5e-5_v3-low-chunk1024-018-035-nn/models+v4+llama7b-chat-5e-5_v3-low-chunk1024-018-035-nn-0.pt"))


    print("model is loaded from FSDP checkpoints")
    
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)