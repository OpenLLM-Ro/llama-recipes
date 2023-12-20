# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm
import numpy as np
import gc


# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)


import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    cleanup,
    clear_gpu_cache,
    get_parameter_dtypes,
    print_model_size,
    get_policies  
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_int8_training, PeftModel
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import time

def main(**kwargs):
    tstart = time.time()

    folds = 5
    runs = 1
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    # torch.cuda.manual_seed(train_config.seed)
    # torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    full_results = {}

    

    # Load the pre-trained model and setup its configuration
    model = LlamaForSequenceClassification.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
        use_auth_token="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"
    )

    # from transformers import LlamaConfig
    # configuration = LlamaConfig(vocab_size=1000, hidden_size=512, intermediate_size=512, num_hidden_layers=128, num_attention_heads=32)
    # model = LlamaForSequenceClassification(configuration)
    print("Labels:", model.num_labels)
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, use_auth_token="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", legacy=False)
    tokenizer.add_special_tokens({"pad_token": "<PAD>", "additional_special_tokens": ["<UN1>", "<UN2>", "<UN3>", "<UN4>", "<UN5>", "<UN6>", "<UN7>"]})
    print(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(tokenizer, dataset_config, split="test")
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")



    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        if train_config.load_peft_model == "False":
            model = get_peft_model(model, peft_config)
        else:
            print("Loading finetuned peft model")
            model = PeftModel.from_pretrained(model, train_config.load_peft_model, config=peft_config, is_trainable=True)
            if train_config.enable_fsdp and fsdp_config.pure_bf16:
                model.to(torch.bfloat16)
        model.print_trainable_parameters()


    
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")


    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(dataset_train, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True)
        if train_config.run_validation:
            val_sampler = DistributedSampler(dataset_val, rank=dist.get_rank(), num_replicas=dist.get_world_size())
        
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8),
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8),
        )
    
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(model.parameters(), lr=train_config.lr, momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16, use_kahan_summation=False)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader, 
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        classification=True
    )

    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
            
    do_show = False
    if train_config.enable_fsdp:
        if local_rank == 0:
            do_show = True
            tfinal = time.time()
    else:
        do_show = True

    if do_show == True:
        epochs = {}
        for i in range(train_config.num_epochs):
            epochs[i] = []
        f = dataset_config.index
        for inner_k, inner_v in results.items():
            if "auc_epoch" in inner_k:
                crt_epoch = int(inner_k.split("auc_epoch")[1])
                epochs[crt_epoch].append(inner_v)
        print(epochs)
        for k,v in epochs.items():
            print("epoch{0}: {1:.4f} | {2:.4f}".format(k+1, np.mean(v), np.std(v)))
        print("Took", (tfinal-tstart)/60, "minutes")

    

if __name__ == "__main__":
    fire.Fire(main)
