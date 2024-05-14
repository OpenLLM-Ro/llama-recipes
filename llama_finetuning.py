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

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq,
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
from peft import get_peft_model, TaskType, prepare_model_for_int8_training
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
import wandb

def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    print("Type of model:", train_config.type_of_model, flush=True)
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    
    

    # Load the pre-trained model and setup its configuration
    # model = LlamaForCausalLM.from_pretrained(train_config.model_name, load_in_8bit=True if train_config.quantization else None, device_map="auto" if train_config.quantization else None, use_auth_token=os.getenv("HF_TOKEN"))
    # from transformers import LlamaConfig
    # configuration = LlamaConfig(vocab_size=1000, hidden_size=512, intermediate_size=512, num_hidden_layers=128, num_attention_heads=32)
    # model = LlamaForCausalLM(configuration)
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")


    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, use_auth_token=os.getenv("HF_TOKEN"), legacy=False)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # if train_config.type_of_model == "foundational":
        
    if train_config.type_of_model == "chat":
        print("Adding special tokens for chat variant", flush=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"]})
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model.resize_token_embeddings(len(tokenizer))#, pad_to_multiple_of=128)
    elif train_config.type_of_model == "foundational":
        print("Setting pad token id for foundational to [UNK]", flush=True)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        # #tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # #model.config.pad_token_id = tokenizer.pad_token_id
        # model.resize_token_embeddings(len(tokenizer))

    # save tokenizer
    tokenizer.save_pretrained(train_config.dist_checkpoint_root_folder+"/tokenizer")

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        print(peft_config)        
        model = get_peft_model(model, peft_config)
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

    dataset_config = generate_dataset_config(train_config, kwargs)
    
    if not train_config.enable_fsdp or rank == 0:
        wandb_run = wandb.init(project="second-wave",
                               config={"learning_rate": train_config.lr, "model_name": train_config.model_name, 
                                       "model type": train_config.type_of_model, "max_words": dataset_config.max_words})
    else:
        wandb_run = None

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
        
    # Create DataLoaders for the training and validation dataset
    print("train batch size:", train_config.batch_size_training, "eval batch size:", train_config.val_batch_size)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True),
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(model.parameters(), lr=train_config.lr, momentum_dtype=torch.bfloat16, variance_dtype=torch.bfloat16, use_kahan_summation=False,
            weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5)
    
    total_train_samples = len(dataset_train) 
    global_batch_size = world_size * train_config.batch_size_training
    steps_for_one_epoch = int(total_train_samples / global_batch_size)
    warmup_steps = int(0.1*steps_for_one_epoch)
    epochs = train_config.num_epochs
    # print("Total train samples:", total_train_samples, flush=True)
    # print("World size:", world_size, flush=True)
    # print("Global batch size:", global_batch_size, flush=True)
    print("Steps for one epoch:", steps_for_one_epoch, flush=True)
    # print("Warmup steps:", warmup_steps, flush=True)
    # print("Base lr:", train_config.lr, flush=True)
    # print("Epochs:", epochs, flush=True)

    def warmup(current_step, warmup_steps, base_lr):
        if current_step <= warmup_steps:
            return float(current_step * base_lr / warmup_steps) / base_lr
        return base_lr


    # scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: warmup(x, warmup_steps, train_config.lr))
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_for_one_epoch*epochs-warmup_steps, eta_min=0.1*train_config.lr, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    # print(scheduler)
    # # sys.exit()
    scheduler = StepLR(optimizer, step_size=steps_for_one_epoch, gamma=train_config.gamma)

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
        wandb_run=wandb_run
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
