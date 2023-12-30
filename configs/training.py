# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool= False 
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=64
    num_epochs: int=1
    num_workers_dataloader: int=2
    lr: float=5e-5
    weight_decay: float=0.0
    gamma: float= 0.90 #0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=64
    dataset = "samsum_dataset"
    micro_batch_size: int=64
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    load_peft_model: str = "False"
    use_fast_kernels: bool = False
    type_of_model: str = "chat"

    
    
    