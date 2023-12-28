# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"

@dataclass
class chat_dataset:
    dataset: str = "chat_dataset"
    train_split: str = "train+dev"
    test_split: str = "test"
    max_words: int = 2048


@dataclass
class foundational_dataset:
    dataset: str = "foundational_dataset"
    train_split: str = "train+dev"
    test_split: str = "test"
    max_words: int = 2048

@dataclass
class roalpaca_dataset:
    dataset: str = "roalpaca_dataset"
    train_split: str = "train+dev"
    test_split: str = "test"
    max_words: int = 2048



@dataclass
class conversations_dataset:
    dataset: str = "conversations_dataset"
    train_split: str = "train+dev"
    test_split: str = "test"
    max_words: int = 2048


@dataclass
class b_dataset:
    dataset: str = "b_dataset"
    ds_path: str = ""
    train_split: str = "train"
    test_split: str = "validation"
    max_words: int = 4096
    index: int = 0
    combine_splits: bool = False

@dataclass
class graph_dataset:
    dataset: str = "graph_dataset"
    train_split: str = "train"
    test_split: str = "dev"
    max_words: int = 1024