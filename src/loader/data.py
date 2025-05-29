from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Optional, List, Union

import logging
import random
import glob

from torch.utils.data import DataLoader

from .data_format.standard import StandardDataset, StandardDataCollator
from .data_format.polynomial import MonomialCollator

@dataclass
class SplitConfig:
    name: str
    batch_size: int
    shuffle: bool = False
    encoding: str = "infix"

def read_split_file(path: str, sample_size: Optional[int] = None, skimming: bool = False) -> tuple[list[str], list[str]]:
    """Read file and split into input and output texts"""
    start_time = time()
    
    try:
        with open(path, "r") as f:
            data = f.read().splitlines()
        if sample_size:
            if skimming:
                random.shuffle(data)
            
            data = data[:sample_size]
            
        input_texts = [line.split(" # ")[0].strip() for line in data]
        target_texts = [line.split(" # ")[1].strip() for line in data]
        
        elapsed = time() - start_time
        logging.info(f"Loaded {len(data)} examples from {path} in {elapsed:.2f} seconds")
        
        return input_texts, target_texts
    
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise

def read_split_files(
    dir_path: str,
    prefix: str,
    encoding: str = "infix",
    sample_size: Optional[int] = None,
    skimming: bool = False
) -> tuple[list[str], list[str]]:
    """
    Read multiple files in directory and split into input and output texts
    prefix: e.g. 'train' or 'test'
    encoding: e.g. 'infix'
    """
    pattern = f"{dir_path}_index_range=*.{encoding}"
    files = sorted(glob.glob(pattern))

    all_input_texts = []
    all_target_texts = []
    for file in files:
        print("file:", file)
        input_texts, target_texts = read_split_file(file, None, skimming)
        all_input_texts.extend(input_texts)
        all_target_texts.extend(target_texts)
        if sample_size and len(all_input_texts) >= sample_size:
            break

    if sample_size:
        all_input_texts = all_input_texts[:sample_size]
        all_target_texts = all_target_texts[:sample_size]
        
    return all_input_texts, all_target_texts

def load_data(
    data_path: Union[str, Path],
    tokenizer = None,
    processor = None,
    subprocessors = None,
    splits: List[SplitConfig] = None,
    return_dataloader: bool = True,
    num_workers: int = None,
    pin_memory: bool = True,
    sample_size: Optional[int] = None,
    train_test_split: Optional[List[int]] = None,
    data_collator_name: Optional[str] = None,
    sample_skimming: bool = False,
    aware_of_padding: bool = False,
    testset_save_path: Optional[str] = None
):
    """Function to load data"""
    splits = [SplitConfig(**split) for split in splits]
    assert len(splits) == 1
    split = splits[0]
    
    encoding = split.encoding
    pattern = f"{data_path}_index_range=*.{split.encoding}"
    files = glob.glob(pattern)
    
    sample_size = sum(train_test_split) if train_test_split else sample_size
    if files:
        input_texts, target_texts = read_split_files(str(data_path), split.name, encoding, sample_size, skimming=sample_skimming)
    else:
        path = f"{data_path}.{split.encoding}"
        input_texts, target_texts = read_split_file(path, sample_size, skimming=sample_skimming)

    
    datasets = []
    if train_test_split is not None:
        assert(sum(train_test_split) <= len(input_texts))
        train_size, test_size = train_test_split[0], train_test_split[1]
        input_texts, test_input_texts = input_texts[:train_size], input_texts[train_size:train_size+test_size]
        target_texts, test_target_texts = target_texts[:train_size], target_texts[train_size:train_size+test_size]
        # Create dataset
        dataset = StandardDataset(
            input_texts=input_texts,
            target_texts=target_texts,
            processor=processor,
            subprocessors=subprocessors
        )
        test_dataset = StandardDataset(
            input_texts=test_input_texts,
            target_texts=test_target_texts,
            processor=processor,
            subprocessors=subprocessors
        )
        
        datasets = [dataset, test_dataset]

        if testset_save_path is not None:
            with open(testset_save_path, "w") as f:
                for inp, tgt in zip(test_input_texts, test_target_texts):
                    f.write(f"{inp} # {tgt}\n")
    else:
        # Create dataset
        dataset = StandardDataset(
            input_texts=input_texts,
            target_texts=target_texts,
            processor=processor,
            subprocessors=subprocessors
        )

        if testset_save_path is not None and split.name == "test":
            with open(testset_save_path, "w") as f:
                for inp, tgt in zip(input_texts, target_texts):
                    f.write(f"{inp} # {tgt}\n")

        datasets = [dataset]

        
    if data_collator_name in ('standard', None):
        data_collator = StandardDataCollator(tokenizer=tokenizer, aware_of_padding=aware_of_padding)
    elif data_collator_name == 'monomial':
        data_collator = MonomialCollator(tokenizer=tokenizer, aware_of_padding=aware_of_padding)
    else:
        raise ValueError(f"Invalid data collator name: {data_collator_name}")
    
    if return_dataloader:    
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(
                dataset,
                batch_size=split.batch_size,
                shuffle=split.shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=data_collator
            )
            datasets[i] = dataloader
    
    return datasets if len(datasets) > 1 else datasets[0], data_collator