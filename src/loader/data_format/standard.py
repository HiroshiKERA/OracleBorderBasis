# loader/data_format/standard.py
from typing import List, Optional
import torch 

from .base import DictDatasetBase, DataCollatorBase
from .processors.base import BaseProcessor

class StandardDataset(DictDatasetBase):
    def __init__(self, 
                 input_texts: List[str], 
                 target_texts: List[str], 
                 processor: Optional[BaseProcessor] = None,
                 subprocessors: Optional[dict[str, BaseProcessor]] = None,
                 **kwargs):
        self.processor = processor
        self.subprocessors = subprocessors if subprocessors is not None else {}
        super().__init__(input_texts, target_texts, **kwargs)
    
    def setup(self, **kwargs):
        """preprocess data"""
        if self.processor is not None:
            self.input_texts = self.processor(self.input_texts, is_target=False)
            self.target_texts = self.processor(self.target_texts, is_target=True)


        subattributes = {}
        for processor_name in self.subprocessors:
            assert(processor_name not in subattributes)
            
            subprocessor = self.subprocessors[processor_name]
            input_attribute = subprocessor(self.input_texts)
            target_attribute = subprocessor(self.target_texts)
            
            subattributes['input_' + processor_name] = input_attribute
            subattributes['target_' + processor_name] = target_attribute
            
        self.subattributes = subattributes
                
    def __getitem__(self, idx):
        item = {
            "input": self.input_texts[idx],
            "target": self.target_texts[idx],
        }
        
        if self.subattributes:        
            subitem = dict(zip(
                self.subattributes.keys(),
                [subattr[idx] for subattr in self.subattributes.values()]
            ))
            item = {**item, **subitem}
        
        return item

    def __len__(self):
        return len(self.input_texts)

class StandardDataCollator(DataCollatorBase):
    def setup(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        self.aware_of_padding = kwargs.get('aware_of_padding', False)
    def _pad_sequences(self, sequences, padding_value=0):
        """Pad sequences and convert to tensor"""
        # Calculate max length
        max_length = max(len(seq) for seq in sequences)
        
        # Apply padding
        padded_sequences = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            padded_seq = seq + [padding_value] * padding_length
            padded_sequences.append(padded_seq)
        
        # '+2' from bos/eos tokens.
        padded = torch.zeros(len(sequences), max_length + 2, dtype=torch.long)
        padded[:, 1:max_length+1] = torch.tensor(padded_sequences)
        
        return padded
    
    def __call__(self, batch):
        
        batch_dict = {}
        
        attributes = batch[0].keys()
        
        if self.tokenizer is None:
            for attribute in attributes:
                attribute_batch = [item[attribute] for item in batch]
                batch_dict[attribute] = attribute_batch

            return batch_dict
            
        for attribute in attributes: 
            attribute_batch = [item[attribute] for item in batch]
            
            if attribute == 'input':
                inputs = self.tokenizer(attribute_batch, padding='longest', return_tensors='pt')
                batch_dict['input_ids'] = inputs['input_ids']
                batch_dict['attention_mask'] = inputs['attention_mask']
                
            elif attribute == 'target':
                targets = self.tokenizer(attribute_batch, padding='longest', return_tensors='pt')
                batch_dict['decoder_input_ids'] = targets['input_ids'][:, :-1].contiguous()
                batch_dict['decoder_attention_mask'] = targets['attention_mask'][:, :-1].contiguous()
                
                labels = targets['input_ids'][:, 1:].contiguous()
                if not self.aware_of_padding:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                batch_dict['labels'] = labels
                
            else:
                if attribute.startswith('target_'):
                    attribute = 'decoder_' + attribute[7:]
                batch_dict[attribute] = self._pad_sequences(attribute_batch, padding_value=0)
                
        return batch_dict               
                    
