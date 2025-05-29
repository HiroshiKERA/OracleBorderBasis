import torch
import itertools as it
from .standard import StandardDataCollator

class MonomialCollator(StandardDataCollator):
    def _pad_sequences(self, sequences, padding_value=0):
        # """Padding sequences"""

        max_length = max(len(seq) for seq in sequences)
                
        num_tokens_per_unit = len(sequences[0][0])

        padding = [0] * (num_tokens_per_unit)

        padded_sequences = []
        attention_masks = []

        # Convert to tensors (+2 for bos)
        batch_size = len(sequences)
        sequence_length = max_length # + 2  # +2 for bos
        
        # Prepare result tensors
        monomial_ids = torch.zeros(batch_size, sequence_length, num_tokens_per_unit, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.long)

        for k, seq in enumerate(sequences):  # seq: poly [SEP] poly [SEP] poly ...
            flattened = seq
            cur_length = len(flattened)
            padding_length = max_length - cur_length
            padded = flattened + [padding] * padding_length
            mask = [1] * cur_length + [0] * padding_length
            
            monomial_ids[k, :, 0] = torch.tensor([item[0] for item in padded])  # coeff
            monomial_ids[k, :, 1:-1] = torch.tensor([item[1:-1] for item in padded])  # exponents
            monomial_ids[k, :, -1] = torch.tensor([item[-1] for item in padded])  # special
            
            attention_mask[k, :] = torch.tensor(mask)

        return {
            'monomial_ids': monomial_ids,  # shape: (batch_size, sequence_length, 3)
            'attention_mask': attention_mask  # shape: (batch_size, sequence_length)
        }
    
    def __call__(self, batch):
        """Process batch
        Args:
            batch: Batch obtained from dataset
        
        Returns:
            batch_dict: Dictionary to pass to model
        """
        batch_dict = {}
        attributes = batch[0].keys()
        
        assert(self.tokenizer is not None)        
        assert('input_monomial_ids' in attributes)
        assert('target_monomial_ids' in attributes)
        
        for attribute in attributes:
            attribute_batch = [item[attribute] for item in batch]
            
            if attribute == 'input':
                pass
                
            elif attribute == 'target':
                targets = self.tokenizer(attribute_batch, padding='longest', return_tensors='pt')
                labels = targets['input_ids'][:, 1:].contiguous()

                if not self.aware_of_padding:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                batch_dict['labels'] = labels
                
            elif 'monomial_ids' in attribute:
                # Process monomial_ids
                prefix = 'decoder_' if attribute.startswith('target_') else ''
                padded = self._pad_sequences(attribute_batch)
                
                batch_dict[f'{prefix}input_ids'] = padded['monomial_ids']
                batch_dict[f'{prefix}attention_mask'] = padded['attention_mask']
                
            else:
                if attribute.startswith('target_'):
                    attribute = 'decoder_' + attribute[7:]
                batch_dict[attribute] = self._pad_sequences(attribute_batch, padding_value=0)[:, :-1]
        
        return batch_dict