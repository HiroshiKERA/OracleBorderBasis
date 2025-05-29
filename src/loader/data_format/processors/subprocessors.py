from typing import List, Dict, Optional
from .base import BaseProcessor, ProcessTarget
import torch
import itertools as it
import warnings

from enum import Enum
from typing import List, Dict, Tuple
import itertools

class MonomialProcessorPlus(BaseProcessor):
    """Processor that converts monomials to (coef_id, exponents)"""
    def __init__(self, num_variables: int, max_degree: int, max_coef: int, target: ProcessTarget = ProcessTarget.BOTH):
        super().__init__(target)
        self.num_variables = num_variables
        self.max_degree = max_degree
        self.max_coef = max_coef
        self.coef_to_id = self._create_coef_dict()
        self.id_to_coef = {v: k for k, v in self.coef_to_id.items()}
        self.special_to_id = self._create_special_dict()
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
            
    def _create_coef_dict(self) -> Dict[int, int]:
        """Map coefficients to IDs"""
        # Assign IDs from 0 to coefficients from 0 to max_coef 
        return {i: i for i in range(self.max_coef + 1)}
    
    def _create_special_dict(self) -> Dict[str, int]:
        """Map special tokens to IDs"""
        return {
            '[SEP]': 0,
            '[PAD]': 1,
            '<s>': 2,
            '</s>': 3,
            '+': 4,
            '[BIGSEP]': 5
        }
    
    def _process_monomial(self, monomial: str) -> Tuple[int]:
        """Convert monomial to (coef_id, pattern_id)"""
        tokens = monomial.strip().split()
        exponents = []
        for token in tokens:  # (prefix) "C5 E2 E1 E0" or (postfix) "E2 E1 E0 C5"
            if token.startswith('C'):
                coef = int(token[1:])  # "C5" -> 5
            elif token.startswith('E'):
                exponents += [int(token[1:])]  # "E2" -> 2
        
        return [self.coef_to_id[coef]] + exponents
    
    def _process_polynomial(self, polynomial: str) -> List[Tuple[int, int]]:
        """Convert polynomial to list of (coef_id, pattern_id)"""
        if not polynomial.strip():  # If empty string
            return []
        monomials = polynomial.split(' + ')
        special_tokens = [self.special_to_id['+'] for _ in monomials[:-1]] + [self.special_to_id['[SEP]']]
        
        return [[*self._process_monomial(mono), op] for mono, op in zip(monomials, special_tokens)]

    def _process(self, text: str) -> List[List[Tuple[int, int]]]:
        """Process entire text"""
        
        bos = list([0] * (self.num_variables + 1) + [self.special_to_id['<s>']])
        eos = list([0] * (self.num_variables + 1) + [self.special_to_id['</s>']])
        
        components = text.split(' [BIGSEP] ')
        processed = []
        for component in components:
            polys = component.split(' [SEP] ')
            _processed = [self._process_polynomial(poly) for poly in polys]
            _processed = list(it.chain(*_processed))
            _processed[-1][-1] = self.special_to_id['[BIGSEP]']
            processed.extend(_processed)
        
        processed[-1][-1] = self.special_to_id['</s>']
        processed = [bos] + processed # + [eos]

        return processed

    def __call__(self, texts: List[str]) -> List[List[List[Tuple[int, int]]]]:
        """Process multiple texts"""        
        ret = [self._process(text) for text in texts]
        return ret


    def is_valid_monomial(self, texts: List[str]) -> List[bool]:
        
        return [self._is_valid_monomial(monomial_text) for monomial_text in texts]
    
    def _is_valid_monomial(self, monomial: str) -> bool:
        items = monomial.split()
        valid = items[0].startswith('C') and all([t.startswith('E') for t in items[1:-1]]) and items[-1] in self.special_to_id
        
        return valid

    def generation_helper(self, monomial_texts: List[str]) -> List[str]:
        monomials = [self._generative_helper(monomial_text) for monomial_text in monomial_texts]
        return monomials
    
    def _generative_helper(self, monomial_text: str) -> str:
        eos = [0] * (self.num_variables + 1) + [self.special_to_id['</s>']]
        
        valid = self._is_valid_monomial(monomial_text)

        if valid:
            special_token = monomial_text.split()[-1]
            monomial = list(self._process_monomial(monomial_text)) + [self.special_to_id[special_token]]
        else:
            monomial = eos

        return monomial
    
    
    def _decode_monomial_token(self, monomial: torch.Tensor, skip_special_tokens: bool = False) -> str:
        coeff, exponents, special_id = monomial[0].item(), monomial[1:-1], monomial[-1].item()
        
        special_token = self.id_to_special[special_id]
        is_eos = special_token == '</s>'
        
        if special_token == '<s>':
            monomial_text = '' if skip_special_tokens else '<s>'
        else: 
            if is_eos and skip_special_tokens: 
                special_token = ''
            
            monomial_text = ' '.join([f'C{self.id_to_coef[coeff]}'] + [f'E{e}' for e in exponents] + [special_token])
         
        return monomial_text.strip(), is_eos
    
    def decode(self, monomial_tokens: torch.Tensor, skip_special_tokens: bool = False, raise_warning: bool = True) -> List[str]:
        
        decoded_tokens = []
        for monomial in monomial_tokens:
            decoded_token, is_eos = self._decode_monomial_token(monomial, skip_special_tokens=skip_special_tokens)
            decoded_tokens.append(decoded_token)
            
            if is_eos:
                break
        
        # give warning if there is no eos token
        if (not is_eos) and raise_warning:
            warnings.warn(f'Generation ended before EOS token was found. If you are decoding a generated sequence, the max_length might be too small.')
        
        decoded_text = ' '.join(decoded_tokens).strip()
        
        return decoded_text
    
    def batch_decode(self, batch_monomial_tokens: torch.Tensor, skip_special_tokens: bool = True, raise_warning: bool = True) -> List[str]:
        return [self.decode(monomial_tokens, skip_special_tokens=skip_special_tokens, raise_warning=raise_warning) for monomial_tokens in batch_monomial_tokens]