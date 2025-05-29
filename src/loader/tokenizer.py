from typing import List, Dict, Union
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import CharDelimiterSplit
from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer as TokenizerBase

# Special tokens for BART
SPECIAL_TOKENS = ['[PAD]', '<s>', '</s>', '[CLS]']
SPECIAL_TOKENS_PLUS = ['[PAD]', '<s>', '</s>', '[CLS]', '[SEP]', '[UNK]', '[BIGSEP]']
SPECIAL_TOKEN_MAP = dict(zip(['pad_token', 'bos_token', 'eos_token', 'cls_token'], SPECIAL_TOKENS))

def set_vocab(
    num_vars: int, 
    field: str = 'QQ',
    max_coeff: int = 100,
    max_degree: int = 10,
    continuous_coefficient: bool = False,
    continuous_exponent: bool = False
) -> List[str]:
    SYMBOLS = [f'x{i}' for i in range(num_vars)]
    OPS = ['+', '*', '^', '/']
    CONSTS = ['[C]']  
    ECONSTS = ['[E]']
    MISC = []
    
    if field in ('RR') and not continuous_coefficient: 
        raise ValueError('RR should use continuous_coefficient=True')
    
    if not continuous_coefficient:
        if field in ('QQ', 'ZZ'):
            CONSTS += [f'C{i}' for i in range(-max_coeff, max_coeff+1)]
        elif field[:2] == 'GF': 
            assert(field[2:].isdigit())
            p = int(field[2:])
            CONSTS += [f'C{i}' for i in range(p)]
        else:
            raise ValueError(f'unknown field: {field}')
    
    if not continuous_exponent:
        ECONSTS += [f'E{i}' for i in range(max_degree+1)]
    
    return SYMBOLS + CONSTS + ECONSTS + OPS + MISC + SPECIAL_TOKENS_PLUS

def set_tokenizer(
    vocab: Union[List[str], Dict[str, int]], 
    max_seq_length: int = 1024
) -> PreTrainedTokenizerFast:
    if type(vocab) is list: 
        vocab = dict(zip(vocab, range(len(vocab))))
    tok = TokenizerBase(WordLevel(vocab))
    tok.pre_tokenizer = CharDelimiterSplit(' ')
    tok.pre_tokenizer = pre_tokenizers.Sequence([CharDelimiterSplit(' ')])
    tok.add_special_tokens(SPECIAL_TOKENS)
    tok.enable_padding()
    tok.no_truncation()
    
    bos_token = SPECIAL_TOKEN_MAP['bos_token']
    eos_token = SPECIAL_TOKEN_MAP['eos_token']
    tok.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[(bos_token, tok.token_to_id(bos_token)), \
                        (eos_token, tok.token_to_id(eos_token))],
    )
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, 
                                        model_max_length=max_seq_length, 
                                        **SPECIAL_TOKEN_MAP)
    return tokenizer

