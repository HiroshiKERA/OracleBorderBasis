from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from .models.custom_bart import CustomBartForConditionalGeneration
from .models.encodings.monomial import MonomialEmbedding


def load_model(
    tokenizer: Optional[PreTrainedTokenizer] = None,
    params = None,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "cuda"
    ):

    if params.model == 'bart':
        config = BartConfig(
            encoder_layers=params.num_encoder_layers,
            encoder_attention_heads=params.nhead,
            decoder_layers=params.num_decoder_layers,
            decoder_attention_heads=params.nhead,
            vocab_size=len(tokenizer.vocab),
            d_model=params.d_model,
            encoder_ffn_dim=params.dim_feedforward,
            decoder_ffn_dim=params.dim_feedforward,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_position_embeddings=params.max_sequence_length,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
        model = BartForConditionalGeneration(config)
        
    elif params.model == 'custom_bart':
        config = BartConfig(
            encoder_layers=params.num_encoder_layers,
            encoder_attention_heads=params.nhead,
            decoder_layers=params.num_decoder_layers,
            decoder_attention_heads=params.nhead,
            vocab_size=len(tokenizer.vocab),
            d_model=params.d_model,
            encoder_ffn_dim=params.dim_feedforward,
            decoder_ffn_dim=params.dim_feedforward,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_position_embeddings=params.max_sequence_length,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
        
        post_token_embeddings = nn.ModuleDict({})
        more_position_embeddings = nn.ModuleDict({})

        # Monomial embedding
        if params.monomial_embedding:
            monomial_embedding = MonomialEmbedding(
                config,
                num_coefficients=int(params.field[2:]),
                max_degree=params.max_degree,
                num_variables=params.num_variables,
                token_expander=params.token_expander
            )
        else:
            monomial_embedding = None
        
        if params.token_type_position_encoding:
            post_token_embeddings['token_types'] = nn.Embedding(
                params.num_variables + 2,
                params.d_model, 
            )

        if params.monomial_type_position_encoding:
            post_token_embeddings['monomial_types'] = nn.Embedding( 
                2000,  # TO DO: make it adaptive (# config.num_variables ** config.max_degree  # too large)
                params.d_model,
            )

        
        model = CustomBartForConditionalGeneration(
            config,
            post_token_embeddings=post_token_embeddings,
            more_position_embeddings=more_position_embeddings,
            monomial_embedding=monomial_embedding
            )
    
    else:
        raise ValueError(f'unknown model: {params.model}')

    return model.to(device) 
    