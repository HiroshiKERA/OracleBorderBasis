from typing import Tuple, Union

import torch
import torch.nn as nn

from transformers.models.bart.configuration_bart import BartConfig
from .mlp import MLP


class ExponentEmbedding(nn.Embedding):
    """Embedding layer for exponent vectors."""
    def __init__(self, 
                num_variables: int,
                max_degree: int,
                embedding_dim: int,
                ):
        super().__init__(num_variables*(max_degree+1), embedding_dim)
        
        shift = torch.arange(num_variables).long() * (max_degree + 1)
        self.register_buffer('shift', shift)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert exponent vectors to embedding vectors.
        
        Args:
            x: (batch_size, seq_length, num_variables) - Exponent vectors
        Returns:
            embeddings: (batch_size, seq_length, num_variables, embedding_dim)
        """        
        z = (x + self.shift).view(x.shape[0], -1)
        z = super().forward(z)
        z = z.view(x.shape[0], x.shape[1], x.shape[2], -1)

        return z.sum(dim=-2)
    
    
class MonomialEmbedding(nn.Module):
    """Embedding layer for monomials.
    
    Takes monomial representations (coefficient and exponents) and converts them to embeddings.
    Can also expand hidden states for decoding.
    """
    def __init__(self, 
                config: BartConfig, 
                num_coefficients: int, 
                max_degree: int,
                num_variables: int,
                token_expander: str = 'mlp1'):
        super().__init__()
        self.d_model = config.d_model
        self.num_variables = num_variables
        self.tokens_per_unit = num_variables + 2  # coefficient + exponents + operator
        
        # For encoding
        self.coef_embeddings = nn.Embedding(num_coefficients + 1, self.d_model)
        self.exponent_embeddings = ExponentEmbedding(num_variables, max_degree, self.d_model)
        self.sepcial_embedding = nn.Embedding(10, self.d_model)
        
        self.token_expander = nn.Linear(self.d_model, self.d_model * self.tokens_per_unit)
        # assert(token_expander.startswith('mlp'))
        # n = int(token_expander[3:])
        # self.token_expander = MLP([self.d_model] * n + [self.d_model * self.tokens_per_unit],
        #                         activation=nn.GELU())

    def encode(self, monomial_ids: torch.Tensor) -> torch.Tensor:
        """Convert monomial ID sequence to embedding vectors.
        
        Args:
            monomial_ids: (batch_size, seq_length, num_variables + 2) - batch of sequences of (coef_id, exponent_id, special_id)
        Returns:
            embeddings: (batch_size, seq_length, d_model)
        """
        coef_ids = monomial_ids[..., 0]
        exponent_ids = monomial_ids[..., 1:-1]
        special_ids = monomial_ids[..., -1]
        
        assert coef_ids.max().item() < self.coef_embeddings.num_embeddings, \
            f"coef_ids has out-of-bounds index {coef_ids.max().item()} (max allowed: {self.coef_embeddings.num_embeddings - 1})"
        assert exponent_ids.max().item() < self.exponent_embeddings.num_embeddings, \
            f"exponent_ids has out-of-bounds index {exponent_ids.max().item()} (max allowed: {self.exponent_embeddings.num_embeddings - 1})"
        assert special_ids.max().item() < self.sepcial_embedding.num_embeddings, \
            f"special_ids has out-of-bounds index {special_ids.max().item()} (max allowed: {self.sepcial_embedding.num_embeddings - 1})"

        embeddings = self.coef_embeddings(coef_ids) + self.exponent_embeddings(exponent_ids) + self.sepcial_embedding(special_ids)
        
        return embeddings
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand hidden states into n+2 token hidden states.
        
        Args:
            hidden_states: (batch_size, seq_length, d_model)
        Returns:
            expanded_states: (batch_size, seq_length * (num_variables + 2), d_model)
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        """
            IMPORTANT: Why hidden_states[:, :-1]? 
            MonomialProcesssor appends the [bos] monomial token. This is equivalent to adding (n+2) infix tokens. 
            
            [C] [E] ... [E] [S] ([C]: coefficient, [E]: exponent, [S]: special token (e.g [EOS], [SEP], ...))
            
            decoder input (infix token space)   :   1  n+2  n+2  n+2  ...  n+2 -- the first token is [BOS].
            decoder input (monomial token space): [Mb] [M1] [M2] ... [ML] -- (L+1 tokens, 1 -> [Mb] (equivalent to n+2 tokens))
            decoder output ( ... )              : [M1] [M2] ... [ML] [Me] -- (L+1 tokens)
            labels (monomial token space)       : [A1] [A2] ... [AL] [Ae] -- (L+1 tokens)
            labels (infix token space)          : n+2  n+2  ... n+2   0   -- (L+1 tokens, [Ae] -> 0, as [AL] can include [EOS] information)
            
        """
        # Expand each token into n+2 tokens
        expanded = self.token_expander(hidden_states[:, :-1])  # (batch, seq, d_model * (n+2))


        # Reshape
        expanded = expanded.view(
            batch_size,
            -1,
            self.tokens_per_unit,
            self.d_model
        )
        
        # Expand along sequence dimension
        expanded = expanded.view(
            batch_size,
            -1,
            self.d_model
        )
        
        return expanded
    
    def forward(self, x: torch.Tensor, mode: str = 'encode') -> torch.Tensor:
        """Unified interface for encoding/decoding.
        
        Args:
            x: Input tensor
            mode: 'encode' or 'decode'
        Returns:
            For encoding: embedding vectors
            For decoding: expanded hidden states
        """
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")