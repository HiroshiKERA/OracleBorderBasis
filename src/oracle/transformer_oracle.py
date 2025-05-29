from typing import List, Tuple, Optional
from sage.all import PolynomialRing
import torch
from src.oracle.oracle import Oracle
from src.dataset.processors.utils import poly_to_sequence, sequence_to_poly
from src.loader.checkpoint import load_pretrained_bag
from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus
import time

import os
class TransformerOracle(Oracle):
    """
    Oracle that uses a transformer model to predict successful polynomial extensions.
    
    This oracle uses a pre-trained BART model to predict which polynomial-variable
    combinations are likely to lead to successful extensions in the border basis computation.
    """
    def __init__(self, ring: PolynomialRing, model_path: str, leading_term_k: int = 5):
        """
        Initialize the oracle with a polynomial ring and model path.
        
        Args:
            ring: The polynomial ring in which computations take place
            model_path: Path to the pre-trained model and tokenizer
            leading_term_k: Number of leading terms to consider for the leading term oracle
        """
        super().__init__(ring)

        # print current directory
        print(os.getcwd())
        
        # Load the pre-trained model and tokenizer
        self.model_bag = load_pretrained_bag(model_path, from_checkpoint=False)
        self.model = self.model_bag['model']
        self.tokenizer = self.model_bag['tokenizer']
        self.model_config = self.model_bag['config']
        self.model.eval()  # Set to evaluation mode
        
        self.mpp = MonomialProcessorPlus(num_variables=self.model_config.num_variables, 
                                         max_degree=self.model_config.max_degree, 
                                         max_coef=int(self.model_config.field[2:]))

        self.leading_term_k = leading_term_k
        
    def predict(self, V: List, L: List, max_length: int = 200) -> List[Tuple[int, str]]:
        """
        Predict which polynomial extensions are likely to be successful.
        
        Args:
            V: List of polynomials to potentially extend
            L: Current computational universe
            
        Returns:
            List of tuples (poly_idx, var_string) representing successful extensions
        """
        # only keep the self.leading_terms leading terms in V
        V_polynomials = [poly_to_sequence(v) for v in V]

        # for each polynomial in V, only keep the self.leading_terms leading terms
        V_polynomials = [poly.split(' + ') for poly in V_polynomials]
        V_polynomials = [poly[:self.leading_term_k] for poly in V_polynomials]
        V_polynomials = [' + '.join(poly) for poly in V_polynomials]

        V_seqs = ' [SEP] '.join(V_polynomials)

        L_seqs = ' [SEP] '.join([poly_to_sequence(poly) for poly in L])

        input_seq = f'{L_seqs} [BIGSEP] {V_seqs}'
        input_ids = self.mpp([input_seq])
        input_ids = torch.tensor(input_ids, device=self.model.device)
        
        with torch.no_grad():
            generated = self.model.generate(input_ids, attention_mask=None, 
                                            monomial_processor=self.mpp, 
                                            tokenizer=self.tokenizer, 
                                            max_length=max_length,
                                            )

            generated_seq = self.mpp.batch_decode(generated, skip_special_tokens=True, raise_warning=True)[0] # [0] to get the first and only sequence
            predictions = []
            for seq in generated_seq.split('[SEP]'):
                try:
                    poly = sequence_to_poly(seq, self.ring)
                    predictions.append(poly)
                except Exception:
                    continue
            oracle_expansions = [(direction, V_lt) for direction, V_lt in zip(predictions[::2], predictions[1::2])]
        return oracle_expansions
        

        """
        Update the oracle's knowledge based on the results of extensions.
        
        Args:
            successful_extensions: List of (poly_idx, var_string) that were successful
            unsuccessful_extensions: List of (poly_idx, var_string) that were unsuccessful
        """
        # The transformer model is pre-trained and not updated during inference
        pass 