from src.dataset.processors.utils import poly_to_sequence, sequence_to_poly
from src.border_basis_lib.improved_border_basis import ImprovedBorderBasisCalculator
from src.border_basis_lib.sanity_checks import OBorderBasisChecker
from src.oracle.transformer_oracle import TransformerOracle
import wandb
import yaml
from joblib import Parallel, delayed
from sage.all import *
from typing import Dict, Any, List, Tuple, Optional
import re
import time
from sage.misc.verbose import set_verbose
from sage.rings.polynomial.toy_buchberger import *


class IBBOracleEvaluation:
    """
    Class to evaluate the Improved Border Basis Oracle with different hyperparameters.
    """
    def __init__(self, dataset_path = None, relative_gap = 0.5, absolute_gap = 100, order_ideal_size = 5, oracle_max_calls = 5, n_variables = 5, field = 127, min_universe = 20, filepath = None, model_url = None, oracle = False, total_degree_external = None, max_degree_external = None, fast_gaussian = False, full_universe_expansion = False):
        """
        Initialize the evaluation class with paths for dataset and results.

        Args:
            dataset_path (str): Path to the dataset to be used for evaluation.
            relative_gap (float): Relative gap for the oracle.
            absolute_gap (int): Absolute gap for the oracle.
            order_ideal_size (int): Order ideal size for the oracle.
            oracle_max_calls (int): Maximum number of calls for the oracle.
            
        """
        #multiprocessing.set_start_method('spawn')

        self.dataset_path = dataset_path
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.order_ideal_size = order_ideal_size
        self.oracle_max_calls = oracle_max_calls
        self.min_universe = min_universe
        self.oracle = oracle

        self.filepath = filepath

        self.field = field
        self.n_variables = n_variables

        self.fast_gaussian = fast_gaussian
        self.full_universe_expansion = full_universe_expansion

        self.total_degree_external = total_degree_external
        self.max_degree_external = max_degree_external
        if model_url is not None:
            self.extract_config_from_model_url(model_url)
            self.save_path = model_url
        
        
        

        

        self.ring = PolynomialRing(GF(self.field), 'x', self.n_variables)

        # set up oracle model
        self.oracle_model = TransformerOracle(self.ring, self.save_path, leading_term_k=self.k)

    def extract_config_from_model_url(self, model_url: str):
        """
        Extracts configuration from the model URL.
        """
        self.field, self.k, self.n_variables, self.bounds, self.total, self.degree, self.terms = self.extract_info_from_model_string(model_url)
        if self.total_degree_external is not None and self.total_degree_external != -1:
            self.total = self.total_degree_external
        if self.max_degree_external is not None and self.max_degree_external != -1:
            self.degree = self.max_degree_external

        if self.terms is None:
            self.terms = 10
        if self.degree is None:
            self.degree = 1
        if self.bounds is None:
            self.bounds = "4_"*self.n_variables
            
        self.filepath = self.create_data_string(self.field, self.n_variables, self.degree, self.terms, self.bounds, self.total)
        print(f"Filepath: {self.filepath}")
        
    
    def extract_info_from_model_string(self, model_url: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]]:
        """
        Extracts field, k, n_variables, bounds, and total from the model URL.
        
        Args:
            model_url (str): The model URL string to extract information from.
        
        Returns:
            Tuple containing field, k, n_variables, bounds, and total.
        """
        field_match = re.search(r'_p(\d+)', model_url)
        k_match = re.search(r'_k(\d+)', model_url)
        n_match = re.search(r'_n(\d+)', model_url)
        bounds_match = re.search(r'_bounds((?:\d+_?)+)', model_url)
        total_match = re.search(r'_total(\d+)', model_url)

        field = int(field_match.group(1)) if field_match else None
        k = int(k_match.group(1)) if k_match else None
        n_variables = int(n_match.group(1)) if n_match else None
        bounds = bounds_match.group(1) if bounds_match else None
        total = total_match.group(1) if total_match else None

        degree = re.search(r'_deg(\d+)', model_url).group(1) if re.search(r'_deg(\d+)', model_url) else None
        terms = re.search(r'_terms(\d+)', model_url).group(1) if re.search(r'_terms(\d+)', model_url) else None

        return field, k, n_variables, bounds, total, degree, terms

    def create_data_string(self, field: int, n: int, degree: int, terms: int, bounds: str, total: str) -> str:
        """
        Constructs a dataset name string based on the given parameters.
        
        Args:
            field (int): The field value.
            n (int): The number of variables.
            degree (int): The degree.
            terms (int): The number of terms.
            bounds (str): The bounds string.
            total (str): The total value.
        
        Returns:
            str: The constructed dataset name.
        """
        return f"data/border_basis/GF{field}_n={n}_deg={degree}_terms={terms}_bounds={bounds}total={total}/test.infix"



    def load_dataset(self):
        """
        Load samples from a file and process them.
        
        Returns:
            List of non-border basis to solve
        """
        processed_samples = []

        print("Starting to compute border basis with oracle...")

        
        Fs = []
        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                # Split input and output
                F_text, G_text = line.strip().split(' # ')
                
                # Split input polynomials
                F_text = F_text.split(' [SEP] ')
                
                # Convert to SageMath polynomials
                F = [sequence_to_poly(f, self.ring) for f in F_text]
                
                Fs.append(F)

                if i > 1000:
                    break

        # only keep first 20
        Fs = Fs[:1000]

        return Fs 
    
    def solve_instance(self, F: List[Any]):
        """
        Solve a single instance of the border basis problem.
        """

        #set_verbose(1)
        calculator = ImprovedBorderBasisCalculator(self.ring, corollary_23=True, N=self.full_universe_expansion, save_universes=True, verbose=False, sorted_V=True, relative_gap=self.relative_gap, absolute_gap=self.absolute_gap, order_ideal_size=self.order_ideal_size, oracle_max_calls=self.oracle_max_calls, min_universe=self.min_universe, leading_term_k=self.k, save_path=self.save_path, oracle=self.oracle, oracle_model=self.oracle_model)

        G, O, timings = calculator.compute_border_basis_optimized(
                F,
                use_fast_elimination=self.fast_gaussian,
                lstabilization_only=False
            )

        

        return timings['total_reductions'], timings['total_time'], timings['fallback_to_border_basis'], -1, -1, timings['step2_lstable_span_improved'], timings['gaussian_elimination_times']

    def run(self):
        # print current dir 
        import os 
        print(os.getcwd())

        Fs = self.load_dataset()

        print(f"Loaded {len(Fs)} instances")
        # Generate samples in parallel using joblib
        results = Parallel(
            n_jobs=1,  # Use all available cores
            backend="threading",  # Use threading backend for better compatibility with CUDA
        )(
            delayed(self.solve_instance)(F)
            for F in Fs
        )


        total_reductions_sum = sum(result[0] for result in results)
        total_time_sum = sum(result[1] for result in results)
        fallback_to_border_basis_sum = sum(result[2] for result in results)
        #zero_reduction_steps_sum = sum(result[3][-1] for result in results)
        total_reductions_list = [result[0] for result in results]
        total_time_list = [result[1] for result in results]
        fallback_to_border_basis_list = [result[2] for result in results]
        #zero_reduction_steps_list = [result[3][-1] for result in results]
        #border_gap_list = [result[4] for result in results]
        step2_lstable_span_improved_list = [result[5] for result in results]
        gaussian_elimination_time_list = [result[6] for result in results]


        return total_reductions_sum, total_time_sum, fallback_to_border_basis_sum, -1, total_reductions_list, total_time_list, fallback_to_border_basis_list, -1, -1, step2_lstable_span_improved_list, gaussian_elimination_time_list

        

    def save_results(self, results):
        """
        Save the evaluation results to the specified path.

        Args:
            results (dict): Dictionary containing evaluation results.
        """
        # Implementation to save results
        pass



if __name__ == "__main__":
    # Initialize wandb
    wandb.init()

    # Access the configuration from wandb
    config = wandb.config

    # Initialize the IBBOracleEvaluation class with the parameters from wandb config
    ibb_tester = IBBOracleEvaluation(
        relative_gap=config.relative_gap,
        absolute_gap=config.absolute_gap,
        order_ideal_size=config.order_ideal_size,
        oracle_max_calls=config.oracle_max_calls,
        min_universe=config.min_universe,
        model_url=config.save_path,
        oracle=config.oracle,
        total_degree_external=config.total_degree,
        max_degree_external=config.max_degree,
        fast_gaussian=config.fast_gaussian,
        full_universe_expansion=config.full_universe_expansion
    )

    # Run the evaluation
    total_reductions, total_time, fallback_to_border_basis, zero_reduction_steps, total_reductions_list, total_time_list, fallback_to_border_basis_list, zero_reduction_steps_list, border_gap_list, step2_lstable_span_improved_list, gaussian_elimination_time_list = ibb_tester.run()

    # Log the result
    wandb.log({"total_reductions": total_reductions, "total_time": total_time, "fallback_to_border_basis": fallback_to_border_basis, "zero_reduction_steps": zero_reduction_steps, "total_reductions_list": total_reductions_list, "total_time_list": total_time_list, "fallback_to_border_basis_list": fallback_to_border_basis_list, "zero_reduction_steps_list": zero_reduction_steps_list, "border_gap_list": border_gap_list, "step2_lstable_span_improved_list": step2_lstable_span_improved_list, "gaussian_elimination_time_list": gaussian_elimination_time_list})

    
