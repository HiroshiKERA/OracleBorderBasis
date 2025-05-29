from typing import Dict, Any, List, Tuple, Optional
import itertools as it
from dataclasses import dataclass
import yaml
from joblib import Parallel, delayed
import numpy as np
from time import time
import re
from pathlib import Path
import signal
from functools import wraps

from sage.all import *
import random  # put this after sage.all as sage.all imports random

import sage.misc.randstate as randstate

from .abstract_problem_generator import AbstractPolynomialProblemGenerator
from ..samplers.polynomial_sampler import PolynomialSampler, PolynomialSamplingConfig
from ..processors.utils import poly_to_sequence, sequence_to_poly
from ...border_basis_lib.improved_border_basis import ImprovedBorderBasisCalculator
from ...border_basis_lib.border_basis import BorderBasisCalculator
from ...border_basis_lib.sanity_checks import OBorderBasisChecker

@dataclass
class ExpansionProblemConfig:
    """Configuration class for expansion generation specific parameters"""
    min_polynomials: int  # min size of F
    max_polynomials: int  # max size of F
    total_degree_bound: int # the maximum degree of polynomials
    degree_bounds: List[int]  # the maximum degree of polynomials in the expansion
    sampling_config: PolynomialSamplingConfig
    index_range: Tuple[int, int] = None  # the range of indices to load

def timeout(seconds=3000, error_message='Function call timed out'):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(seconds=3000)  # 30 second timeout
def compute_border_basis_with_timeout(calculator, F, **kwargs):
    return calculator.compute_border_basis_optimized(F, **kwargs)

class ExpansionGenerator(AbstractPolynomialProblemGenerator):
    """Generator for polynomial expansion problems.
    
    This generator creates problems where the input is a set of polynomials F,
    and the output G contains the expanded form of F after applying certain operations.
    """
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration specific to polynomial expansion generation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # First validate basic polynomial configuration through parent
        config = super()._validate_config(config)
        
        # Validate required sections
        required_sections = ["polynomial", "dataset"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate polynomial configuration
        poly_config = config["polynomial"]
        required_poly = ["max_degree", "max_terms"]
        for param in required_poly:
            if param not in poly_config:
                raise ValueError(f"Missing required polynomial parameter: {param}")
                
        if "min_degree" in poly_config and poly_config["min_degree"] > poly_config["max_degree"]:
            raise ValueError("min_degree cannot be greater than max_degree")
                
        # Validate dataset configuration
        dataset_config = config["dataset"]
        required_dataset = ["min_polynomials", "max_polynomials"]
        for param in required_dataset:
            if param not in dataset_config:
                raise ValueError(f"Missing required dataset parameter: {param}")
                
        if dataset_config["min_polynomials"] > dataset_config["max_polynomials"]:
            raise ValueError("min_polynomials cannot be greater than max_polynomials")
            
        
        return config

    def __init__(self, save_dir: str, config: Dict[str, Any]):
        """
        Initialize polynomial expansion generator.
        
        Args:
            save_dir: Directory to save generated datasets
            config: Configuration dictionary
        """
        # First initialize parent class (this will validate config and setup ring)
        super().__init__(save_dir, config)

        # Retrieve values from the configuration
        self.field_type = self.config["field"]["type"]
        self.field_param = self.config["field"].get("param", None)
        self.num_variables = self.config["ring"]["num_variables"]
        self.term_order = self.config["ring"]["term_order"]

        self.index_range = self.config["dataset"].get("index_range", None)

        # This is the number of leading terms to include in V sequence
        self.leading_terms = self.config["dataset"].get("leading_terms_saved", -1)
        self.include_coefficient = self.config["dataset"].get("include_coefficient", False)
        if self.field_type == "GF":
            var_string = ', '.join([f'x{i}' for i in range(self.num_variables)])
            self.ring = PolynomialRing(GF(self.field_param), var_string, order=self.term_order)
        else:
            raise ValueError(f"Unsupported field type: {self.field_type}")
        
        # Now we can safely use self.config and self.ring to setup problem-specific components
        poly_config = self.config["polynomial"]
        dataset_config = self.config["dataset"]
        
        # Create problem-specific configuration
        self.problem_config = ExpansionProblemConfig(
            min_polynomials=dataset_config["min_polynomials"],
            max_polynomials=dataset_config["max_polynomials"],
            degree_bounds=dataset_config.get("degree_bounds", None),
            total_degree_bound=dataset_config.get("total_degree_bound", None),
            sampling_config=PolynomialSamplingConfig(
                max_degree=poly_config["max_degree"],
                max_num_terms=poly_config["max_terms"],
                min_degree=poly_config.get("min_degree", 0),
                max_coeff=poly_config.get("coefficient", {}).get("max"),
                num_bound=poly_config.get("coefficient", {}).get("bound"),
                nonzero_instance=poly_config.get("nonzero", False),
            )
        )
        
        # Initialize polynomial sampler
        self.sampler = PolynomialSampler(
            ring=self.ring,
            degree_sampling=poly_config.get("degree_sampling", "uniform"),
            term_sampling=poly_config.get("term_sampling", "uniform"),
            strictly_conditioned=poly_config.get("strictly_conditioned", True),
            config=self.problem_config.sampling_config.__dict__
        )

    def load_and_process(self, filepath: str, index_range: Tuple[int, int] = None) -> List[Tuple[str, str]]:
        """
        Load samples from a file and process them.
        
        Args:
            filepath: Path to the input file containing samples
            index_range: Tuple of (start, end) indices to load
        Returns:
            List of (input_sequence, output_sequence) pairs
        """
        processed_samples = []

        # print("Starting to process samples...")
        if index_range is not None:
            print("index_range", index_range)
        
        Fs = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                # check if index_range is not None and i is not in the range
                if index_range is not None and i < index_range[0]:
                    continue
                if index_range is not None and i > index_range[1]:
                    break
                # Split input and output
                F_text, G_text = line.strip().split(' # ')
                
                # Split input polynomials
                F_text = F_text.split(' [SEP] ')
                
                # Convert to SageMath polynomials
                F = [sequence_to_poly(f, self.ring) for f in F_text]
                
                Fs.append(F)

        return Fs 
    
    def generate_sample(self, F: List[Any], k_last_calls: int = None, sort_V: bool = True) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        """
        Generate a single sample with its statistics.
        
        Each sample consists of:
        - Input polynomials F   
        - Output polynomials G (expanded form of F)
        - Statistics about the generation
        
        Returns:
            Tuple containing (input_polynomials, output_expansion, statistics)
        """
        randstate.set_random_seed()
        
        calculator = ImprovedBorderBasisCalculator(self.ring, corollary_23=True, N=100, save_universes=True, verbose=False, sorted_V=sort_V)

        #calculator = BorderBasisCalculator(self.ring, save_universes=True, verbose=False)

        s = time()
        try:
            G, O, timings = compute_border_basis_with_timeout(
                calculator, 
                F,
                use_fast_elimination=True,
                lstabilization_only=False
            )
        except TimeoutError:
            print(f"Timeout occurred for sample, skipping...")
            return [], {"timeout": True}
        #print("time", time() - s)

        processed_samples = []
        
        # record the ratio of non-expansion polynomials to the size of V
        non_expansion_polynomials_ratios = []
        
        # get the last L-stable span call
        last_Lstable_span_dataset = calculator.datasets[-1]
        if k_last_calls is not None:
            last_Lstable_span_dataset = last_Lstable_span_dataset[-k_last_calls:]
        
        for i, sample in enumerate(last_Lstable_span_dataset):

            # F_sequence = ' [SEP] '.join([poly_to_sequence(f) for f in F])
            L = sample[0]
            V = sample[1]

            non_expansion_polynomials = sample[2]
            non_expansion_polynomials_ratios.append(len(non_expansion_polynomials) / len(V))

            expansion_directions = sample[3]
            

            L_sequence = ' [SEP] '.join([poly_to_sequence(l) for l in L])

            # only keep the self.leading_terms leading terms in V
            V_polynomials = [poly_to_sequence(v) for v in V]

            # for each polynomial in V, only keep the self.leading_terms leading terms
            V_polynomials = [poly.split(' + ') for poly in V_polynomials]
            V_polynomials = [poly[:self.leading_terms] for poly in V_polynomials]
            V_polynomials = [' + '.join(poly) for poly in V_polynomials]

            V_sequence = ' [SEP] '.join(V_polynomials)

            # if include_coefficient is false, remove all coefficients from V_sequence, i.e. remove all strings of the form 'Cnumber'
            if not self.include_coefficient:
                V_sequence = ' [SEP] '.join([re.sub(r'C\d+', '', v) for v in V_sequence.split(' [SEP] ')])

            # V_sequence = ' [SEP] '.join([poly_to_sequence(v) for v in V])
            # E_sequence = ' [SEP] '.join([f'{poly_to_sequence(e)} [SEP] {poly_to_sequence(V[i].lt())}' for e, i in zip(successful_expansion_directions, expansion_directions)])
            
            E_sequence = ' [SEP] '.join([f'{poly_to_sequence(self.ring(e))} [SEP] {poly_to_sequence(V[i].lm())}' for e, i in expansion_directions])

            # if expansion_directions is empty, add a dummy element
            if not expansion_directions:
                # compute number of variables   
                num_vars = self.ring.ngens()
                E_sequence = ' '.join(["C1"] + ["E0"] * num_vars + ["[SEP]"] + ["C1"] + ["E0"] * num_vars)

            processed_samples.append(f"{L_sequence} [BIGSEP] {V_sequence} # {E_sequence}")
        
        
        ## randomly choose s samples from processed_samples and the corresponding timings
        #if num_samples_to_pick > 0:
        #    processed_samples = random.sample(processed_samples, num_samples_to_pick)
        
        timings["non_expansion_polynomials_ratios"] = non_expansion_polynomials_ratios
        
        ## Note: timings will not be accurate if num_samples_to_pick > 0.         
        return processed_samples, timings
        
        
    def save_dataset(self, samples: List[str], statistics: List[Any], tag: str, data_tag: Optional[str] = None) -> None:
        """
        Save the generated dataset to a file.
        
        Args:
            samples: List of sample strings (L # V # successful_expansion_directions)
            statistics: List of statistics about the generation
            tag: Tag for the dataset
            data_tag: Optional tag for the dataset
        """
        # Create directory if it doesn't exist
        if self.index_range is not None:
            tag = f"{tag}_index_range={self.index_range[0]}_{self.index_range[1]}"
        else:
            tag = tag
        
        # Construct base path for files
        base_path = self.save_dir / tag

        print("base_path", base_path)
        
        # Save dataset in infix format
        with open(base_path.with_suffix('.infix'), 'w') as f:
            for sample in samples:
                f.write(sample + '\n')


    def _calculate_sample_statistics(self, F: List[Any], G: List[Any], generation_time: float) -> str:
        """
        Calculate statistics for a single generated sample.
        
        Args:
            F: List of input polynomials
            G: List of output polynomials (expanded form)
            generation_time: Time taken to generate this sample
            
        Returns:
            String containing statistics about the sample
        """
        F_stats = self.calculate_polynomial_statistics(F)
        G_stats = self.calculate_polynomial_statistics(G)
        
        return f"generation_time: {generation_time}, num_input_polynomials: {len(F)}, num_output_polynomials: {len(G)}, input_polynomials: {F_stats}, output_polynomials: {G_stats}"
    
    @classmethod
    def from_yaml(cls, yaml_path: str, save_dir: str) -> 'ExpansionGenerator':
        """
        Create a generator instance from a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            save_dir: Directory to save generated datasets
            
        Returns:
            Initialized ExpansionGenerator
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(save_dir, config)
    

    def run(self, 
            num_samples: int,
            n_jobs: int = -1,
            data_tag: Optional[str] = "train",
            verbose: bool = True
            ) -> Tuple[List[Tuple[List[Any], List[Any]]], Dict[str, Any]]:
        
        """
        Generate multiple samples using parallel processing.
        
        Args:
            num_samples: Number of samples to generate
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to display progress information
            
        Returns:
            Tuple containing (list of samples, overall statistics)
        """        
        start_time = time()

        field = self.config["field"]["type"]
        field_param = self.config["field"].get("param", None)
        num_vars = self.config["ring"]["num_variables"]
        
        k_last_calls = self.config["dataset"].get("k_last_calls", None)
        sort_V = self.config["dataset"].get("sort_V", True)
        # num_samples_to_pick = self.config["dataset"].get("num_samples_to_pick", -1)
        
        # Create a more unique name by including degree bounds and other parameters
        degree_bounds = self.problem_config.degree_bounds
        max_degree = self.problem_config.sampling_config.max_degree
        max_terms = self.problem_config.sampling_config.max_num_terms
        total_degree_bound = self.problem_config.total_degree_bound
        # Create a unique identifier for the configuration
        # config_id = f"{field}{field_param}_n={num_vars}_deg={max_degree}_terms={max_terms}"
        # config_id = f"dataset_analysis/{field}{field_param}_n={num_vars}_deg={max_degree}_terms={max_terms}"
        
        
        # if degree_bounds:
        #     bounds_str = '_'.join(map(str, degree_bounds))
        #     config_id += f"_bounds={bounds_str}"
        # if total_degree_bound:
        #     config_id += f"_total={total_degree_bound}"

        source_data_path = str(self.save_dir)
        # replaced `border_basis` with `expansion`
        source_data_path = Path(source_data_path.replace("expansion", "border_basis"))
        
        print(f"Loading dataset from {source_data_path}/{data_tag}.infix")
        Fs = self.load_and_process(f'{source_data_path}/{data_tag}.infix', index_range=self.index_range)
        
        Fs = Fs[:num_samples]
        
        # Generate samples in parallel using joblib
        results = Parallel(
            n_jobs=n_jobs,
            backend="multiprocessing",
            verbose=verbose
        )(
            delayed(self.generate_sample)(F, k_last_calls=k_last_calls, sort_V=sort_V)
            for F in Fs
        )

        results = [result for result in results if 'timeout' not in result[1]]  # remove timeouts

        samples = [item for sublist in results for item in sublist[0] if results[1]]
        timings = []
        for result in results:
            result[1]["step2_lstable_span_improved"] = -1#sum(result[1]["step2_lstable_span_improved"])
            timings.append(result[1])

        # create dict, key is step2_lstable_span_improved and total_efficiency and value is list of values
        overall_stats = {"step2_lstable_span_improved": [], "total_efficiency": [], "non_expansion_polynomials_ratios": []}
        for timing in timings:
            overall_stats["step2_lstable_span_improved"].append(timing["step2_lstable_span_improved"])
            # overall_stats["total_efficiency"].append(timing["total_efficiency"])
            overall_stats["non_expansion_polynomials_ratios"].append(timing["non_expansion_polynomials_ratios"])
        
        
        # Calculate overall statistics
        total_time = time() - start_time
        # overall_stats = self._calculate_overall_statistics(
        #     sample_stats,
        #     total_time=total_time,
        #     num_samples=num_samples
        # )
        # compute average timings across all instanc

        #overall_stats = {f"bb_computation_instance_{i}": timing for i, timing in enumerate(timings)}
        #print("overall_stats", overall_stats)
        
        return samples, overall_stats
    
    def _get_dataset_dir(self, tag: Optional[str] = None) -> Path:
        """
        Get directory path for dataset based on configuration parameters.
        
        Args:
            tag: Optional tag to add to directory name
            
        Returns:
            Path object for the dataset directory
        """
        field = self.config["field"]["type"]
        field_param = self.config["field"].get("param", None)
        num_vars = self.config["ring"]["num_variables"]
        
        # Get configuration parameters
        degree_bounds = self.problem_config.degree_bounds
        max_degree = self.problem_config.sampling_config.max_degree
        max_terms = self.problem_config.sampling_config.max_num_terms
        total_degree_bound = self.problem_config.total_degree_bound
        
        # Create a unique identifier for the configuration
        config_id = f"{field}{field_param}_n={num_vars}_deg={max_degree}_terms={max_terms}"
        if degree_bounds:
            bounds_str = '_'.join(map(str, degree_bounds))
            config_id += f"_bounds={bounds_str}"
        if total_degree_bound:
            config_id += f"_total={total_degree_bound}"
            
        # Add tag if provided
        if tag:
            config_id = f"{config_id}_{tag}"
            
        return self.save_dir / config_id

if __name__ == "__main__":
    pass