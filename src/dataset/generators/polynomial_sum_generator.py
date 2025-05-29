from typing import Dict, Any, List, Tuple, Optional
import random
from dataclasses import dataclass
import yaml
from joblib import Parallel, delayed
import numpy as np
from time import time
import sage.misc.randstate as randstate

from .abstract_problem_generator import AbstractPolynomialProblemGenerator
from ..samplers.polynomial_sampler import PolynomialSampler, PolynomialSamplingConfig
from ..processors.utils import poly_to_sequence

@dataclass
class PolynomialSumProblemConfig:
    """Configuration class for polynomial sum generation specific parameters"""
    min_polynomials: int
    max_polynomials: int
    sampling_config: PolynomialSamplingConfig

class PolynomialSumGenerator(AbstractPolynomialProblemGenerator):
    """Generator for polynomial sum problems.
    
    This generator creates problems where the input is a list of polynomials F,
    and the output G contains the sum of F (i.e., G = sum(F)).
    """
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration specific to polynomial sum generation.
        
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
        Initialize polynomial sum generator.
        
        Args:
            save_dir: Directory to save generated datasets
            config: Configuration dictionary
        """
        # First initialize parent class (this will validate config and setup ring)
        super().__init__(save_dir, config)
        
        # Now we can safely use self.config and self.ring to setup problem-specific components
        poly_config = self.config["polynomial"]
        dataset_config = self.config["dataset"]
        
        # Create problem-specific configuration
        self.problem_config = PolynomialSumProblemConfig(
            min_polynomials=dataset_config["min_polynomials"],
            max_polynomials=dataset_config["max_polynomials"],
            sampling_config=PolynomialSamplingConfig(
                max_degree=poly_config["max_degree"],
                max_num_terms=poly_config["max_terms"],
                min_degree=poly_config.get("min_degree", 0),
                max_coeff=poly_config.get("coefficient", {}).get("max"),
                num_bound=poly_config.get("coefficient", {}).get("bound"),
                nonzero_instance=poly_config.get("nonzero", False)
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


    def generate_sample(self) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        """
        Generate a single sample with its statistics.
        
        Each sample consists of:
        - Input polynomials F
        - Output polynomials G (sum of F)
        - Statistics about the generation
        
        Returns:
            Tuple containing (input_polynomials, output_sum, statistics)
        """
        # Set random seed for SageMath's random state
        randstate.set_random_seed()
        
        # Choose number of polynomials for this sample
        num_polys = random.randint(
            self.problem_config.min_polynomials,
            self.problem_config.max_polynomials
        )
        
        # Start timing the generation
        start_time = time()
        
        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)
        
        # Generate sum for output
        G = []
        current_sum = self.ring(0)
        for f in F:
            current_sum += f
            G.append(current_sum)

        # Calculate statistics for this sample
        stats = self._calculate_sample_statistics(F, G, time() - start_time)
            
        return F, G, stats
    
    def _calculate_sample_statistics(self, F: List[Any], G: List[Any], generation_time: float) -> Dict[str, Any]:
        """
        Calculate statistics for a single generated sample.
        
        Args:
            F: List of input polynomials
            G: List of output polynomials (partial sums)
            generation_time: Time taken to generate this sample
            
        Returns:
            Dictionary containing statistics about the sample
        """
        F_stats = self.calculate_polynomial_statistics(F)
        G_stats = self.calculate_polynomial_statistics(G)
        
        return {
            "generation_time": generation_time,
            "num_input_polynomials": len(F),
            "input_polynomials": F_stats,
            "output_polynomials": G_stats
        }
    
    def run(self, 
            num_samples: int,
            n_jobs: int = -1,
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
        
        # Generate samples in parallel using joblib
        results = Parallel(
            n_jobs=n_jobs,
            backend="multiprocessing",
            verbose=verbose
        )(
            delayed(self.generate_sample)()
            for _ in range(num_samples)
        )
        
        # Unzip the results
        F_list, G_list, sample_stats = zip(*results)
        
        # Calculate overall statistics
        total_time = time() - start_time
        overall_stats = self._calculate_overall_statistics(
            sample_stats,
            total_time=total_time,
            num_samples=num_samples
        )
        
        return list(zip(F_list, G_list)), overall_stats
    
    def _calculate_sample_statistics(self, F: List[Any], G: List[Any], generation_time: float) -> Dict[str, Any]:
        """Calculate statistics for a single generated sample."""
        F_tokens = [str(poly_to_sequence(p)).split() for p in F]
        G_tokens = [str(poly_to_sequence(p)).split() for p in G]
        
        token_counts = {
            "F_token_counts": [len(tokens) for tokens in F_tokens],
            "G_token_counts": [len(tokens) for tokens in G_tokens]
        }
        
        system_stats = self.calculate_system_statistics(F, G, token_counts)
        system_stats["generation_time"] = generation_time
        
        return system_stats
    
    def _calculate_overall_statistics(self, 
                                    sample_stats: List[Dict[str, Any]],
                                    total_time: float,
                                    num_samples: int) -> Dict[str, Any]:
        """Calculate overall statistics from all generated samples."""
        stats = {
            "total_time": total_time,
            "samples_per_second": num_samples / total_time,
            "num_samples": num_samples,
            "avg_generation_time": float(np.mean([s["generation_time"] for s in sample_stats])),
            "std_generation_time": float(np.std([s["generation_time"] for s in sample_stats])),
            "min_generation_time": float(np.min([s["generation_time"] for s in sample_stats])),
            "max_generation_time": float(np.max([s["generation_time"] for s in sample_stats]))
        }
        
        return stats
    
    
    @classmethod
    def from_yaml(cls, yaml_path: str, save_dir: str) -> 'PolynomialProdGenerator':
        """
        Create a generator instance from a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            save_dir: Directory to save generated datasets
            
        Returns:
            Initialized PolynomialProdGenerator
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        save_path = save_dir / "config.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
            
        return cls(save_dir, config)
    
