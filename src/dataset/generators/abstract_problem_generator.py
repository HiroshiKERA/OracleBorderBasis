from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from sage.all import PolynomialRing, QQ, GF, RR
import json 
import numpy as np
import yaml 

from .dataset_generator import DatasetGenerator
from ..processors.utils import poly_to_sequence, sequence_to_poly
from ...misc.utils import convert_sage_types

class AbstractPolynomialProblemGenerator(DatasetGenerator):
    """Base class for polynomial problem generators"""
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for polynomial problems.

        Required config structure:
        - field:
            type: str ("QQ", "GF", "RR")
            param: int (only for GF, prime characteristic)
        - ring:
            num_variables: int
            term_order: str
        """
        if not config:
            raise ValueError("Configuration must not be None or empty")
            
        # Check field configuration
        if "field" not in config:
            raise ValueError("Missing required section: field")
            
        field_config = config["field"]
        if "type" not in field_config:
            raise ValueError("Missing required parameter: field.type")
            
        if field_config["type"] not in ["QQ", "GF", "RR"]:
            raise ValueError(f"Unsupported field type: {field_config['type']}")
            
        if field_config["type"] == "GF" and ("param" not in field_config or not field_config["param"]):
            raise ValueError("field.param required for GF field type")

        # Check ring configuration
        if "ring" not in config:
            raise ValueError("Missing required section: ring")
            
        ring_config = config["ring"]
        required_ring = ["num_variables", "term_order"]
        for param in required_ring:
            if param not in ring_config:
                raise ValueError(f"Missing required parameter: ring.{param}")
            
        if ring_config["num_variables"] < 1:
            raise ValueError("ring.num_variables must be positive")

        return config
    
    def _setup_ring(self) -> PolynomialRing:
        """
        Setup polynomial ring based on configuration.
        
        Returns:
            SageMath polynomial ring
        """
        # Setup base field
        field_config = self.config["field"]
        if field_config["type"] == "QQ":
            field = QQ
        elif field_config["type"] == "GF":
            field = GF(field_config["param"])
        else:  # RR
            field = RR
            
        # Create polynomial ring
        ring_config = self.config["ring"]
        return PolynomialRing(
            field,
            ring_config["num_variables"],
            'x',
            order=ring_config["term_order"]
        )
    
    def __init__(self, save_dir: str, config: Dict[str, Any]):
        """
        Initialize polynomial problem generator.
        
        Args:
            save_dir: Directory to save generated datasets
            config: Configuration dictionary with polynomial-specific parameters
        """
        # First call parent class initialization
        super().__init__(save_dir, config)
        
        # Validate configuration (this sets self.config in parent class)
        validated_config = self._validate_config(config)
        self.config = validated_config
        
        # Setup ring based on validated configuration
        self.ring = self._setup_ring()
        
    def _save_raw(self, samples: List[Tuple[List[Any], List[Any]]], base_path: Path) -> None:
        """
        Save polynomial systems in text format, which is handy for SageMath.
        Format: f1 | f2 | ... | fs # g1 | g2 | ... | gt
        
        Args:
            samples: List of (F, G) pairs where F and G are lists of polynomials
            base_path: Base path for the output file
        """
        with open(f"{base_path}.raw", "w") as f:
            for F, G in samples:
                F_str = " | ".join(str(p) for p in F)
                G_str = " | ".join(str(p) for p in G)
                f.write(f"{F_str} # {G_str}\n")

    def _save_infix(self, samples, base_path):
        """
        Save polynomial systems in infix format and collect token statistics.
        Format: each line contains "F_tokens # G_tokens" where F_tokens and G_tokens
        are space-separated tokens with [SEP] between polynomials.

        Args:
            samples: List of (F, G) pairs where F and G are lists of polynomials
            base_path: Base path for the output file

        Returns:
            Dictionary containing token counts for F and G systems
        """
        token_counts_F = []
        token_counts_G = []
        
        with open(f"{base_path}.infix", "w") as f:
            for F, G in samples:
                F_tokens = [str(poly_to_sequence(p)).split() for p in F]
                F_str = " [SEP] ".join(" ".join(tokens) for tokens in F_tokens)
                token_counts_F.extend(len(tokens) for tokens in F_tokens)
                
                G_tokens = [str(poly_to_sequence(p)).split() for p in G]
                G_str = " [SEP] ".join(" ".join(tokens) for tokens in G_tokens)
                token_counts_G.extend(len(tokens) for tokens in G_tokens)
                
                f.write(f"{F_str} # {G_str}\n")
        
        return {
            "F_token_counts": token_counts_F,
            "G_token_counts": token_counts_G
        }

                

    def _save_json(self, samples: List[Tuple[List[Any], List[Any]]], base_path: Path) -> None:
        """
        Save polynomial systems in JSON format.
        
        Args:
            samples: List of (F, G) pairs where F and G are lists of polynomials
            base_path: Base path for the output file
        """
        data = []
        for F, G in samples:
            data.append({
                "input": [str(p) for p in F],
                "output": [str(p) for p in G]
            })
        
        with open(f"{base_path}.json", "w") as f:
            json.dump(data, f, indent=2)
                
                
    def _get_dataset_dir(self, tag: Optional[str] = None) -> Path:
        """
        Get directory path for dataset based on field and number of variables.
        
        Args:
            tag: Optional tag to add to directory name
            
        Returns:
            Path object for the dataset directory
        """
        field_config = self.config["field"]
        ring_config = self.config["ring"]
        
        # Construct base directory name
        if field_config["type"] == "GF":
            field_str = f"GF{field_config['param']}"
        else:
            field_str = field_config["type"]
            
        dir_name = f"{field_str}_n={ring_config['num_variables']}"
        
        # Add tag if provided
        if tag:
            dir_name = f"{dir_name}_{tag}"
            
        return self.save_dir / dir_name
        
    def save_dataset(self, 
                        samples: List[Tuple[List[Any], List[Any]]], 
                        statistics: Dict[str, Any],
                        tag: str = "train", 
                        data_tag: Optional[str] = None) -> None:
        """
        Save the generated dataset and its statistics.
        
        Args:
            samples: List of (F, G) pairs
            statistics: Dictionary containing dataset statistics
            tag: Dataset tag (e.g., "train", "test", "valid")
            data_tag: Optional tag for the dataset directory
        """
        # Create directory if it doesn't exist
        # save_dir = self._get_dataset_dir(data_tag)
        # save_dir.mkdir(parents=True, exist_ok=True)
        
        # # Construct base path for files
        base_path = self.save_dir / tag
        
        # Save dataset in infix format and get token counts
        token_counts = self._save_infix(samples, base_path)
        
        # Get the first sample's F and G for statistics
        # (using the first sample is sufficient as all samples have similar structure)
        if samples:
            F, G = samples[0]
            system_stats = self.calculate_system_statistics(F, G, token_counts)
            statistics.update(system_stats)
        
        # Save statistics
        with open(f"{base_path}_stats.yaml", "w") as f:
            yaml.dump(statistics, f)

    def calculate_polynomial_statistics(self, polys: List[Any]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of polynomials.
        
        Args:
            polys: List of polynomials
            
        Returns:
            Dictionary containing statistical information about the polynomial system
        """
        # Basic statistics
        num_polys = len(polys)
        
        if num_polys == 0:
            return {
                "num_polynomials": 0,
                "total_degree": 0,
                "total_terms": 0
            }
        
        # Calculate degrees
        degrees = [p.total_degree() for p in polys]
        
        # Calculate number of terms
        num_terms = [len(p.monomials()) for p in polys]
        
        # Calculate coefficient statistics
        coeffs = []
        for p in polys:
            if self.config["field"]["type"] == "QQ":
                # For QQ, consider both numerators and denominators
                coeffs.extend([abs(c.numerator()) for c in p.coefficients()])
                coeffs.extend([abs(c.denominator()) for c in p.coefficients()])
            elif self.config["field"]["type"] == "RR":
                # For RR, take absolute values
                coeffs.extend([abs(c) for c in p.coefficients()])
            else:  # GF
                # For finite fields, just take the values
                coeffs.extend([int(c) for c in p.coefficients()])
        
        stats = {
            # System size statistics
            "num_polynomials": num_polys,
            "total_degree": sum(degrees),
            "total_terms": sum(num_terms),
            
            # Degree statistics
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "avg_degree": float(np.mean(degrees)),
            "std_degree": float(np.std(degrees)),
            
            # Term count statistics
            "max_terms": max(num_terms),
            "min_terms": min(num_terms),
            "avg_terms": float(np.mean(num_terms)),
            "std_terms": float(np.std(num_terms)),
            
            # Coefficient statistics
            "max_coeff": max(coeffs) if coeffs else 0,
            "min_coeff": min(coeffs) if coeffs else 0,
            "avg_coeff": float(np.mean(coeffs)) if coeffs else 0,
            "std_coeff": float(np.std(coeffs)) if coeffs else 0,
            
            # Additional system properties
            "density": float(sum(num_terms)) / (num_polys * (1 + max(degrees)) ** self.ring.ngens())
        }
        
        return stats

    def calculate_token_statistics(self, token_counts: List[int]) -> Dict[str, Any]:
        """
        Calculate statistics about number of tokens from pre-computed token counts.
        
        Args:
            token_counts: List of token counts
            
        Returns:
            Dictionary containing token statistics
        """
        if not token_counts:
            return {
                "max_tokens": 0,
                "min_tokens": 0,
                "avg_tokens": 0,
                "median_tokens": 0,
                "std_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "avg_tokens": float(np.mean(token_counts)),
            "median_tokens": float(np.median(token_counts)),
            "std_tokens": float(np.std(token_counts)),
            "total_tokens": sum(token_counts)
        }

    def calculate_system_statistics(self, F: List[Any], G: List[Any], token_counts: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Calculate statistics for input and output polynomial systems.
        
        Args:
            F: List of input polynomials
            G: List of output polynomials
            token_counts: Dictionary containing token counts for F and G systems
            
        Returns:
            Dictionary containing statistical information about both systems
        """
        # Calculate polynomial statistics
        F_poly_stats = self.calculate_polynomial_statistics(F)
        G_poly_stats = self.calculate_polynomial_statistics(G)
        
        # Calculate token statistics using pre-computed counts
        F_token_stats = self.calculate_token_statistics(token_counts["F_token_counts"])
        G_token_stats = self.calculate_token_statistics(token_counts["G_token_counts"])
        
        stats = {
            "input_system": {
                "polynomials": F_poly_stats,
                "tokens": F_token_stats
            },
            "output_system": {
                "polynomials": G_poly_stats,
                "tokens": G_token_stats
            },
            "total_polynomials": F_poly_stats["num_polynomials"] + G_poly_stats["num_polynomials"],
            "total_terms": F_poly_stats["total_terms"] + G_poly_stats["total_terms"],
            "total_tokens": F_token_stats["total_tokens"] + G_token_stats["total_tokens"]
        }
        
        return convert_sage_types(stats)
    
    
class AbstractArithmeticProblemGenerator(DatasetGenerator):
    """Base class for arithmetic problem generators"""
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for arithmetic problems.

        Required config structure:
        - field:
            type: str ("GF")
            param: int (prime characteristic)
        """
        if not config:
            raise ValueError("Configuration must not be None or empty")
            
        # Check field configuration
        if "field" not in config:
            raise ValueError("Missing required section: field")
            
        field_config = config["field"]
        if "type" not in field_config:
            raise ValueError("Missing required parameter: field.type")
            
        if field_config["type"] != "GF":
            raise ValueError(f"Only GF field type is supported for arithmetic problems")
            
        if "param" not in field_config or not field_config["param"]:
            raise ValueError("field.param required for GF field type")

        return config
    
    def _setup_field(self) -> Any:
        """
        Setup base field based on configuration.
        
        Returns:
            SageMath finite field
        """
        field_config = self.config["field"]
        return GF(field_config["param"])
    
    def __init__(self, save_dir: str, config: Dict[str, Any]):
        """
        Initialize arithmetic problem generator.
        
        Args:
            save_dir: Directory to save generated datasets
            config: Configuration dictionary with arithmetic-specific parameters
        """
        # First call parent class initialization
        super().__init__(save_dir, config)
        
        # Validate configuration
        validated_config = self._validate_config(config)
        self.config = validated_config
        
        # Setup field based on validated configuration
        self.field = self._setup_field()

    def _save_raw(self, samples: List[Tuple[List[Any], List[Any]]], base_path: Path) -> None:
        """
        Save number sequences in text format.
        Format: f1 f2 ... fs # g1 g2 ... gt
        
        Args:
            samples: List of (F, G) pairs where F and G are lists of numbers
            base_path: Base path for the output file
        """
        with open(f"{base_path}.raw", "w") as f:
            for F, G in samples:
                F_str = " ".join(str(n) for n in F)
                G_str = " ".join(str(n) for n in G)
                f.write(f"{F_str} # {G_str}\n")

    def _save_infix(self, samples, base_path):
        """
        Save number sequences in infix format and collect token statistics.
        Format: each line contains "F_tokens # G_tokens" where F_tokens and G_tokens
        are space-separated numbers.

        Args:
            samples: List of (F, G) pairs where F and G are lists of numbers
            base_path: Base path for the output file

        Returns:
            Dictionary containing token counts for F and G sequences
        """
        token_counts_F = []
        token_counts_G = []
        
        with open(f"{base_path}.infix", "w") as f:
            for F, G in samples:
                F_str = " ".join(f'C{n}' for n in F)
                token_counts_F.append(len(F))
                
                G_str = " ".join(f'C{n}' for n in G)
                token_counts_G.append(len(G))
                
                f.write(f"{F_str} # {G_str}\n")
                
        return {
            "F_token_counts": token_counts_F,
            "G_token_counts": token_counts_G
        }

    def calculate_sequence_statistics(self, sequence: List[Any]) -> Dict[str, Any]:
        """
        Calculate statistics for a sequence of numbers.
        
        Args:
            sequence: List of numbers
            
        Returns:
            Dictionary containing statistical information about the sequence
        """
        if not sequence:
            return {
                "num_elements": 0,
                "max_value": 0,
                "min_value": 0,
                "avg_value": 0,
                "std_value": 0
            }
            
        values = [int(n) for n in sequence]
        return {
            "num_elements": len(values),
            "max_value": max(values),
            "min_value": min(values),
            "avg_value": float(np.mean(values)),
            "std_value": float(np.std(values))
        }

    def calculate_token_statistics(self, token_counts: List[int]) -> Dict[str, Any]:
        """
        Calculate statistics about token counts.
        
        Args:
            token_counts: List of token counts
            
        Returns:
            Dictionary containing statistical information about token counts
        """
        if not token_counts:
            return {
                "max_tokens": 0,
                "min_tokens": 0,
                "avg_tokens": 0,
                "median_tokens": 0,
                "std_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "avg_tokens": float(np.mean(token_counts)),
            "median_tokens": float(np.median(token_counts)),
            "std_tokens": float(np.std(token_counts)),
            "total_tokens": sum(token_counts)
        }

    def calculate_system_statistics(self, F: List[Any], G: List[Any], token_counts: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Calculate statistics for input and output number sequences.
        
        Args:
            F: List of input numbers
            G: List of output numbers
            token_counts: Dictionary containing token counts for F and G sequences
            
        Returns:
            Dictionary containing statistical information about both sequences
        """
        # Calculate sequence statistics
        F_seq_stats = self.calculate_sequence_statistics(F)
        G_seq_stats = self.calculate_sequence_statistics(G)
        
        # Calculate token statistics using pre-computed counts
        F_token_stats = self.calculate_token_statistics(token_counts["F_token_counts"])
        G_token_stats = self.calculate_token_statistics(token_counts["G_token_counts"])
        
        stats = {
            "input_sequence": {
                "sequence": F_seq_stats,
                "tokens": F_token_stats
            },
            "output_sequence": {
                "sequence": G_seq_stats,
                "tokens": G_token_stats
            },
            "total_elements": F_seq_stats["num_elements"] + G_seq_stats["num_elements"],
            "total_tokens": F_token_stats["total_tokens"] + G_token_stats["total_tokens"]
        }
        
        return convert_sage_types(stats)

    def _get_dataset_dir(self, tag: Optional[str] = None) -> Path:
        """
        Get directory path for dataset based on field configuration.
        
        Args:
            tag: Optional tag to add to directory name
            
        Returns:
            Path object for the dataset directory
        """
        field_config = self.config["field"]
        
        # Construct base directory name
        field_str = f"GF{field_config['param']}"
        dir_name = field_str
        
        # Add tag if provided
        if tag:
            dir_name = f"{dir_name}_{tag}"
            
        return self.save_dir / dir_name

    def save_dataset(self, 
                    samples: List[Tuple[List[Any], List[Any]]], 
                    statistics: Dict[str, Any],
                    tag: str = "train", 
                    data_tag: Optional[str] = None) -> None:
        """
        Save the generated dataset and its statistics.
        
        Args:
            samples: List of (F, G) pairs
            statistics: Dictionary containing dataset statistics
            tag: Dataset tag (e.g., "train", "test", "valid")
            data_tag: Optional tag for the dataset directory
        """
        # Create directory if it doesn't exist
        save_dir = self._get_dataset_dir(data_tag)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct base path for files
        base_path = save_dir / tag
        
        # Save dataset in infix format and get token counts
        token_counts = self._save_infix(samples, base_path)
        
        # Get the first sample's F and G for statistics
        # (using the first sample is sufficient as all samples have similar structure)
        if samples:
            F, G = samples[0]
            system_stats = self.calculate_system_statistics(F, G, token_counts)
            statistics.update(system_stats)
        
        # Save statistics
        with open(f"{base_path}_stats.yaml", "w") as f:
            yaml.dump(statistics, f)