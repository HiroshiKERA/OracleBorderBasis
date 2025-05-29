from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import random
from sage.all import PolynomialRing, QQ, GF, RR, ZZ, matrix, binomial, randint, prod

@dataclass
class PolynomialSamplingConfig:
    """Configuration for polynomial sampling"""
    max_degree: int
    max_num_terms: int
    min_degree: int = 0
    max_coeff: Optional[int] = None  # Used for RR and ZZ
    num_bound: Optional[int] = None  # Used for QQ
    nonzero_instance: bool = False
    
class PolynomialSampler:
    """Generator for random polynomials with specific constraints"""
    
    def __init__(
        self,
        ring: PolynomialRing,
        degree_sampling: str = 'uniform',  # 'uniform' or 'fixed'
        term_sampling: str = 'uniform',    # 'uniform' or 'fixed'
        strictly_conditioned: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize polynomial sampler
        
        Args:
            ring: SageMath polynomial ring
            degree_sampling: How to sample degree ('uniform' or 'fixed')
            term_sampling: How to sample number of terms ('uniform' or 'fixed')
            strictly_conditioned: Whether to strictly enforce conditions
            config: Sampling configuration
        """
        self.ring = ring
        self.field = ring.base_ring()
        self.degree_sampling = degree_sampling
        self.term_sampling = term_sampling
        self.strictly_conditioned = strictly_conditioned
        
        # Set default config if none provided
        default_config = {
            'max_degree': 5,
            'min_degree': 0,
            'max_num_terms': 10,
            'max_coeff': 10 if self.field in (RR, ZZ) else None,
            'num_bound': 10 if self.field == QQ else None,
            'nonzero_instance': False
        }
        config = config or default_config
        self.config = PolynomialSamplingConfig(**config)
        
    def sample(
        self,
        num_samples: int = 1,
        size: Optional[Tuple[int, int]] = None,
        density: float = 1.0,
        matrix_type: Optional[str] = None
    ) -> Union[List[Any], List[matrix]]:
        """
        Generate random polynomial samples
        
        Args:
            num_samples: Number of samples to generate
            size: If provided, generate matrix of polynomials with given size
            density: Probability of non-zero entries in matrix
            matrix_type: Special matrix type (e.g., 'unimodular_upper_triangular')
            
        Returns:
            List of polynomials or polynomial matrices
        """
        if size is not None:
            return [
                self._sample_matrix(size, density, matrix_type)
                for _ in range(num_samples)
            ]
        else:
            return [
                self._sample_polynomial()
                for _ in range(num_samples)
            ]
            
    def _sample_polynomial(self, max_attempts: int = 100) -> Any:
        """Generate a single random polynomial"""
        # Determine degree
        if self.degree_sampling == 'uniform':
            degree = randint(self.config.min_degree, self.config.max_degree)
        else:  # fixed
            degree = self.config.max_degree
            
        # Determine number of terms
        max_possible_terms = binomial(degree + self.ring.ngens(), degree)
        max_terms = min(self.config.max_num_terms, max_possible_terms)
        
        if self.term_sampling == 'uniform':
            num_terms = randint(1, max_terms)
        else:  # fixed
            num_terms = max_terms
            
        # Generate polynomial with retry logic
        for attempt in range(max_attempts):
            p = self._generate_random_polynomial(degree, num_terms)
            
            # Check conditions
            if p == 0 and self.config.nonzero_instance:
                continue
                
            if p.total_degree() < self.config.min_degree:
                continue
                
            if not self.strictly_conditioned:
                break
                
            if p.total_degree() == degree and len(p.monomials()) == num_terms:
                break
                
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f'Failed to generate polynomial satisfying conditions after {max_attempts} attempts'
                )
                
        return p
    
    def _generate_random_polynomial(self, degree: int, num_terms: int) -> Any:
        """Generate a random polynomial with given degree and number of terms"""
        choose_degree = self.degree_sampling == 'uniform'
        
        if self.field == QQ:
            return self.ring.random_element(
                degree=degree,
                terms=num_terms,
                num_bound=self.config.num_bound,
                choose_degree=choose_degree
            )
        elif self.field == RR:
            return self.ring.random_element(
                degree=degree,
                terms=num_terms,
                min=-self.config.max_coeff,
                max=self.config.max_coeff,
                choose_degree=choose_degree
            )
        elif self.field == ZZ:
            return self.ring.random_element(
                degree=degree,
                terms=num_terms,
                x=-self.config.max_coeff,
                y=self.config.max_coeff + 1,
                choose_degree=choose_degree
            )
        else:  # Finite field
            return self.ring.random_element(
                degree=degree,
                terms=num_terms,
                choose_degree=choose_degree
            )
            
    def _sample_matrix(
        self,
        size: Tuple[int, int],
        density: float = 1.0,
        matrix_type: Optional[str] = None,
        max_attempts: int = 100
    ) -> matrix:
        """Generate a matrix of random polynomials"""
        rows, cols = size
        num_entries = prod(size)
        
        # Generate polynomial entries
        entries = []
        for _ in range(num_entries):
            p = self._sample_polynomial(max_attempts)
            # Apply density
            if random.random() >= density:
                p *= 0
            entries.append(p)
            
        # Create matrix
        M = matrix(self.ring, rows, cols, entries)
        
        # Apply special matrix type constraints
        if matrix_type == 'unimodular_upper_triangular':
            for i in range(rows):
                for j in range(cols):
                    if i == j:
                        M[i,j] = 1
                    elif i > j:
                        M[i,j] = 0
                        
        return M

def compute_max_coefficient(poly: Any) -> int:
    """Compute maximum absolute coefficient value in a polynomial"""
    coeffs = poly.coefficients()
    field = poly.base_ring()
    
    if not coeffs:
        return 0
        
    if field == RR:
        return max(abs(c) for c in coeffs)
    else:  # QQ case
        return max(max(abs(c.numerator()), abs(c.denominator())) for c in coeffs)

def compute_matrix_max_coefficient(M: matrix) -> int:
    """Compute maximum absolute coefficient value in a polynomial matrix"""
    return max(compute_max_coefficient(p) for p in M.list())