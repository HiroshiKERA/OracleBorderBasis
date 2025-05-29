from typing import Dict, Any, Optional
from dataclasses import dataclass
from sage.all import GF, RR, QQ, ZZ

@dataclass
class NumberSamplingConfig:
    """Number sampling configuration"""
    max_value: Optional[int] = None  # Maximum value for RR, ZZ
    min_value: Optional[int] = None  # Minimum value for RR, ZZ
    num_bound: Optional[int] = None  # Maximum value for denominator in QQ
    nonzero: bool = False  # Whether to exclude zero

class NumberSampler:
    """Number sampler from a specified field"""
    
    def __init__(
        self,
        field,  # Coefficient field (GF, RR, QQ, ZZ, etc.)
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the number sampler
        
        Args:
            field: Field to sample from
            config: Sampling configuration
        """
        self.field = field
        
        # Default settings
        default_config = {
            'max_value': 10 if field in (RR, ZZ) else None,
            'min_value': -10 if field in (RR, ZZ) else None,
            'num_bound': 10 if field == QQ else None,
            'nonzero': False
        }
        
        # Update settings
        if config is not None:
            default_config.update(config)
            
        self.config = NumberSamplingConfig(**default_config)
        
    def sample(self) -> Any:
        """
        Sample a number according to the configuration
        
        Returns:
            Sampled number
        """
        while True:
            if isinstance(self.field, type(GF(2))):  # For finite fields
                result = self.field.random_element()
                
            elif self.field == QQ:  # For rational number field
                num = ZZ.random_element(self.config.min_value or -self.config.num_bound,
                                      self.config.max_value or self.config.num_bound + 1)
                den = ZZ.random_element(1, self.config.num_bound + 1)
                result = QQ(num) / QQ(den)
                
            else:  # For RR, ZZ, etc.
                result = self.field.random_element(self.config.min_value,
                                                 self.config.max_value + 1)
            
            # Resample if nonzero flag is set and result is 0
            if not (self.config.nonzero and result == 0):
                return result
