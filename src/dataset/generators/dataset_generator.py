from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
import json
from joblib import Parallel, delayed
import numpy as np
from datetime import datetime

@dataclass
class DatasetStatistics:
    """Base class for dataset statistics"""
    sample_count: int
    generation_time: float
    statistics_by_sample: List[Dict[str, Any]]
    
    def summarize(self, metrics: List[str] = ['mean', 'std', 'max', 'min']) -> Dict[str, float]:
        """Summarize statistics across all samples"""
        summary = {}
        for key in self.statistics_by_sample[0].keys():
            if isinstance(self.statistics_by_sample[0][key], (int, float)):
                values = [item[key] for item in self.statistics_by_sample]
                if 'mean' in metrics:
                    summary[f'{key}_mean'] = float(np.mean(values))
                if 'std' in metrics:
                    summary[f'{key}_std'] = float(np.std(values))
                if 'max' in metrics:
                    summary[f'{key}_max'] = float(np.max(values))
                if 'min' in metrics:
                    summary[f'{key}_min'] = float(np.min(values))
        return summary

class DatasetGenerator(ABC):
    """Base class for dataset generation"""
    
    def __init__(self, save_dir: str, config: Dict[str, Any]):
        """
        Initialize dataset generator
        
        Args:
            save_dir: Directory to save generated datasets
            config: Configuration dictionary
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._validate_config(config)
        
    @abstractmethod
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Validated configuration dictionary
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def generate_sample(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a single sample
        
        Returns:
            Tuple of (generated_sample, sample_statistics)
        """
        pass
        
    def generate_dataset(self, 
                        num_samples: int,
                        n_jobs: int = -1,
                        seed: Optional[int] = None
                        ) -> Tuple[List[Any], DatasetStatistics]:
        """
        Generate multiple samples with parallel processing
        
        Args:
            num_samples: Number of samples to generate
            n_jobs: Number of parallel jobs (-1 for all cores)
            seed: Random seed
            
        Returns:
            Tuple of (samples, statistics)
        """
        if seed is not None:
            np.random.seed(seed)
            
        start_time = datetime.now()
        
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=True)(
            delayed(self.generate_sample)() for _ in range(num_samples)
        )
        
        samples, stats = zip(*results)
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        return list(samples), DatasetStatistics(
            sample_count=num_samples,
            generation_time=generation_time,
            statistics_by_sample=list(stats)
        )
    
    def save_dataset(self,
                    samples: List[Any],
                    statistics: DatasetStatistics,
                    tag: str = "train",
                    save_format: str = "txt") -> None:
        """
        Save generated dataset and its statistics
        
        Args:
            samples: List of generated samples
            statistics: Dataset statistics
            tag: Dataset tag (e.g., "train", "test")
            save_format: Format to save the dataset ("txt" or "json")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.save_dir / f"dataset_{tag}_{timestamp}"
        
        # Save configuration
        with open(f"{base_path}_config.yaml", "w") as f:
            yaml.dump(self.config, f)
            
        # Save statistics
        stats_summary = statistics.summarize()
        with open(f"{base_path}_stats.json", "w") as f:
            json.dump({
                "summary": stats_summary,
                "by_sample": statistics.statistics_by_sample,
                "generation_time": statistics.generation_time,
                "sample_count": statistics.sample_count
            }, f, indent=2)
            
        # Save dataset
        if save_format == "txt":
            self._save_txt(samples, base_path)
        elif save_format == "json":
            self._save_json(samples, base_path)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
    
    @abstractmethod
    def _save_raw(self, samples: List[Any], base_path: Path) -> None:
        """Save dataset in text format (raw)"""
        pass

    @abstractmethod
    def _save_infix(self, samples: List[Any], base_path: Path) -> None:
        """Save dataset in text format (infix)"""
        pass
    
    @abstractmethod
    def _save_json(self, samples: List[Any], base_path: Path) -> None:
        """Save dataset in JSON format"""
        pass