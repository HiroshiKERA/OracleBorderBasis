from dataclasses import dataclass
from pathlib import Path
import yaml
import json
import logging
from typing import Optional, List
import os

@dataclass
class ExperimentConfig:
    exp_name: str
    task: str
    batch_size: int
    epochs: int

    def __post_init__(self):
        """Handle additional attributes"""
        pass

    def save_config(self, save_path: Path):
        """Save configuration"""
        config_dict = {}
        
        # Convert all attributes to dict
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # Convert Path objects to strings
                if isinstance(value, Path):
                    value = str(value)
                config_dict[key] = value
        
        # Save in JSON format
        config_path = save_path / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Also save in YAML format for readability
        yaml_path = save_path / 'training_config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load_config(cls, config_path: Path):
        """Load saved configuration"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Create instance
        config = cls()
        
        # Load configuration
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix == '.yaml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Dynamically add remaining attributes
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        return config

    @classmethod
    def load_configs(cls, config_paths: List[Path]):
        """Load configurations from multiple YAML files"""
        configs = []
        for config_path in config_paths:
            config = cls.load_config(config_path)
            configs.append(config)
        return configs

    @classmethod
    def from_yaml(cls, yaml_path: str, experiment_name: str, task_name: Optional[str] = None, 
                base_config_path: Optional[str] = "config/base.yaml",
                tasks_config_path: Optional[str] = "config/tasks.yaml"):
        """
        Load configurations from multiple YAML files
        
        Args:
            yaml_path: Path to experiment configuration file
            experiment_name: Name of the experiment
            task_name: Task name (if not specified, use settings from yaml_path)
            base_config_path: Path to base configuration file (default: "config/base.yaml")
            tasks_config_path: Path to task configuration file (default: "config/tasks.yaml")
        """
        # Load base settings
        base_settings = {}
        if base_config_path and Path(base_config_path).exists():
            with open(base_config_path) as f:
                base_config = yaml.safe_load(f)
                base_settings = base_config.get('base', {})

        # Load task settings
        task_settings = {}
        if task_name and tasks_config_path and Path(tasks_config_path).exists():
            with open(tasks_config_path) as f:
                tasks_config = yaml.safe_load(f)
                if task_name not in tasks_config["tasks"]:
                    raise ValueError(f"Unknown task: {task_name}")
                task_settings = tasks_config["tasks"][task_name]

        # Load experiment settings
        with open(yaml_path) as f:
            exp_config = yaml.safe_load(f)
            if experiment_name not in exp_config["experiments"]:
                raise ValueError(f"Unknown experiment: {experiment_name}")
            experiment_settings = exp_config["experiments"][experiment_name]

        # Merge settings (priority: command line > experiment settings > task settings > base settings)
        merged_config = {**base_settings, **task_settings, **experiment_settings}

        # Get keys corresponding to explicit fields
        allowed_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in merged_config.items() if k in allowed_keys}

        # Create instance and dynamically add other fields as attributes
        instance = cls(**filtered_config)
        for k, v in merged_config.items():
            if k not in allowed_keys:
                setattr(instance, k, v)
        
        return instance

    def setup_experiment(self, dryrun: bool = False):
        """Setup experiment (create directories, configure logging, etc.)"""
        data_name = os.path.basename(self.data_path)
        save_file = '_'.join(self.tags)
        save_file = f'{save_file}_m={self.training_size}_{data_name}'
        
        base_path = Path("results/train")
        group = 'dryrun' if dryrun else self.group
        save_path = base_path / group / self.task / f'{save_file}_m={self.training_size}'
        if hasattr(self, 'extra_path'):
            save_path = save_path / self.extra_path
        
        # Create directory
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.save_config(save_path)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(save_path / "training_process.log"),
                logging.StreamHandler()
            ]
        )
        
        return save_path

