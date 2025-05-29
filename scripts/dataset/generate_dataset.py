#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime
import wandb
from src.dataset.generators.polynomial_prod_generator import PolynomialProdGenerator
from src.dataset.generators.polynomial_sum_generator import PolynomialSumGenerator
from src.dataset.generators.border_basis_generator import BorderBasisGenerator
from src.dataset.generators.expansion_generator import ExpansionGenerator
from src.misc.token_counter import count_tokens_in_file

def setup_logger(log_dir: Path, timestamp: str):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log file
        timestamp: Timestamp string for log file name
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"generation_log_{timestamp}.txt")
        ]
    )
    return logging.getLogger(__name__)

def adjust_config(input_dict):
    """
    Expand keys with dots in them into nested dictionaries.
    Example:
        {
          "a.b": 1,
          "a.c": 2,
          "d": 3,
          "f.g.h": 4
        }
    becomes:
        {
          "a": {
            "b": 1,
            "c": 2
          },
          "d": 3,
          "f": {
            "g": {
              "h": 4
            }
          }
        }
    """
    output = {}
    for key, value in input_dict.items():
        parts = key.split(".")
        
        # Navigate through 'output' according to the split parts
        current = output
        for part in parts[:-1]:
            # If 'part' is not in 'current', set it to an empty dict
            current = current.setdefault(part, {})
        
        # The last part points to the final value
        current[parts[-1]] = value
    
    return output

def get_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to problem configuration YAML file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/polynomial_sum",
        help="Directory to save generated datasets"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="If None, infer task from save_dir (default: None)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    parser.add_argument(
        "--data_tag",
        type=str,
        default=None,
        help="Optional tag for the dataset directory"
    )

    parser.add_argument(
        "--wandb", 
        action='store_true', 
        help="Enable wandb logging")
    
    return parser

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create timestamp for log and summary files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup base save directory
    
    if args.task == "prod":
        DatasetGenerator = PolynomialProdGenerator
    elif args.task == "sum":
        DatasetGenerator = PolynomialSumGenerator
    elif args.task == "border_basis":
        DatasetGenerator = BorderBasisGenerator
    elif args.task == "expansion":
        DatasetGenerator = ExpansionGenerator
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    try:
        # Load dataset sizes from config
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # dict to name space
        config = argparse.Namespace(**config)
        
        if args.task in ('border_basis', 'expansion'):
            degree_bounds = '_'.join(map(str, config.dataset['degree_bounds']))
            data_tag = f"GF{config.field['param']}_n={config.ring['num_variables']}_deg={config.polynomial['max_degree']}_terms={config.polynomial['max_terms']}_bounds={degree_bounds}_total={config.dataset['total_degree_bound']}"
        else:
            data_tag = f"GF{config.field['param']}_n={config.ring['num_variables']}"
            
        save_dir = os.path.join(args.save_dir, data_tag)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generator
        generator = DatasetGenerator.from_yaml(args.config, save_dir)
        

        if args.wandb:
            wandb.init(project="border_basis", config=config)
            config = wandb.config
            config = adjust_config(config)
            name_space = argparse.Namespace(**config)
            degree_bounds = '_'.join(map(str, name_space.dataset['degree_bounds']))
            data_tag = f"GF{name_space.field['param']}_n={name_space.ring['num_variables']}_deg={name_space.polynomial['max_degree']}_terms={name_space.polynomial['max_terms']}_bounds={degree_bounds}_total={name_space.dataset['total_degree_bound']}"
            save_dir = os.path.join(args.save_dir, data_tag)
            save_dir = Path(save_dir)
            # reinitialize generator with the adjusted config
            generator = DatasetGenerator(save_dir, config)
            config = name_space
        else: 
            if args.task in ('border_basis', 'expansion'):
                degree_bounds = '_'.join(map(str, config.dataset['degree_bounds']))
                data_tag = f"GF{config.field['param']}_n={config.ring['num_variables']}_deg={config.polynomial['max_degree']}_terms={config.polynomial['max_terms']}_bounds={degree_bounds}_total={config.dataset['total_degree_bound']}"
            else:
                data_tag = f"GF{config.field['param']}_n={config.ring['num_variables']}"
            save_dir = os.path.join(args.save_dir, data_tag)
            save_dir = Path(save_dir)
            # save_dir.mkdir(parents=True, exist_ok=True)

            # Initialize generator
            generator = DatasetGenerator.from_yaml(args.config, save_dir)
        # Get dataset-specific directory
        # dataset_dir = generator._get_dataset_dir()
        # dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger to write to the dataset directory
        logger = setup_logger(save_dir, timestamp)

        logger.info(f"Using configuration from: {config}")

        # Log configuration
        logger.info(f"Using configuration from: {args.config}")
        logger.info(f"Saving datasets to: {save_dir}")
        logger.info(f"Using {args.n_jobs} parallel jobs")
        
        dataset_config = config.dataset
        
        train_num_samples = dataset_config["num_samples_train"]
        test_num_samples = dataset_config["num_samples_test"]
        # train_num_samples = 10
        # test_num_samples = 10
        
        
        # Generate training set
        logger.info(f"Generating training set ({dataset_config['num_samples_train']} samples)...")
        if args.task == "expansion":
            # for expansion, we need to specify the train tag 
            train_samples, train_stats = generator.run(
                num_samples=train_num_samples,
                n_jobs=args.n_jobs,
                data_tag="train"
            )
        else:
            train_samples, train_stats = generator.run(
                num_samples=train_num_samples,
                n_jobs=args.n_jobs
            )

        
        generator.save_dataset(train_samples, train_stats, tag="train", data_tag=args.data_tag)
        logger.info("Training set generation completed")

        if args.wandb:
            wandb.log({f"bb/{key}": value for key, value in train_stats.items()})
        
        # Generate test set
        logger.info(f"Generating test set ({dataset_config['num_samples_test']} samples)...")
        if args.task == "expansion":
            # for expansion, we need to specify the test tag 
            test_samples, test_stats = generator.run(
                num_samples=test_num_samples,
                n_jobs=args.n_jobs,
                data_tag="test"
            )
        else:
            test_samples, test_stats = generator.run(
                num_samples=test_num_samples,
                n_jobs=args.n_jobs)

        generator.save_dataset(test_samples, test_stats, tag="test", data_tag=args.data_tag)
        logger.info("Test set generation completed")
        
        #test_file_path = save_dir / "test.infix"
        #test_stats = count_tokens_in_file(str(test_file_path))
        #logger.info(f"Test set token statistics: {test_stats}")

        # Save summary of all splits in the dataset directory
        summary = {
            "train": train_stats,
        #    "test": test_stats,
            # "test_tokens": test_stats
        }
        with open(save_dir / f"dataset_summary.yaml", "w") as f:
            yaml.dump(summary, f)
        
        logger.info("Dataset generation completed successfully")

        if args.wandb:
            # Log all test statistics to wandb
            for key, value in test_stats.items():
                wandb.log({f"test_{key}": value})

        
        # Count tokens in the test set
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()