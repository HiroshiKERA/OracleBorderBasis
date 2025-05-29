import yaml
import os
import pandas as pd
import subprocess
from typing import Dict

def create_config(base_config_path: str, num_vars: int, total_degree: int) -> Dict:
    """Create a config dictionary with specified parameters"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update parameters
    config['ring']['num_variables'] = num_vars
    config['dataset']['degree_bounds'] = [5] * num_vars
    config['dataset']['total_degree_bound'] = total_degree
    
    return config

def save_config(config: Dict, output_path: str):
    """Save config to a YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def run_experiment(
    base_config_path: str,
    num_vars: int,
    total_degree: int,
    k: int,
    results_dir: str
) -> pd.DataFrame:
    """Run a single experiment with specified parameters"""
    
    # Create experiment directory
    exp_name = f"n{num_vars}_d{total_degree}_k{k}"
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create and save config
    config = create_config(base_config_path, num_vars, total_degree)
    config_path = os.path.join(exp_dir, "config.yaml")
    save_config(config, config_path)
    
    # Generate datasets
    subprocess.run([
        "python3", "scripts/dataset/generate_dataset.py",
        "--config", config_path,
        "--task", "border_basis",
        "--save_dir", "./data/border_basis/dataset_analysis",
        "--n_jobs", "-1"
    ], check=True)
    
    subprocess.run([
        "python3", "scripts/dataset/generate_dataset.py",
        "--config", config_path,
        "--task", "expansion",
        "--save_dir", "./data/expansion/dataset_analysis",
        "--n_jobs", "-1"
    ], check=True)
    
    # Calculate token statistics
    dataset_path = f"./data/expansion/dataset_analysis/GF31_n={num_vars}_deg=3_terms=20_bounds={'_'.join(['5']*num_vars)}_total={total_degree}/test"
    
    result = subprocess.run([
        "python3", "src/misc/monomial_token_counter.py",
        "--config_path", config_path,
        "--data_path", dataset_path,
        "--k", str(k),
        "--output_dir", exp_dir
    ], capture_output=True, text=True)
    
    # Read and return statistics
    stats_df = pd.read_csv(os.path.join(exp_dir, "token_statistics.csv"))
    return stats_df

def main():
    # Read parameter ranges
    with open("config/experiments/token_analysis_ranges.yaml", 'r') as f:
        ranges = yaml.safe_load(f)['parameter_ranges']
    
    # Create results directory
    results_dir = "results/token_size_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize DataFrame for all results
    all_results = []
    
    # Run experiments for all combinations
    for num_vars in range(ranges['num_variables']['min'], ranges['num_variables']['max'] + 1):
        for total_degree in range(ranges['total_degree_bound']['min'], ranges['total_degree_bound']['max'] + 1):
            for k in [1, 5, 10]:
                print(f"Running experiment: n={num_vars}, d={total_degree}, k={k}")
                try:
                    stats_df = run_experiment(
                        "config/problems/border_basis.yaml",
                        num_vars,
                        total_degree,
                        k,
                        results_dir
                    )
                    all_results.append(stats_df)
                except Exception as e:
                    print(f"Error in experiment: n={num_vars}, d={total_degree}, k={k}")
                    print(e)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(results_dir, "all_results.csv"), index=False)

if __name__ == "__main__":
    main() 