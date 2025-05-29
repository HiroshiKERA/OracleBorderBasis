import os
import pandas as pd
import matplotlib.pyplot as plt
from src.misc.monomial_token_counter import main

def analyze_token_stats(base_dir: str, n_values: list):
    """
    Analyze token statistics for each n value and aggregate results
    
    Args:
        base_dir (str): Base directory (e.g. 'data/prod')
        n_values (list): List of n values to analyze
    """
    all_stats = []
    
    for n in n_values:
        data_dir = os.path.join(base_dir, f'GF7_n={n}')
        config_path = os.path.join(data_dir, 'config.yaml')
        data_path = os.path.join(data_dir, 'test')
        
        print(f"Analyzing n={n}...")
        stats_df, _ = main(config_path, data_path, k=-1, output_dir=data_dir)
        all_stats.append(stats_df)
    
    # Combine all results
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Save results
    output_path = os.path.join(base_dir, 'combined_token_statistics.csv')
    combined_stats.to_csv(output_path, index=False)
    print(f"Combined statistics saved to {output_path}")

    return combined_stats

def plot_stat(df, base_dir):
    # Get data for each n
    n_values = df['num_variables']

    # Calculate total, min and max for infix and monomial
    infix_means = df['infix_input_mean'] + df['infix_target_mean']
    monomial_means = df['monomial_input_mean'] + df['monomial_target_mean']

    infix_mins = df['infix_input_max']  # Treating max as min (min column needed)
    infix_maxs = df['infix_target_max']
    monomial_mins = df['monomial_input_max']  # Treating max as min (min column needed)
    monomial_maxs = df['monomial_target_max']

    # Set font size
    font = {'size': 14}
    plt.rc('font', **font)

    # Plot
    plt.figure(figsize=(8, 6))

    # Plot infix
    plt.plot(n_values, infix_means, marker='o', label='Infix Representation', linestyle='--')
    plt.fill_between(n_values, infix_mins, infix_maxs, alpha=0.2) #  label='Infix Min-Max Range'

    # Plot monomial
    plt.plot(n_values, monomial_means, marker='o', label='Monomial Representation')
    plt.fill_between(n_values, monomial_mins, monomial_maxs, alpha=0.2) # , label='Monomial Min-Max Range'

    plt.xlabel('Number of Variables (n)')
    plt.ylabel('Number of Tokens')
    # plt.yscale('log')  # Change y-axis to log scale
    # plt.yticks([1e2, 1e3, 1e4], [r'$10^2$', r'$10^3$', r'$10^4$'])
    plt.title('Token Count vs Number of Variables')
    plt.legend()
    plt.grid(axis='y', which='major', linestyle='--')
    plt.tight_layout()
    # plt.show()

    plt.savefig(os.path.join(base_dir, 'infix_vs_monomial_stats.pdf'))


if __name__ == "__main__":
    base_dir = "data/prod"
    n_values = [2, 3, 4, 5]
    df = analyze_token_stats(base_dir, n_values)
    plot_stat(df, base_dir) 