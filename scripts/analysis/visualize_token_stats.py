import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_plot_style():
    """Set up the plot style"""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12

def plot_fixed_vars(df: pd.DataFrame, output_dir: str):
    """Create plots with fixed number of variables"""
    os.makedirs(os.path.join(output_dir, "fixed_vars"), exist_ok=True)
    
    for num_vars in df['num_variables'].unique():
        df_vars = df[df['num_variables'] == num_vars]
        
        # Plot for monomial tokens
        plt.figure(figsize=(12, 8))
        for k in [1, 5, 10]:
            df_k = df_vars[df_vars['k'] == k]
            plt.plot(df_k['total_degree_bound'], df_k['monomial_input_max'], 
                    marker='o', label=f'Input (k={k})')
            plt.plot(df_k['total_degree_bound'], df_k['monomial_target_max'], 
                    marker='s', label=f'Target (k={k})')
        
        plt.xlabel('Total Degree Bound')
        plt.ylabel('Maximum Token Length')
        plt.title(f'Monomial Tokens (n={num_vars})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "fixed_vars", f"monomial_n{num_vars}.pdf"))
        plt.close()
        
        # Plot for infix tokens
        plt.figure(figsize=(12, 8))
        for k in [1, 5, 10]:
            df_k = df_vars[df_vars['k'] == k]
            plt.plot(df_k['total_degree_bound'], df_k['infix_input_max'], 
                    marker='o', label=f'Input (k={k})')
            plt.plot(df_k['total_degree_bound'], df_k['infix_target_max'], 
                    marker='s', label=f'Target (k={k})')
        
        plt.xlabel('Total Degree Bound')
        plt.ylabel('Maximum Token Length')
        plt.title(f'Infix Tokens (n={num_vars})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "fixed_vars", f"infix_n{num_vars}.pdf"))
        plt.close()

def plot_fixed_degree(df: pd.DataFrame, output_dir: str):
    """Create plots with fixed total degree"""
    os.makedirs(os.path.join(output_dir, "fixed_deg"), exist_ok=True)
    
    for total_degree in df['total_degree_bound'].unique():
        df_deg = df[df['total_degree_bound'] == total_degree]
        
        # Plot for monomial tokens
        plt.figure(figsize=(12, 8))
        for k in [1, 5, 10]:
            df_k = df_deg[df_deg['k'] == k]
            plt.plot(df_k['num_variables'], df_k['monomial_input_max'], 
                    marker='o', label=f'Input (k={k})')
            plt.plot(df_k['num_variables'], df_k['monomial_target_max'], 
                    marker='s', label=f'Target (k={k})')
        
        plt.xlabel('Number of Variables')
        plt.ylabel('Maximum Token Length')
        plt.title(f'Monomial Tokens (d={total_degree})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "fixed_deg", f"monomial_d{total_degree}.pdf"))
        plt.close()
        
        # Plot for infix tokens
        plt.figure(figsize=(12, 8))
        for k in [1, 5, 10]:
            df_k = df_deg[df_deg['k'] == k]
            plt.plot(df_k['num_variables'], df_k['infix_input_max'], 
                    marker='o', label=f'Input (k={k})')
            plt.plot(df_k['num_variables'], df_k['infix_target_max'], 
                    marker='s', label=f'Target (k={k})')
        
        plt.xlabel('Number of Variables')
        plt.ylabel('Maximum Token Length')
        plt.title(f'Infix Tokens (d={total_degree})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "fixed_deg", f"infix_d{total_degree}.pdf"))
        plt.close()

def create_summary_tables(df: pd.DataFrame, output_dir: str):
    """Create summary tables"""
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    
    # Summary by number of variables
    var_summary = df.groupby(['num_variables', 'k']).agg({
        'monomial_input_max': 'max',
        'monomial_target_max': 'max',
        'infix_input_max': 'max',
        'infix_target_max': 'max'
    }).round(2)
    
    var_summary.to_csv(os.path.join(output_dir, "tables", "summary_by_variables.csv"))
    
    # Summary by total degree
    deg_summary = df.groupby(['total_degree_bound', 'k']).agg({
        'monomial_input_max': 'max',
        'monomial_target_max': 'max',
        'infix_input_max': 'max',
        'infix_target_max': 'max'
    }).round(2)
    
    deg_summary.to_csv(os.path.join(output_dir, "tables", "summary_by_degree.csv"))

def main():
    # Read results
    results_dir = "results/token_size_analysis"
    df = pd.read_csv(os.path.join(results_dir, "all_results.csv"))
    
    # Set up plot style
    setup_plot_style()
    
    # Create visualizations
    plot_fixed_vars(df, results_dir)
    plot_fixed_degree(df, results_dir)
    
    # Create summary tables
    create_summary_tables(df, results_dir)

if __name__ == "__main__":
    main() 