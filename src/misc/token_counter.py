import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

def count_tokens_in_file(file_path: str) -> Dict[str, float]:
    """
    Calculate token counts for each line in a file

    Args:
        file_path: Path to target file

    Returns:
        Dictionary containing token count statistics
    """
    
    F_lengths = []
    G_lengths = []
    F_G_lengths = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Split by '#' to get input and output
            parts = line.strip().split(' # ')
            
            F_length = sum([len(part.strip().split()) for part in parts[:-1]])
            G_length = len(parts[-1].strip().split())
            F_G_length = F_length + G_length
            F_G_lengths.append(F_G_length)
            F_lengths.append(F_length)
            G_lengths.append(G_length)
    
    # Calculate statistics
    stats = {
        'num_samples': len(F_G_lengths),
        'avg_F_length': float(np.mean(F_lengths)),
        'std_F_length': float(np.std(F_lengths)),
        'min_F_length': float(np.min(F_lengths)),
        'max_F_length': float(np.max(F_lengths)),
        'median_F_length': float(np.median(F_lengths)),
        'avg_G_length': float(np.mean(G_lengths)),
        'std_G_length': float(np.std(G_lengths)),
        'min_G_length': float(np.min(G_lengths)),
        'max_G_length': float(np.max(G_lengths)),
        'median_G_length': float(np.median(G_lengths)),
        'avg_F_G_length': float(np.mean(F_G_lengths)),
        'std_F_G_length': float(np.std(F_G_lengths)),
        'min_F_G_length': float(np.min(F_G_lengths)),
        'max_F_G_length': float(np.max(F_G_lengths)),
        'median_F_G_length': float(np.median(F_G_lengths)),
    }
    
    return stats


def main():
    """
    Main function
    
    Usage example:
        python token_counter.py path/to/dataset.txt
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze token counts in dataset')
    parser.add_argument('--file_path', type=str, help='Path to dataset file to analyze')
    args = parser.parse_args()
    
    # Run token count analysis
    stats = count_tokens_in_file(args.file_path)
    
    # Display results
    
    print(f'Number of sampels: {stats["num_samples"]}')
    
    print(f'F length:')
    print(f'  Average: {stats["avg_F_length"]:.2f}')
    print(f'  Standard deviation: {stats["std_F_length"]:.2f}') 
    print(f'  Minimum: {stats["min_F_length"]}')
    print(f'  Maximum: {stats["max_F_length"]}')
    print(f'  Median: {stats["median_F_length"]}')
    print()
    
    print(f'G length:')
    print(f'  Average: {stats["avg_G_length"]:.2f}')
    print(f'  Standard deviation: {stats["std_G_length"]:.2f}') 
    print(f'  Minimum: {stats["min_G_length"]}')
    print(f'  Maximum: {stats["max_G_length"]}')
    print(f'  Median: {stats["median_G_length"]}')
    print()
    
    print(f'F + G length:')
    print(f'  Average: {stats["avg_F_G_length"]:.2f}')
    print(f'  Standard deviation: {stats["std_F_G_length"]:.2f}') 
    print(f'  Minimum: {stats["min_F_G_length"]}')
    print(f'  Maximum: {stats["max_F_G_length"]}')
    print(f'  Median: {stats["median_F_G_length"]}')
    

if __name__ == '__main__':
    main()
