#!/usr/bin/env python3

import argparse
from pathlib import Path

import itertools as it
import pandas as pd

from sage.all import *

from src.misc.monomial_token_counter import main as monomial_token_counter


def get_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    return parser

def list_paths(ps=[7, 31, 127], ns=[3, 4, 5], base='./data/border_basis', skip_patterns=None):
    base = Path(base)
    for p, n in it.product(ps, ns):
        if (p, n) in skip_patterns:
            continue
        
        comb = f"p={p}_n={n}"
        degree_bounds = '_'.join([str(4)] * n)
        dirname = Path(f'GF{p}_n={n}_deg=1_terms=10_bounds={degree_bounds}_total={2}')
        data_path = base / dirname / 'test.infix'

        yield {'tag': (p, n), 'data_path': data_path}

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    stat_list = []

    for p, n, k in list(it.product([7, 31, 127], [3, 4, 5], [1, 3, 5])):
        config = f"config/problems/border_basis_GF{p}_n={n}.yaml"

        degree_bounds = '_'.join([str(4)] * n)
        dirname = Path(f'GF{p}_n={n}_deg=1_terms=10_bounds={degree_bounds}_total={2}')
        data_path = f"./data/expansion/{dirname}/test"
        
        stats_df, _ = monomial_token_counter(config, data_path, k, sample_size=args.num_samples)
        
        row = {'p': p, 'n': n, 'k': k}
        row.update(stats_df.iloc[0].to_dict())
        stat_list.append(row)

    df = pd.DataFrame(stat_list)
    df.to_csv(f"results/eval/expansion_data_token_count_samples={args.num_samples}.csv", index=False)

if __name__ == "__main__":
    main()