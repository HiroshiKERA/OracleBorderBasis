#!/usr/bin/env python3

import argparse
import itertools as it
import pandas as pd
from tqdm import tqdm

from src.dataset.generators.border_basis_generator import BorderBasisGenerator
from src.border_basis_lib.border_basis_sampling import BorderBasisGenerator as BorderBasisSampler

from sage.all import *
import sage.misc.randstate as randstate


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


def run(args, config):
    
    # Setup base save directory
    DatasetGenerator = BorderBasisGenerator
    
    generator = DatasetGenerator.from_yaml(config, None)
    
    degree_bounds = generator.problem_config.degree_bounds
    border_basis_sampler = BorderBasisSampler(generator.ring)
    ret = border_basis_sampler.random_border_basis(degree_bounds=degree_bounds, total_degree_bound=generator.problem_config.total_degree_bound, max_sampling=100)
    G = ret['basis']
    G = matrix(G).transpose()
    
    randstate.set_random_seed()
    
    num_vars = generator.ring.ngens()
    basis_size = G.nrows()
    
    results = dict([(m, []) for m in range(num_vars, 2*num_vars+1)])
    
    for i in tqdm(range(args.num_samples)):
        for m in range(num_vars, 2*num_vars+1):
            A = generator.sampler.sample(num_samples=1, size=(m, basis_size))[0]
            F = A * G
            F = F.list()
            
            is_ideal_unchanged = ideal(F) == ideal(G.list())
            
            results[m].append(is_ideal_unchanged)
    
    stats = {}
    for m in range(num_vars, 2*num_vars+1):
        success_rate = sum(results[m]) / len(results[m]) 
        stats[m] = success_rate
    
    field_order = generator.ring.base_ring().order()
    num_vars = generator.ring.ngens()
    
    col_width = 7  # Column width

    # Header
    header = f'(p, n) = ({field_order}, {num_vars}) |' + ''.join(f"{m:>{col_width}d}" for m in range(num_vars, 2*num_vars+1))
    print(header)

    # Separator line
    print('-' * len(header))

    # Data row
    data_label = 'success rate'
    data_line = data_label.ljust(len(f'(p, n) = ({field_order}, {num_vars}) |')) \
                + ''.join(f"{stats[m]:>{col_width}.2f}" for m in range(num_vars, 2*num_vars+1))
    print(data_line)
    
    return stats, field_order, num_vars


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    configs = [f"config/problems/border_basis_GF{p}_n={n}.yaml" for p, n in it.product([7, 31, 127], [3, 4, 5])]
    stat_list = []
    for config in configs:
        stats, field_order, num_vars = run(args, config)
        # Add field order (p) and number of variables (n) to dict before appending
        row = {'p': field_order, 'n': num_vars}
        row.update(stats)
        stat_list.append(row)

    df = pd.DataFrame(stat_list)
    df.to_csv(f"results/eval/back_transform_analysis_num_samples={args.num_samples}.csv", index=False)

if __name__ == "__main__":
    main()