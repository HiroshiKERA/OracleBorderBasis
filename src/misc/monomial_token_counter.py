from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus
from src.loader.data_format.processors.base import ProcessorChain
from src.loader.data import load_data
from src.loader.tokenizer import set_tokenizer, set_vocab
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def calc_stats(lengths: List[int]) -> Dict[str, float]:
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'max': np.max(lengths),
    }

def get_stats_df(
    input_ids_lengths: List[int],
    target_ids_lengths: List[int],
    input_monomial_ids_lengths: List[int],
    target_monomial_ids_lengths: List[int],
    config: Dict,
    k: int
) -> pd.DataFrame:
    # Create a dictionary with all statistics
    stats_dict = {
        'num_variables': config['ring']['num_variables'],
        # 'total_degree_bound': config['dataset']['total_degree_bound'],
        # 'degree_bounds': str(config['dataset']['degree_bounds']),
        # 'k': k,
        'infix_input_max': calc_stats(input_ids_lengths)['max'],
        'infix_input_mean': calc_stats(input_ids_lengths)['mean'],
        'infix_target_max': calc_stats(target_ids_lengths)['max'],
        'infix_target_mean': calc_stats(target_ids_lengths)['mean'],
        'monomial_input_max': calc_stats(input_monomial_ids_lengths)['max'],
        'monomial_input_mean': calc_stats(input_monomial_ids_lengths)['mean'],
        'monomial_target_max': calc_stats(target_monomial_ids_lengths)['max'],
        'monomial_target_mean': calc_stats(target_monomial_ids_lengths)['mean'],
    }
    return pd.DataFrame([stats_dict])

def main(config_path: str, data_path: str, k: int, output_dir: str = '', sample_size: int = None) -> Tuple[pd.DataFrame, Dict]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vocab = set_vocab(
        num_vars=config['ring']['num_variables'],
        field=f"{config['field']['type']}{config['field']['param']}",
        max_coeff=config['polynomial']['coefficient']['max'],
        max_degree=config['polynomial']['max_degree']
    )
    tokenizer = set_tokenizer(vocab, max_seq_length=1000)


    num_variables = config['ring']['num_variables']
    max_degree = 20
    field_order = config['field']['param']

    # set up vocab and tokenizer
    # data path
    test_data_path  = data_path

    data_collator_name = 'monomial'

    _processors = []
    if k > 0:
        separator, supersparator = ' [SEP] ', ' [BIGSEP] '
        _processors.append(ExtractKLeadingTermsProcessor(k, separator=separator, supersparator=supersparator))

    subprocessors = {}
    subprocessors['monomial_ids'] = MonomialProcessorPlus(
                num_variables=num_variables,
                max_degree=max_degree,
                max_coef=int(field_order)  # 'GF7' -> 7
            )

    processor = ProcessorChain(_processors) 

    # load test dataset
    test_dataset, data_collator = load_data(
        data_path=test_data_path,
        processor=processor,
        subprocessors=subprocessors,
        splits=[{"name": "test", "batch_size": 32, "shuffle": False}],
        sample_size=sample_size,
        tokenizer=tokenizer,
        return_dataloader=False,
        data_collator_name=data_collator_name
    )
    
    input_ids_lengths = [len(sample['input'].split()) for sample in test_dataset]
    target_ids_lengths = [len(sample['target'].split()) for sample in test_dataset]
    
    input_monomial_ids_lengths = [len(sample['input_monomial_ids']) for sample in test_dataset]
    target_monomial_ids_lengths = [len(sample['target_monomial_ids']) for sample in test_dataset]

    # print statistics of input_monomial_ids_lengths and target_monomial_ids_lengths
    # print it beautifully in table format 
    # only show the first decimal of float numbers
    
    print('-' * 50)
    print('Path info')
    print('-' * 50)
    print(f"{'Config':<15} {config_path}")
    print(f"{'Dataset':<15} {test_data_path}")
    
    print()
    
    print('-' * 50)
    print('Setup info')
    print('-' * 50)
    print(f"{'Variables':<15} {num_variables}")
    print(f"{'Max degree':<15} {max_degree}")
    print(f"{'Field order':<15} {field_order}")
    print(f"{'K-leading V':<15} {k}")

    print()

    print('-' * 50)
    print(f"{'Statistic':<10} {'Input':<10} {'Target':<10}// in infix tokens")
    print('-' * 50)
    print(f"{'Mean':<10} {calc_stats(input_ids_lengths)['mean']:<10.1f} {calc_stats(target_ids_lengths)['mean']:<10.1f}")
    print(f"{'Median':<10} {calc_stats(input_ids_lengths)['median']:<10.1f} {calc_stats(target_ids_lengths)['median']:<10.1f}")
    print(f"{'Std':<10} {calc_stats(input_ids_lengths)['std']:<10.1f} {calc_stats(target_ids_lengths)['std']:<10.1f}")
    print(f"{'Max':<10} {calc_stats(input_ids_lengths)['max']:<10} {calc_stats(target_ids_lengths)['max']:<10}")

    print()

    print('-' * 50)
    print(f"{'Statistic':<10} {'Input':<10} {'Target':<10}// in monomial tokens" )
    print('-' * 50)
    print(f"{'Mean':<10} {calc_stats(input_monomial_ids_lengths)['mean']:<10.1f} {calc_stats(target_monomial_ids_lengths)['mean']:<10.1f}")
    print(f"{'Median':<10} {calc_stats(input_monomial_ids_lengths)['median']:<10.1f} {calc_stats(target_monomial_ids_lengths)['median']:<10.1f}")
    print(f"{'Std':<10} {calc_stats(input_monomial_ids_lengths)['std']:<10.1f} {calc_stats(target_monomial_ids_lengths)['std']:<10.1f}")
    print(f"{'Max':<10} {calc_stats(input_monomial_ids_lengths)['max']:<10} {calc_stats(target_monomial_ids_lengths)['max']:<10}")

    if output_dir:
        plt.hist(input_monomial_ids_lengths)
        plt.savefig(f"{output_dir}/monomial_token_counter_input_sequence_length.pdf")
        
        # clear the current figure
        plt.clf()
        
        plt.hist(target_monomial_ids_lengths)
        plt.savefig(f"{output_dir}/monomial_token_counter_target_sequence_length.pdf")
    # plt.show()
    
    # Create and return DataFrame
    stats_df = get_stats_df(
        input_ids_lengths,
        target_ids_lengths,
        input_monomial_ids_lengths,
        target_monomial_ids_lengths,
        config,
        k
    )
    
    return stats_df, config

if __name__ == "__main__":
    '''
    usage 
    config_path: config/problems/border_basis.yaml
    k: 1
    python3 src/misc/monomial_token_conter.py --config_path ${config_path} --k ${k}
    '''
    
    
    parser = argparse.ArgumentParser(description='Run monomial token counter')
    parser.add_argument('--config_path', type=str, help='Path to config file (e.g., config/problems/border_basis.yaml)')
    parser.add_argument('--data_path', type=str, help='Path to data file (e.g., data/expansion/GF31_n=3/test.infix)')
    # parser.add_argument('--k_F', type=int, help='k-leading F')
    parser.add_argument('--k', type=int, default=-1, help='k-leading V')
    parser.add_argument('--output_dir', type=str, help='Output directory', default='')
    args = parser.parse_args()
    
    stats_df, _ = main(args.config_path, args.data_path, args.k, args.output_dir)
    
    # If output_dir is specified, save the DataFrame
    if args.output_dir:
        stats_df.to_csv(f"{args.output_dir}/token_statistics.csv", index=False)