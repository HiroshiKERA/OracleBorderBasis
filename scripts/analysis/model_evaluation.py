import itertools as it 
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader

from src.loader.checkpoint import load_pretrained_bag
from src.loader.data import load_data
from src.loader.data_format.processors.base import ProcessorChain
from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus

from src.evaluation.prediction_analysis import generate_predictions, evaluate_predictions, print_evaluation_summary

def list_paths(ps=[7, 31, 127], ns=[3, 4, 5], ks=[1, 3, 5], base='./results/train/main/expansion', skip_patterns=None):
    base = Path(base)
    for p, n, k in it.product(ps, ns, ks):
        if (p, n, k) in skip_patterns:
            continue
        
        comb = f"p={p}_n={n}_k={k}"
        degree_bounds = '_'.join([str(4)] * n)
        dirname = f'skim_{comb}_m=1000000_GF{p}_n={n}_deg={1}_terms=10_bounds={degree_bounds}_total={2}_m=1000000'
        
        model_path = base / dirname
        test_path = model_path / 'test'

        yield {'tag': (p, n, k), 'model_path': model_path, 'test_path': test_path}
        
def load_model_and_data(model_path, test_path):
    # load model
    bag = load_pretrained_bag(model_path, from_checkpoint=False)
    model, tokenizer, config = bag['model'], bag['tokenizer'], bag['config']
    model.eval();
    
    
    # load test data
    data_collator_name = 'monomial'

    processor = ProcessorChain([ExtractKLeadingTermsProcessor(config.num_leading_terms)])

    subprocessors = {}
    subprocessors['monomial_ids'] = MonomialProcessorPlus(
                num_variables=config.num_variables,
                max_degree=config.max_degree,
                max_coef=int(config.field[2:])  # 'GF7' -> 7
            )

    test_dataset, data_collator = load_data(
        data_path=test_path,
        sample_size=1000, 
        processor=processor,
        subprocessors=subprocessors,
        splits=[{"name": "test", "batch_size": 1, "shuffle": False}],
        tokenizer=tokenizer,
        return_dataloader=False,  # return dataloader if True
        data_collator_name=data_collator_name
    )
    
    return model, tokenizer, config, test_dataset, data_collator

def main():
    skip_patterns = []
    
    results_list = []
    for path in list(list_paths(skip_patterns=skip_patterns)):
        print(path['model_path'])
        if not path['model_path'].exists():
            print(f"Model path {path['model_path']} does not exist")
            continue
        
        model, tokenizer, config, test_dataset, data_collator = load_model_and_data(path['model_path'], path['test_path'])
        
        testloader = dataloader = DataLoader(test_dataset, batch_size=250, shuffle=False, collate_fn=data_collator)
        
        # 3. Generate Predictions
        predictions, labels, generation_time_stats = generate_predictions(model, dataloader, tokenizer, config)

        # 4. Evaluate Predictions
        # SageMath related evaluation will be skipped or show error messages if the environment is not available.
        results, detailed_results = evaluate_predictions(predictions, labels, config)

        p, n, k = path['tag']
        results_list.append({
            'p': p,
            'n': n,
            'k': k,
            'no_expansion_accuracy': results.get('no_expansion_accuracy'),
            'no_expansion_total': results.get('no_expansion_total'),
            'no_expansion_correct': results.get('no_expansion_correct'),
            'expansion_samples': results['expansion_metrics'].get('samples_analyzed'),
            'true_positives': results['expansion_metrics'].get('total_true_positives'),
            'false_positives': results['expansion_metrics'].get('total_false_positives'),
            'false_negatives': results['expansion_metrics'].get('total_false_negatives'),
            'precision': results['expansion_metrics'].get('precision'),
            'recall': results['expansion_metrics'].get('recall'),
            'f1_score': results['expansion_metrics'].get('f1_score'),
        })

    df = pd.DataFrame(results_list)
    df.to_csv('results/eval/expansion_evaluation_summary.csv', index=False)
    print('saved to results/eval/expansion_evaluation_summary.csv')


if __name__ == '__main__':
    main()

