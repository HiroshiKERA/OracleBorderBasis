# src/evaluation/prediction_analysis.py

import os
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sage.all import PolynomialRing, GF  # Import SageMath
import time  # Import time for measurement
from tqdm.auto import tqdm # Import tqdm for progress bars
import numpy as np # Import numpy for stats

from src.loader.checkpoint import load_pretrained_bag
from src.loader.data import load_data
from src.loader.data_format.processors.base import ProcessorChain
from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus
from src.dataset.processors.utils import sequence_to_poly
from src.misc.utils import to_cuda

def load_model_and_tokenizer(checkpoint_path: str, from_checkpoint: bool = True):
    """
    Loads the model, tokenizer, and config from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        from_checkpoint (bool): Whether to load from a checkpoint.

    Returns:
        tuple: (model, tokenizer, config)
    """
    print(f"Loading model from: {checkpoint_path}")
    bag = load_pretrained_bag(checkpoint_path, from_checkpoint=from_checkpoint, cuda=True)
    model = bag['model']
    tokenizer = bag['tokenizer']
    config = bag['config']
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer, config

def load_evaluation_data(data_path: str, config, tokenizer, sample_size: int = None):
    """
    Loads the evaluation data.

    Args:
        data_path (str): Path to the data.
        config: Model configuration object.
        tokenizer: Tokenizer.

    Returns:
        tuple: (test_dataset, data_collator)
    """
    print(f"Loading data from: {data_path}")
    with open(Path(data_path) / 'config.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)

    _processors = [ExtractKLeadingTermsProcessor(config.num_leading_terms)]
    subprocessors = {
        'monomial_ids': MonomialProcessorPlus(
            num_variables=config.num_variables,
            max_degree=config.max_degree,
            max_coef=int(config.field[2:])
        )
    }
    processor = ProcessorChain(_processors)
    data_collator_name = 'monomial' # Fixed value from notebook

    test_data_path = Path(data_path) / 'test'
    test_dataset, data_collator = load_data(
        data_path=test_data_path,
        processor=processor,
        subprocessors=subprocessors,
        splits=[{"name": "test", "batch_size": 32, "shuffle": False}], # Fixed value
        tokenizer=tokenizer,
        return_dataloader=False,
        data_collator_name=data_collator_name,
        sample_size=sample_size
    )
    print("Data loaded successfully.")
    return test_dataset, data_collator

def generate_predictions(model, dataloader, tokenizer, config, device='cuda'):
    """
    Generates predictions using the model and measures generation time per batch.

    Args:
        model: The loaded model.
        dataloader: DataLoader for the evaluation data.
        tokenizer: Tokenizer.
        config: Model configuration.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        tuple: (all_predictions, all_labels, generation_time_stats)
               generation_time_stats contains timing info per batch.
    """
    model.to(device)
    all_predictions = []
    all_labels = []
    batch_times = [] # To store time taken for each batch

    mpp = MonomialProcessorPlus(
        num_variables=config.num_variables,
        max_degree=config.max_degree,
        max_coef=int(config.field[2:])
    )

    print("Generating predictions...")
    total_generation_start_time = time.time()
    # Wrap dataloader with tqdm for progress bar
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            batch_start_time = time.time() # Start timing for the batch

            batch = to_cuda(batch)
            
            max_length = batch['labels'].shape[-1] + 1
            generated_ids = model.generate(
                batch['input_ids'],
                batch['attention_mask'],
                monomial_processor=mpp,
                tokenizer=tokenizer,
                max_length=max_length # Use determined max_length
            )

            predictions_text = mpp.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(predictions_text)

            labels = batch['labels'].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_labels.extend(labels_text)

            # Ensure CUDA synchronization if timing GPU operations precisely
            if device == 'cuda':
                 torch.cuda.synchronize()
            batch_end_time = time.time() # End timing for the batch
            batch_times.append(batch_end_time - batch_start_time)

    total_generation_end_time = time.time()
    print("Prediction generation complete.")

    # Calculate generation time statistics
    generation_time_stats = {}
    if batch_times:
        generation_time_stats['times_per_batch'] = batch_times
        generation_time_stats['total_time'] = sum(batch_times) # More accurate total time from summed batches
        generation_time_stats['average_time_per_batch'] = np.mean(batch_times)
        generation_time_stats['std_dev_time_per_batch'] = np.std(batch_times)
        generation_time_stats['min_time_per_batch'] = min(batch_times)
        generation_time_stats['max_time_per_batch'] = max(batch_times)
        generation_time_stats['num_batches'] = len(batch_times)
    else:
        # Fallback if no batches were processed
        generation_time_stats['total_time'] = total_generation_end_time - total_generation_start_time
        generation_time_stats['times_per_batch'] = []
        generation_time_stats['average_time_per_batch'] = 0.0
        generation_time_stats['std_dev_time_per_batch'] = 0.0
        generation_time_stats['min_time_per_batch'] = 0.0
        generation_time_stats['max_time_per_batch'] = 0.0
        generation_time_stats['num_batches'] = 0


    return all_predictions, all_labels, generation_time_stats


def evaluate_predictions(predictions: list, labels: list, config):
    """
    Compares and evaluates generated predictions against ground truth labels. Uses SageMath.

    Args:
        predictions (list): List of predicted strings.
        labels (list): List of ground truth strings.
        config: Model configuration.

    Returns:
        tuple: (results_dict, detailed_results_list)
               results_dict contains overall metrics.
               detailed_results_list contains per-sample analysis.
    """
    print("Evaluating predictions...")
    results = {
        'total_samples': len(predictions),
        'no_expansion_total': 0,
        'no_expansion_correct': 0,
        'expansion_metrics': {
            'total_true_positives': 0,
            'total_false_positives': 0,
            'total_false_negatives': 0,
            'samples_analyzed': 0,
        }
    }
    null_exponent = ' '.join(['E0'] * config.num_variables)
    no_expansion_text = f'C1 {null_exponent} [SEP] C1 {null_exponent}' # Fixed value from notebook

    try:
        field = GF(int(config.field[2:]))
        ring = PolynomialRing(field, 'x', config.num_variables, order='degrevlex')
        sage_available = True
        print("SageMath environment detected.")
    except NameError:
        print("SageMath not found. Skipping SageMath-based evaluation.")
        sage_available = False
        ring = None # ring is None if SageMath is not available

    detailed_results = []

    # Wrap enumeration with tqdm for progress bar
    for i, (pred_str, gt_str) in enumerate(tqdm(zip(predictions, labels), total=len(predictions), desc="Evaluating Samples")):
        sample_result = {
            'id': i,
            'prediction': pred_str,
            'ground_truth': gt_str,
            'is_no_expansion_gt': False,
            'no_expansion_correct': None,
            'expansion_analysis': None,
            'detailed_comparison': []
        }

        # 1. No Expansion Accuracy
        is_no_expansion_gt = (gt_str == no_expansion_text)
        sample_result['is_no_expansion_gt'] = is_no_expansion_gt
        if is_no_expansion_gt:
            results['no_expansion_total'] += 1
            is_correct = (pred_str == gt_str)
            sample_result['no_expansion_correct'] = is_correct
            if is_correct:
                results['no_expansion_correct'] += 1
        else:
            # 2. Expansion Metrics (True Positives, False Positives, False Negatives) using SageMath
            results['expansion_metrics']['samples_analyzed'] += 1
            if sage_available and ring is not None:
                try:
                    preds = pred_str.split('[SEP]')
                    pred_directions_str, pred_leading_terms_str = preds[::2], preds[1::2]
                    gts = gt_str.split('[SEP]')
                    gt_directions_str, gt_leading_terms_str = gts[::2], gts[1::2]

                    # Remove empty strings
                    pred_directions_str = [s for s in pred_directions_str if s.strip()]
                    pred_leading_terms_str = [s for s in pred_leading_terms_str if s.strip()]
                    gt_directions_str = [s for s in gt_directions_str if s.strip()]
                    gt_leading_terms_str = [s for s in gt_leading_terms_str if s.strip()]

                    # Error handling for unequal lengths (use minimum length for comparison)
                    min_len_pred = min(len(pred_directions_str), len(pred_leading_terms_str))
                    min_len_gt = min(len(gt_directions_str), len(gt_leading_terms_str))

                    # Convert to SageMath objects
                    pred_direction_set_sage = [sequence_to_poly(s, ring) for s in pred_directions_str[:min_len_pred]]
                    pred_lt_set_sage = [sequence_to_poly(s, ring) for s in pred_leading_terms_str[:min_len_pred]]
                    pred_expansion_set = set(zip(pred_direction_set_sage, pred_lt_set_sage)) # Convert to set

                    gt_direction_set_sage = [sequence_to_poly(s, ring) for s in gt_directions_str[:min_len_gt]]
                    gt_lt_set_sage = [sequence_to_poly(s, ring) for s in gt_leading_terms_str[:min_len_gt]]
                    gt_expansion_set = set(zip(gt_direction_set_sage, gt_lt_set_sage)) # Convert to set

                    true_positive = len(pred_expansion_set.intersection(gt_expansion_set))
                    false_positive = len(pred_expansion_set) - true_positive
                    false_negative = len(gt_expansion_set) - true_positive

                    results['expansion_metrics']['total_true_positives'] += true_positive
                    results['expansion_metrics']['total_false_positives'] += false_positive
                    results['expansion_metrics']['total_false_negatives'] += false_negative

                    sample_result['expansion_analysis'] = {
                        'true_positive': true_positive,
                        'false_positive': false_positive,
                        'false_negative': false_negative,
                        'pred_count': len(pred_expansion_set),
                        'gt_count': len(gt_expansion_set)
                    }
                except Exception as e:
                     # Provide more context in the error message if possible
                     print(f"Error processing sample {i} ('{pred_str[:20]}...' vs '{gt_str[:20]}...') with SageMath: {e}")
                     sample_result['expansion_analysis'] = {'error': str(e)}


            # 3. Detailed Comparison (String-based)
            preds = pred_str.split('[SEP]')
            pred_directions, pred_leading_terms = preds[::2], preds[1::2]
            gts = gt_str.split('[SEP]')
            gt_directions, gt_leading_terms = gts[::2], gts[1::2]

            # Remove empty strings
            pred_directions = [s for s in pred_directions if s.strip()]
            pred_leading_terms = [s for s in pred_leading_terms if s.strip()]
            gt_directions = [s for s in gt_directions if s.strip()]
            gt_leading_terms = [s for s in gt_leading_terms if s.strip()]

            max_len = max(len(pred_directions), len(gt_directions))
            for j in range(max_len):
                pred_dir = pred_directions[j].strip() if j < len(pred_directions) else "N/A"
                pred_lt = pred_leading_terms[j].strip() if j < len(pred_leading_terms) else "N/A"
                gt_dir = gt_directions[j].strip() if j < len(gt_directions) else "N/A"
                gt_lt = gt_leading_terms[j].strip() if j < len(gt_leading_terms) else "N/A"

                dir_correct = (pred_dir == gt_dir) and (pred_dir != "N/A") and (gt_dir != "N/A")
                lt_correct = (pred_lt == gt_lt) and (pred_lt != "N/A") and (gt_lt != "N/A")
                both_correct = dir_correct and lt_correct

                sample_result['detailed_comparison'].append({
                    'pred_direction': pred_dir,
                    'pred_leading_term': pred_lt,
                    'gt_direction': gt_dir,
                    'gt_leading_term': gt_lt,
                    'direction_correct': dir_correct,
                    'lt_correct': lt_correct,
                    'both_correct': both_correct
                })

        detailed_results.append(sample_result)

    # Calculate overall metrics
    if results['no_expansion_total'] > 0:
        results['no_expansion_accuracy'] = results['no_expansion_correct'] / results['no_expansion_total']
    else:
        results['no_expansion_accuracy'] = None # Or 0 or NaN, depending on preference

    if results['expansion_metrics']['samples_analyzed'] > 0 and sage_available:
        tp = results['expansion_metrics']['total_true_positives']
        fp = results['expansion_metrics']['total_false_positives']
        fn = results['expansion_metrics']['total_false_negatives']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results['expansion_metrics']['precision'] = precision
        results['expansion_metrics']['recall'] = recall
        results['expansion_metrics']['f1_score'] = f1
    else:
         results['expansion_metrics']['precision'] = None
         results['expansion_metrics']['recall'] = None
         results['expansion_metrics']['f1_score'] = None

    print("Evaluation complete.")
    return results, detailed_results

def print_evaluation_summary(results: dict, detailed_results: list, generation_time_stats: dict, sample_id_to_print: int = None):
    """
    Prints a summary of the evaluation results, generation time, and details for a specific sample ID.

    Args:
        results (dict): Overall results from evaluate_predictions.
        detailed_results (list): List of detailed results from evaluate_predictions.
        generation_time_stats (dict): Timing statistics from generate_predictions.
        sample_id_to_print (int, optional): Sample ID to print details for.
    """
    print("\n--- Evaluation Summary ---")
    print(f"Total samples processed: {results['total_samples']}")

    # Print Generation Time Stats
    print("\nGeneration Time Statistics (per batch):")
    if generation_time_stats and generation_time_stats['num_batches'] > 0:
        print(f"  Number of Batches: {generation_time_stats['num_batches']}")
        print(f"  Total Time: {generation_time_stats['total_time']:.4f} seconds")
        print(f"  Average Time: {generation_time_stats['average_time_per_batch']:.6f} seconds")
        print(f"  Std Dev Time: {generation_time_stats['std_dev_time_per_batch']:.6f} seconds")
        print(f"  Min Time:     {generation_time_stats['min_time_per_batch']:.6f} seconds")
        print(f"  Max Time:     {generation_time_stats['max_time_per_batch']:.6f} seconds")
    else:
        print("  No generation timing data available.")

    if results['no_expansion_accuracy'] is not None:
        print(f"\nNo Expansion Accuracy: {results['no_expansion_accuracy']:.4f} ({results['no_expansion_correct']} / {results['no_expansion_total']})")
    else:
        print("\nNo 'No Expansion' samples found.")

    if 'precision' in results['expansion_metrics'] and results['expansion_metrics']['precision'] is not None :
        print("\nExpansion Metrics (based on SageMath comparison):")
        print(f"  Samples Analyzed: {results['expansion_metrics']['samples_analyzed']}")
        print(f"  Total True Positives: {results['expansion_metrics']['total_true_positives']}")
        print(f"  Total False Positives: {results['expansion_metrics']['total_false_positives']}")
        print(f"  Total False Negatives: {results['expansion_metrics']['total_false_negatives']}")
        print(f"  Precision: {results['expansion_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['expansion_metrics']['recall']:.4f}")
        print(f"  F1 Score:  {results['expansion_metrics']['f1_score']:.4f}")
    elif results['expansion_metrics']['samples_analyzed'] > 0:
         print("\nExpansion Metrics (SageMath comparison skipped or failed):")
         print(f"  Samples Analyzed: {results['expansion_metrics']['samples_analyzed']}")
    else:
        print("\nNo expansion samples found for SageMath comparison.")


    if sample_id_to_print is not None and 0 <= sample_id_to_print < len(detailed_results):
        print(f"\n--- Detailed Comparison for Sample ID: {sample_id_to_print} ---")
        sample_data = detailed_results[sample_id_to_print]
        print(f"Prediction:\n{sample_data['prediction']}")
        print(f"\nGround Truth:\n{sample_data['ground_truth']}")

        if sample_data['is_no_expansion_gt']:
             print(f"\nType: No Expansion (Correct: {sample_data['no_expansion_correct']})")
        else:
            print("\nType: Expansion")
            if sample_data['expansion_analysis']:
                if 'error' in sample_data['expansion_analysis']:
                     print(f"  SageMath Analysis Error: {sample_data['expansion_analysis']['error']}")
                else:
                    print(f"  SageMath Analysis:")
                    print(f"    Predicted Expansions: {sample_data['expansion_analysis']['pred_count']}")
                    print(f"    Ground Truth Expansions: {sample_data['expansion_analysis']['gt_count']}")
                    print(f"    True Positives:  {sample_data['expansion_analysis']['true_positive']}")
                    print(f"    False Positives: {sample_data['expansion_analysis']['false_positive']}")
                    print(f"    False Negatives: {sample_data['expansion_analysis']['false_negative']}")
            else:
                print("  SageMath Analysis: Skipped or Not Available")


            print("\nDetailed String Comparison:")
            print('-'*100)
            print(f'{"Prediction":<45} | {"Ground Truth":<45} | Correct')
            print('-'*100)
            print(f'{"Direction":<21} | {"Leading Term":<21} | {"Direction":<21} | {"Leading Term":<21} | {"Dir."} | {"LT"}  | {"Both"}')
            print('-'*100)
            for comp in sample_data['detailed_comparison']:
                 print(f"{comp['pred_direction']:<21} | {comp['pred_leading_term']:<21} | {comp['gt_direction']:<21} | {comp['gt_leading_term']:<21} | {str(comp['direction_correct']):<4} | {str(comp['lt_correct']):<3} | {str(comp['both_correct']):<4}")
            print('-'*100)

    print("\n--- End of Summary ---")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Load a Transformer model, generate predictions, and evaluate them.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model directory (containing pytorch_model.bin, config.json, etc.)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory (containing test/, config.yaml)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction.')
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    parser.add_argument('--sample_id', type=int, default=None, help='Specific sample ID to print detailed comparison for.')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to print detailed comparison for.')
    # parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use if device is cuda.')


    args = parser.parse_args()

    # # Set CUDA device if specified
    # if args.device == 'cuda':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    #     if not torch.cuda.is_available():
    #         print(f"Warning: CUDA device specified ({args.gpu_id}) but CUDA not available. Switching to CPU.")
    #         args.device = 'cpu'
    #     else:
    #          print(f"Using CUDA device: {torch.cuda.current_device()}")
    #          print(f"(Note: This index refers to the GPU visible to PyTorch based on CUDA_VISIBLE_DEVICES='{args.gpu_id}')")


    # 1. Load Model
    model, tokenizer, config = load_model_and_tokenizer(args.model_path)

    # 2. Load Data
    test_dataset, data_collator = load_evaluation_data(args.data_path, config, tokenizer, sample_size=args.sample_size)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    # 3. Generate Predictions
    predictions, labels, generation_time_stats = generate_predictions(model, dataloader, tokenizer, config)

    # 4. Evaluate Predictions
    # SageMath related evaluation will be skipped or show error messages if the environment is not available.
    results, detailed_results = evaluate_predictions(predictions, labels, config)

    # 5. Print Summary
    print_evaluation_summary(results, detailed_results, generation_time_stats, sample_id_to_print=args.sample_id)

    print("\nAnalysis script finished.")
