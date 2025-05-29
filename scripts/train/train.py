import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from time import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import wandb
import string
from transformers import set_seed

from src.evaluation.generation import generation_accuracy
from src.loader.data import load_data
from src.loader.model import load_model
from src.loader.tokenizer import set_vocab, set_tokenizer
from src.loader.data_format.processors.base import ProcessorChain, CoefficientPostfixProcessor, ExtractLastProcessor, MultiCoefficientPostfixProcessor, ExtractLeadingTermProcessor
from src.loader.data_format.processors.expansion import ExtractKLeadingTermsProcessor
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus
from src.misc.utils import count_cuda_devices
from src.trainer.trainer import CustomTrainer, CustomTrainingArguments
from src.trainer.utils import compute_metrics, preprocess_logits_for_metrics
from scripts.train.experiment import ExperimentConfig

# Warning settings
import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

# Environment variable settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_parser():
    """Generate parameter parser"""
    parser = argparse.ArgumentParser(description="Polynomial Transformer Training")

    # Basic settings
    parser.add_argument("--config", type=str, help="Path to experiment config")
    parser.add_argument("--experiment", type=str, help="Experiment name in config")
    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--save_path", type=str, default="./dumped", help="Save path")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--group", type=str, default="", help="Experiment group")
    parser.add_argument("--task", type=str, default="sum", help="Task name")

    # Model parameters
    parser.add_argument("--model", type=str, default="bart", help="Model type")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attention_dropout", type=float, default=0, help="Attention dropout rate")
    parser.add_argument("--encoding_method", type=str, default="standard")

    # Task settings
    parser.add_argument("--use_classification", action="store_true", default=True, help="Enable classification head")
    parser.add_argument("--use_regression", action="store_true", default=False, help="Enable regression head")
    parser.add_argument("--classification_weight", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--regression_weight", type=float, default=1.0, help="Regression loss weight")

    # Data settings
    parser.add_argument("--num_variables", type=int, default=2, help="Number of variables")
    parser.add_argument("--field", type=str, default="QQ", help="Field type (QQ, RR, or GFP)")
    parser.add_argument("--max_coefficient", type=int, default=1000, help="Maximum coefficient value")
    parser.add_argument("--max_degree", type=int, default=10, help="Maximum polynomial degree")
    parser.add_argument("--coeff_encoding", type=str, default="none", 
                       choices=["none", "prefix", "postfix", "postfix_input", "postfix_target"])

    # Training parameters
    parser.add_argument("--max_sequence_length", type=int, default=10000, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers")
    parser.add_argument("--training_size", type=int, default=-1, help="Limit training size (-1 for full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", choices=["adamw", "adam", "sgd"], help="Optimizer type")

    # Other settings
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--wandb_id", type=str, default=None, help="W&B run ID for resuming")
    parser.add_argument("--dryrun", action="store_true", help="Run in debug mode")
    parser.add_argument("--split_coeff_exp", action="store_true", help="Split coefficient and exponent")
    
    parser.add_argument("--token_type_position_encoding", action="store_true", default=False, help="Use token type position encoding")
    parser.add_argument("--monomial_type_position_encoding", action="store_true", default=False, help="Use monomial type position encoding")
    parser.add_argument("--monomial_id_embedding", action="store_true", default=False, help="Use monomial-id embedding")
    parser.add_argument("--monomial_embedding", action="store_true", default=False, help="Use monomial embedding")
    parser.add_argument("--token_expander", action="store_true", default='mlp2', help="Use monomial embedding")
    parser.add_argument("--coeff_token_size", type=int, default=1)
    
    parser.add_argument("--num_leading_terms", type=int, default=None, help="Number of leading terms to extract from V")
    parser.add_argument("--train_sample_skimming", action="store_true", default=False, help="Train with sample skimming")
    parser.add_argument("--aware-of-padding", action="store_true", default=False, help="Aware of padding")
    parser.add_argument("--train_test_split", action="store_true", default=False, help="Train test split")
    parser.add_argument("--save_wandb_artifact", action="store_true", default=False, help="Save artifact")
    

    return parser

def fix_seeds(seed=42):
    """Fix random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    torch.use_deterministic_algorithms(True)

    
def setup_wandb(config: ExperimentConfig, trainer_config, wandb_id=None):
    os.environ["WANDB_CACHE_DIR"] = os.path.expanduser("~/.cache/wandb")
    os.environ["WANDB_ARTIFACT_CACHE_SIZE"] = "1GB"
    
    #########################################################
    ## use this if you encounter an error at saving artifacts
    #########################################################
    # import wandb.sdk.artifacts.artifact_file_cache as afc
    # afc._get_sys_umask_threadsafe = lambda: 0o022
    
    """Setup Weights & Biases"""
    tags = []
    if config.use_classification:
        tags.append('classification')
    if config.use_regression:
        tags.append('regression')
    if hasattr(config, 'embedding_type'):
        tags.append(f'embed-{config.embedding_type}')

    datetime_str = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S")
    data_name = os.path.basename(config.data_path)
    save_file = '_'.join(config.tags)
    run_name_without_timestamp = f'{config.task}_{save_file}_m={config.training_size}'
    run_name = f'{run_name_without_timestamp}_{data_name}_{datetime_str}'

    if config.dryrun:
        config.group = f'dryrun'

    run = wandb.init(
        project=config.exp_name,
        name=run_name,
        group=config.group,
        config={
            **vars(config),
            **trainer_config.__dict__,
        },
        tags=tags,
        id=wandb_id,
        resume="allow"
    )
    return run, run_name_without_timestamp

def upload_artifact_without_checkpoints(output_dir, artifact_name, artifact_type, extra_files=None):
    """
    Upload files and directories under output_dir as wandb artifacts, excluding checkpoint directories.
    extra_files: List of additional files to upload (specified by path)
    """
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    output_dir = Path(output_dir)
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint'):
            continue
        if item.is_file():
            artifact.add_file(str(item))
        elif item.is_dir():
            artifact.add_dir(str(item))
    if extra_files:
        for f in extra_files:
            artifact.add_file(str(f))
    wandb.log_artifact(artifact)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load and setup experiment configuration
    if args.config and args.experiment:
        # Load settings from YAML
        config = ExperimentConfig.from_yaml(args.config, args.experiment)
        
        # Check argparse values
        cmd_args = {}
        for k, v in vars(args).items():
            if v != parser.get_default(k):  # If different from default
                cmd_args[k] = v
        
        # Merge configurations
        for k, v in vars(args).items():
            if hasattr(config, k):
                # If attribute exists in config
                if k in cmd_args:  # Only override if explicitly specified in command line
                    setattr(config, k, cmd_args[k])
            else:
                # Add attribute if not in config
                setattr(config, k, v)  # Include default values
        
        save_path = config.setup_experiment(dryrun=args.dryrun)
    else:
        # Use regular argparse values if no YAML file
        config = ExperimentConfig(**vars(args))
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)


    # Fix seeds
    fix_seeds(config.seed)

    # Initialize tokenizer before loading data
    vocab = set_vocab(
        num_vars=config.num_variables,
        field=config.field,
        max_coeff=config.max_coefficient,
        max_degree=config.max_degree,
        continuous_coefficient=config.use_regression  
    )
    tokenizer = set_tokenizer(vocab, max_seq_length=config.max_sequence_length)

    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    config.save_config(save_path)

    _processors = []
    if config.coeff_encoding == 'postfix': 
        _processors.append(CoefficientPostfixProcessor())
    if config.learning_target == 'last':
        separator = ' ' if config.task.startswith('arithmetic') else ' [SEP] '
        _processors.append(ExtractLastProcessor(separator=separator))
    if config.learning_target == 'leading_term':
        separator = ' ' if config.task.startswith('arithmetic') else ' [SEP] '
        _processors.append(ExtractLeadingTermProcessor(separator=separator))
    if config.learning_target == 'k-leading-terms':
        ## only for expansion task
        separator, supersparator = ' [SEP] ', ' [BIGSEP] '
        _processors.append(ExtractKLeadingTermsProcessor(config.num_leading_terms, separator=separator, supersparator=supersparator))
    if config.coeff_token_size > 1:
        _processors.append(MultiCoefficientPostfixProcessor())
        coeff_words = []
        for word in vocab: 
            if word.startswith('C'):
                coeff_words += [word + string.ascii_lowercase[i] for i in range(config.coeff_token_size)]
        vocab += coeff_words
        tokenizer = set_tokenizer(vocab, max_seq_length=config.max_sequence_length)
        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

    processor = ProcessorChain(_processors) 

    subprocessors = {}
    data_collator_name = None
    if config.token_type_position_encoding:
        subprocessors['token_types'] = TokenTypeProcessor(config.num_variables)
    if config.monomial_type_position_encoding:
        subprocessors['monomial_types'] = MonomialTypeProcessor()
    if config.monomial_embedding:    
        data_collator_name = 'monomial'
        subprocessors['monomial_ids'] = MonomialProcessorPlus(
            num_variables=config.num_variables,
            max_degree=config.max_degree,
            max_coef=int(config.field[2:])  # 'GF7' -> 7
        )

    # Load data
    train_size = config.training_size 
    train_size = 1000 if args.dryrun else train_size
    
    train_test_split = [train_size, config.test_size] if config.train_test_split else None
    train_dataset, data_collator = load_data(
        data_path=f"{config.data_path}/train",
        processor=processor,
        subprocessors=subprocessors,
        splits=[{"name": "train", "batch_size": config.batch_size, "shuffle": True}],
        tokenizer=tokenizer,
        sample_size=train_size,
        return_dataloader=False,
        data_collator_name=data_collator_name,
        sample_skimming=config.train_sample_skimming,
        aware_of_padding=config.aware_of_padding,
        train_test_split=train_test_split,
        testset_save_path=os.path.join(save_path, "test.infix")
    )


    if not config.train_test_split:
        test_size = config.test_size
        test_dataset, _ = load_data(
            data_path=f"{config.data_path}/test",
            processor=processor,
            subprocessors=subprocessors,
            splits=[{"name": "test", "batch_size": config.test_batch_size, "shuffle": False}],
            tokenizer=tokenizer,
            sample_size=test_size,
            return_dataloader=False,
            data_collator_name=data_collator_name,
            # sample_skimming=config.sample_skimming
            aware_of_padding=False,
            testset_save_path=os.path.join(save_path, "test.infix")
        )
    else:
        train_dataset, test_dataset = train_dataset

    print(len(train_dataset), len(test_dataset))
    print('############################')
    print(f'Coefficient encoding: {config.coeff_encoding}') 
    print('Example input:\n', test_dataset[0]['input'])
    print('\nExample target:\n', test_dataset[0]['target'])
    print('############################')

    

    if config.monomial_type_position_encoding:
        num_monomial_types = len(test_dataset.subprocessors['monomial_types'].type_dict)
        print(f'Number of monomial types: {num_monomial_types}')

    # Setup trainer
    # Complete trainer configuration
    trainer_config = CustomTrainingArguments(
        output_dir=str(save_path),
        
        # Model output settings
        use_classification=config.use_classification,
        use_regression=config.use_regression,
        
        # Training settings
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.batch_size // count_cuda_devices(),
        per_device_eval_batch_size=config.test_batch_size // count_cuda_devices(),
        lr_scheduler_type="constant" if config.optimizer.startswith('schedule_free') else "linear",
        
        # Optimization related
        bf16=True,  # Use bfloat16
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        optim=config.optimizer,  # Set optimizer type
        
        # Dataloader settings
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        
        # Evaluation and saving settings
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        # load_best_model_at_end=True,
        
        # Logging settings
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",
        
        # Others
        remove_unused_columns=False,
        seed=config.seed,
        disable_tqdm=True,
    )

    # Setup wandb
    run, run_name_without_timestamp = setup_wandb(config, trainer_config, wandb_id=args.wandb_id)
    
    # Model initialization function
    def model_init():
        return load_model(params=config, tokenizer=tokenizer)

    # Initialize trainer
    trainer = CustomTrainer(
        args=trainer_config,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    # Execute training and evaluation
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    
    # === Save model to wandb artifact ===
    if config.save_wandb_artifact:
        run_name = run_name_without_timestamp
        run_name = run_name.replace('=', '')
        artifact_name = f"model-{run_name}"
        artifact_type = "model"
        extra_files = [os.path.join(save_path, "test.infix")] if os.path.exists(os.path.join(save_path, "test.infix")) else None
        upload_artifact_without_checkpoints(trainer.args.output_dir, artifact_name, artifact_type, extra_files=extra_files)


    # Calculate evaluation metrics
    metrics = train_result.metrics
    test_metrics = trainer.evaluate(metric_key_prefix="test")
    metrics.update(test_metrics)

    # Evaluate generation accuracy
    test_loader = trainer.get_eval_dataloader()
    
    
    monomial_processor = subprocessors['monomial_ids'] if 'monomial_ids' in subprocessors else None
    
    compute_support_acc = config.task.startswith('polynomial')
    scores = generation_accuracy(
        trainer.model,
        test_loader,
        max_length=config.max_sequence_length,
        tokenizer=tokenizer,
        monomial_processor=monomial_processor,
        disable_tqdm=True,
        model_name=config.model,
        compute_support_acc=compute_support_acc
    )

    metrics.update({
        'test_generation_accuracy': scores['acc'],
        'test_generation_support_accuracy': scores['support_acc'],
        'test_generation_runtime': scores['runtime_per_batch']
    })

    # Save results
    trainer.save_metrics("all", metrics)
    wandb.log(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()