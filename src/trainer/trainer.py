import torch
import wandb
from typing import Optional
from transformers import Trainer, TrainingArguments

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, 
                 *args, 
                 max_steps_per_epoch: Optional[int] = None, 
                 use_classification: bool = True,
                 use_regression: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps_per_epoch = max_steps_per_epoch
        self.use_classification = use_classification
        self.use_regression = use_regression

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_history = []
        self.current_stage = 1  # Add stage information

    def _prepare_inputs(self, inputs):
        """Prepare inputs"""
        return {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def compute_loss(self, model, inputs, return_outputs=False, ignore_index=-100, num_items_in_batch=None):
        outputs = model(**inputs)
            
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
                
        self.log_metrics(outputs, inputs, model, ignore_index=ignore_index)

        return (loss, outputs) if return_outputs else loss

    def log_metrics(self, outputs, inputs, model, ignore_index=-100):
        if not self.is_world_process_zero():
            return
        
        loss_value = outputs.loss.mean().item() if outputs.loss is not None else 0.0
        metrics = {"train/loss": loss_value}
        
        # Add stage information (if self.current_stage exists)
        if hasattr(self, 'current_stage'):
            metrics["train/current_stage"] = self.current_stage

        # Calculate average of parameter weights
        with torch.no_grad():
            param_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.requires_grad:
                    param_norm += torch.norm(param).item() ** 2
                    param_count += param.numel()
            
            if param_count > 0:
                avg_param_norm = (param_norm / param_count) ** 0.5
                metrics["train/avg_param_norm"] = avg_param_norm

        # Calculate classification metrics
        if (self.args.use_classification and 
            outputs.logits is not None and 
            'labels' in inputs):
            
            labels = inputs['labels']
            logits = outputs.logits
            
            valid_mask = labels != ignore_index
            valid_labels = labels[valid_mask]
            valid_logits = logits[valid_mask]
            
            if len(valid_labels) > 0:
                predictions = torch.argmax(valid_logits, dim=-1)
                metrics["train/classification_error"] = (
                    (predictions != valid_labels).float().mean().item()
                )

        # Calculate regression metrics
        if (self.args.use_regression and 
            outputs.regression_logits is not None and 
            'regression_labels' in inputs):
            
            labels = inputs['regression_labels']
            logits = outputs.regression_logits
            
            valid_mask = labels.isfinite() & (labels != ignore_index)
            
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                valid_logits = logits[valid_mask]
                
                # Regression evaluation metrics (e.g., MSE)
                mse = torch.nn.functional.mse_loss(valid_logits, valid_labels)
                metrics["train/regression_mse"] = mse.item()

        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2

        # Add to log history
        self.log_history.append(metrics)

        metrics["gpu_memory_used_MB"] = gpu_memory_allocated
        metrics["gpu_memory_reserved_MB"] = gpu_memory_reserved

        wandb.log(metrics)
        

    def set_stage(self, stage):
        """Set the stage"""
        self.current_stage = stage

