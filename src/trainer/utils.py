import numpy as np
from sklearn.metrics import accuracy_score


def preprocess_logits_for_metrics(outputs, labels):
    """Preprocess prediction results
    
    Returns:
        tuple: (classification_predictions, regression_predictions)
        - classification_predictions: Predictions for classification task (if exists)
        - regression_predictions: Predictions for regression task (if exists)
    """
    return outputs[0].argmax(dim=-1)


def compute_metrics(eval_preds, ignore_index=-100):
    """Calculate evaluation metrics
    
    Args:
        eval_preds: Tuple of (predictions, labels)
            - predictions: Output from preprocess_logits_for_metrics
            - labels: Tuple of (classification_labels, regression_labels)
    """
    
    predictions, labels = eval_preds
    classification_preds = predictions
    
    metrics = {}
    
    if classification_preds is not None:
        classification_labels = labels[0] if isinstance(labels, tuple) else labels
        valid_mask = classification_labels != ignore_index
        
        if valid_mask.any():
            valid_preds = classification_preds[valid_mask]
            valid_labels = classification_labels[valid_mask]
            error = 1 - accuracy_score(valid_labels, valid_preds)
            metrics["classification_error"] = error
    
    return metrics