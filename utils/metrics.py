"""
Evaluation metrics for model validation
"""
import numpy as np
from typing import Dict, List


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient"""
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]


def mean_absolute_error(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate MAE"""
    return np.mean(np.abs(true - pred))


def root_mean_squared_error(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate RMSE"""
    return np.sqrt(np.mean((true - pred) ** 2))


def functional_connectivity_similarity(fc_true: np.ndarray, 
                                       fc_pred: np.ndarray) -> Dict:
    """
    Compare functional connectivity matrices
    
    Args:
        fc_true: True connectivity matrix
        fc_pred: Predicted connectivity matrix
    
    Returns:
        Dictionary of similarity metrics
    """
    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices_from(fc_true, k=1)
    true_vec = fc_true[triu_idx]
    pred_vec = fc_pred[triu_idx]
    
    return {
        'correlation': float(correlation_coefficient(true_vec, pred_vec)),
        'mae': float(mean_absolute_error(true_vec, pred_vec)),
        'rmse': float(root_mean_squared_error(true_vec, pred_vec))
    }


def classification_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          num_classes: int) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    # Handle both integer and one-hot encoded labels
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    class_metrics = []
    for c in range(num_classes):
        true_positive = np.sum((y_true == c) & (y_pred == c))
        false_positive = np.sum((y_true != c) & (y_pred == c))
        false_negative = np.sum((y_true == c) & (y_pred != c))
        
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        class_metrics.append({
            'class': c,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
    
    # Macro average
    macro_f1 = np.mean([m['f1_score'] for m in class_metrics])
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_metrics': class_metrics
    }


def network_efficiency(connectivity_matrix: np.ndarray) -> Dict:
    """
    Calculate network efficiency metrics
    
    Args:
        connectivity_matrix: Connectivity matrix
    
    Returns:
        Efficiency metrics
    """
    n = len(connectivity_matrix)
    
    # Global efficiency (inverse of path lengths)
    # Simplified calculation
    inv_matrix = np.where(connectivity_matrix > 0, 
                         1.0 / connectivity_matrix, 
                         0)
    
    global_eff = np.sum(inv_matrix) / (n * (n - 1))
    
    # Local efficiency (average of nodal efficiencies)
    local_effs = []
    for i in range(n):
        neighbors = np.where(connectivity_matrix[i] > 0)[0]
        if len(neighbors) > 1:
            subgraph = connectivity_matrix[np.ix_(neighbors, neighbors)]
            inv_subgraph = np.where(subgraph > 0, 1.0 / subgraph, 0)
            local_eff = np.sum(inv_subgraph) / (len(neighbors) * (len(neighbors) - 1))
            local_effs.append(local_eff)
    
    local_eff = np.mean(local_effs) if local_effs else 0.0
    
    return {
        'global_efficiency': float(global_eff),
        'local_efficiency': float(local_eff)
    }
