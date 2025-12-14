"""
Metrics Utility Module
Wrapper functions for performance metric calculations.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from typing import List


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Wrapper for sklearn.metrics.accuracy_score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score (0-1)
    """
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate precision score.
    
    Wrapper for sklearn.metrics.precision_score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
    
    Returns:
        Precision score
    """
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate recall score.
    
    Wrapper for sklearn.metrics.recall_score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
    
    Returns:
        Recall score
    """
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate F1 score.
    
    Wrapper for sklearn.metrics.f1_score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method
    
    Returns:
        F1 score
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Generate detailed classification report.
    
    Wrapper for sklearn.metrics.classification_report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report string with per-class metrics
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )


def print_metrics_summary(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    model_name: str = "Model"
) -> None:
    """
    Print formatted metrics summary.
    
    Args:
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        model_name: Name of model
    """
    print(f"\n{model_name} Performance:")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'='*50}\n")


# Module test
if __name__ == "__main__":
    print("="*60)
    print("Metrics Utility Module Test")
    print("="*60)
    print("\n✓ Module loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_accuracy()")
    print("  - calculate_precision()")
    print("  - calculate_recall()")
    print("  - calculate_f1_score()")
    print("  - get_classification_report()")
    print("  - print_metrics_summary()")
