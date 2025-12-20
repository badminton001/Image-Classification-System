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
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """Generate detailed classification report."""
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
    """Print formatted metrics summary."""
    print(f"\n{model_name} Performance:")
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print()


# Module test
if __name__ == "__main__":
    print("Metrics Utility Module Test")
    print("\nModule loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_accuracy()")
    print("  - calculate_precision()")
    print("  - calculate_recall()")
    print("  - calculate_f1_score()")
    print("  - get_classification_report()")
    print("  - print_metrics_summary()")
