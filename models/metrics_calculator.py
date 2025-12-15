"""
Metrics Calculator Module
Calculates performance metrics (Accuracy, Precision, Recall, F1) and generates reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score (0-1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score
    """
    accuracy = accuracy_score(y_true, y_pred)
    return float(accuracy)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        Precision score
    """
    # Handle edge cases with zero_division parameter
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    return float(precision)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        Recall score
    """
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    return float(recall)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
    
    Returns:
        F1 score
    """
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return float(f1)


def get_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> str:
    """
    Generate classification report with per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report as formatted string
    """
    # Generate report using sklearn
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        zero_division=0
    )
    
    return report


def get_model_performance_dict(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate all metrics and return as dictionary.
    
    Args:
        model_name: Name of the model
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary containing:
        - accuracy: float
        - precision: float
        - recall: float
        - f1_score: float
        - confusion_matrix: numpy array
        - classification_report: str
    """
    # Calculate all metrics
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, average='weighted')
    recall = calculate_recall(y_true, y_pred, average='weighted')
    f1 = calculate_f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = get_classification_report(y_true, y_pred, class_names)
    
    # Organize results
    performance_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return performance_dict


def save_evaluation_results(
    evaluation_results: Dict[str, Dict[str, Any]],
    output_path: str = 'results/evaluation_results.txt'
) -> None:
    """
    Save evaluation results to text file.
    
    Args:
        evaluation_results: Results from evaluate_all_models()
        output_path: Path to save results
    """
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for model_name, metrics in evaluation_results.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*70}\n\n")
            
            # Write metrics
            f.write(f"Overall Performance Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
            
            # Write classification report
            f.write(f"Detailed Classification Report:\n")
            f.write(f"{'-'*70}\n")
            f.write(metrics['classification_report'])
            f.write(f"\n{'-'*70}\n")
            
            # Write confusion matrix
            f.write(f"\nConfusion Matrix:\n")
            cm = metrics['confusion_matrix']
            for row in cm:
                f.write("  " + " ".join(f"{val:4d}" for val in row) + "\n")
            f.write("\n")
        
        # Summary comparison
        f.write(f"\n{'='*70}\n")
        f.write("SUMMARY COMPARISON\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write(f"{'-'*70}\n")
        
        for model_name, metrics in evaluation_results.items():
            f.write(f"{model_name:<20} "
                   f"{metrics['accuracy']:>10.4f} "
                   f"{metrics['precision']:>10.4f} "
                   f"{metrics['recall']:>10.4f} "
                   f"{metrics['f1_score']:>10.4f}\n")
        
        # Find best model
        best_model = max(evaluation_results.keys(), 
                        key=lambda k: evaluation_results[k]['accuracy'])
        best_acc = evaluation_results[best_model]['accuracy']
        
        f.write(f"\n{'='*70}\n")
        f.write(f"Best Model: {best_model} (Accuracy: {best_acc:.4f})\n")
        f.write(f"{'='*70}\n")
    
    print(f"\nEvaluation results saved to {output_path}")


# Module test
if __name__ == "__main__":
    print("="*60)
    print("Metrics Calculator Module Test")
    print("="*60)
    print("\nModule loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_accuracy()")
    print("  - calculate_precision()")
    print("  - calculate_recall()")
    print("  - calculate_f1_score()")
    print("  - get_classification_report()")
    print("  - get_model_performance_dict()")
    print("  - save_evaluation_results()")
