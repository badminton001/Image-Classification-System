"""
Model Evaluation Module
Evaluates trained models on test sets and compares performance across multiple models.
"""

import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple, Any, List
from .metrics_calculator import get_model_performance_dict

def evaluate_single_model(model: keras.Model, X_test, y_test) -> Tuple[float, float]:

    """Evaluate model on test set."""
    print(f"\nEvaluating {model.name}...")
    
    # Use Keras evaluate method
    results = model.evaluate(X_test, y_test, verbose=0)
    loss = results[0]
    accuracy = results[1]
    
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy


def get_predictions(model: keras.Model, X_test) -> np.ndarray:

    """Get model predictions for test set."""
    print(f"Getting predictions from {model.name}...")
    predictions = model.predict(X_test, verbose=0)
    
    return predictions


def get_predicted_classes(predictions: np.ndarray) -> np.ndarray:

    """Convert prediction probabilities to class labels."""
    # Use argmax to get the class with highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predicted_classes


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    """Generate confusion matrix."""
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Generate confusion matrix using sklearn
    cm = confusion_matrix(y_true, y_pred)
    
    return cm


def evaluate_all_models(
    models_dict: Dict[str, keras.Model],
    X_test,
    y_test,
    class_names: List[str]
) -> Dict[str, Dict[str, Any]]:

    """Evaluate all models and compare performance."""
    print("\nStarting Model Evaluation and Comparison")
    
    all_results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        predictions = get_predictions(model, X_test)
        predicted_classes = get_predicted_classes(predictions)
        
        # Convert y_test if one-hot encoded
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        # Calculate all performance metrics
        performance = get_model_performance_dict(
            model_name, 
            y_true, 
            predicted_classes, 
            class_names
        )
        
        all_results[model_name] = performance
        
        # Display summary
        print(f"\n{model_name} Performance Summary:")
        print(f"  Accuracy:  {performance['accuracy']:.4f}")
        print(f"  Precision: {performance['precision']:.4f}")
        print(f"  Recall:    {performance['recall']:.4f}")
        print(f"  F1-Score:  {performance['f1_score']:.4f}")
    
    # Find best model
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    best_accuracy = all_results[best_model_name]['accuracy']
    
    print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}\n")
    
    return all_results


def compare_model_metrics(evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:

    """Extract metrics from evaluation results for comparison visualization."""
    comparison = {
        'model_names': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for model_name, metrics in evaluation_results.items():
        comparison['model_names'].append(model_name)
        comparison['accuracy'].append(metrics['accuracy'])
        comparison['precision'].append(metrics['precision'])
        comparison['recall'].append(metrics['recall'])
        comparison['f1_score'].append(metrics['f1_score'])
    
    return comparison


# Module test
if __name__ == "__main__":
    print("Evaluation Module Test")
    print("\nModule loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_single_model()")
    print("  - get_predictions()")
    print("  - get_predicted_classes()")
    print("  - get_confusion_matrix()")
    print("  - evaluate_all_models()")
    print("  - compare_model_metrics()")
