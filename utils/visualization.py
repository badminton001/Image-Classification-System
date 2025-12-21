"""Visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json
import os
import sys

import logging

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import matplotlib
    
    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
    
    # Force non-interactive mode
    plt.ioff()
except Exception as exc:  # pragma: no cover - matplotlib may be missing
    plt = None
    matplotlib = None
    _import_error = exc
else:
    _import_error = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from config import FIGURE_DPI, RESULTS_DIR, SHOW_PLOTS
from utils.helpers import create_directories
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _require_matplotlib() -> None:
    if plt is None:
        message = f"matplotlib is required for visualization: {_import_error}"
        logger.error(message)
        raise ImportError(message)


def _ensure_dir(save_path: Path | str) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_history(history: Any) -> Dict[str, List[float]]:
    """
    Accept different history formats and return a dict.
    """
    if history is None:
        return {}
    if isinstance(history, Mapping):
        if "history" in history and isinstance(history["history"], Mapping):
            return dict(history["history"])
        return dict(history)
    if hasattr(history, "history"):
        return dict(getattr(history, "history"))
    return {}


def plot_training_history(history_dict: Dict[str, Any], save_path: Path | str = RESULTS_DIR / "training_history.png") -> Path:
    """Plot training history."""
    _require_matplotlib()
    create_directories()
    if not history_dict:
        logger.warning("No training history available to plot.")
        return Path(save_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    loss_ax, acc_ax = axes

    for model_name, history in history_dict.items():
        hist = _normalize_history(history)
        if not hist:
            logger.warning("Empty history for model %s; skipping.", model_name)
            continue
        epochs = range(1, len(hist.get("loss", [])) + 1)
        if hist.get("loss"):
            loss_ax.plot(epochs, hist.get("loss", []), label=f"{model_name} loss")
        if hist.get("val_loss"):
            loss_ax.plot(epochs, hist.get("val_loss", []), linestyle="--", label=f"{model_name} val_loss")
        if hist.get("accuracy"):
            acc_ax.plot(epochs, hist.get("accuracy", []), label=f"{model_name} acc")
        if hist.get("val_accuracy"):
            acc_ax.plot(epochs, hist.get("val_accuracy", []), linestyle="--", label=f"{model_name} val_acc")

    loss_ax.set_title("Training and Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(True)

    acc_ax.set_title("Training and Validation Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()
    acc_ax.grid(True)

    plt.tight_layout()
    out_path = _ensure_dir(save_path)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved training history plot to %s", out_path)
    return out_path


def plot_confusion_matrix(cm: Any, class_names: Sequence[str], model_name: str, save_path: Path | str) -> Path:
    """Plot confusion matrix."""
    _require_matplotlib()
    create_directories()
    if np is None:
        raise ImportError("numpy is required to plot confusion matrix.")

    matrix = np.array(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be a square 2D array.")
    labels = list(class_names)
    if len(labels) != matrix.shape[0]:
        logger.warning("Number of class names does not match confusion matrix size.")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    tick_marks = range(matrix.shape[0])
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, format(value, "d"), ha="center", va="center", color="white" if value > threshold else "black")

    plt.tight_layout()
    out_path = _ensure_dir(save_path)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)
    return out_path


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], save_path: Path | str = RESULTS_DIR / "model_comparison.png") -> Path:
    """Plot model comparison charts."""
    _require_matplotlib()
    create_directories()
    if np is None:
        raise ImportError("numpy is required to plot model comparison.")
    if not metrics_dict:
        logger.warning("No metrics provided for comparison.")
        return Path(save_path)

    models = list(metrics_dict.keys())
    accuracy_values = [metrics_dict[m].get("accuracy") for m in models]
    f1_values = [metrics_dict[m].get("f1_score") for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    has_accuracy = any(v is not None for v in accuracy_values)
    has_f1 = any(v is not None for v in f1_values)
    if not (has_accuracy or has_f1):
        logger.warning("Neither accuracy nor f1_score found in metrics; skipping plot.")
        return Path(save_path)

    if has_accuracy:
        ax.bar(x - width / 2, [v if v is not None else 0 for v in accuracy_values], width, label="Accuracy")
    if has_f1:
        ax.bar(x + width / 2, [v if v is not None else 0 for v in f1_values], width, label="F1-score")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = _ensure_dir(save_path)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", out_path)
    return out_path


def plot_prediction_distribution(predictions: Any, class_names: Sequence[str], save_path: Path | str) -> Path:
    """Plot prediction distribution."""
    logger.info("Starting plot_prediction_distribution...")
    _require_matplotlib()
    create_directories()
    if np is None:
        raise ImportError("numpy is required to plot prediction distribution.")
    
    logger.info("Converting predictions to numpy array...")
    probs = np.array(predictions)
    logger.info(f"Predictions shape: {probs.shape}, dtype: {probs.dtype}")
    
    if probs.ndim == 2:
        logger.info("Averaging prediction probabilities across batch dimension.")
        probs = probs.mean(axis=0)
    if probs.ndim != 1:
        raise ValueError("Predictions should be 1D or 2D array-like.")

    labels = list(class_names)
    if len(labels) != len(probs):
        logger.warning("Number of class names does not match prediction length; falling back to indices.")
        labels = [str(i) for i in range(len(probs))]

    logger.info("Creating matplotlib figure...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Distribution")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    logger.info(f"Saving figure to {save_path}...")
    out_path = _ensure_dir(save_path)
    
    try:
        fig.savefig(out_path, dpi=FIGURE_DPI)
        logger.info("Figure saved successfully")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")
        plt.close(fig)
        raise
    
    # Always close figure, never show
    plt.close(fig)
    logger.info(f"Saved prediction distribution plot to {out_path}")
    return out_path


def display_image_with_prediction(image_path: Path | str, prediction: Dict[str, Any], save_path: Path | str) -> Path:
    """Display image with prediction overlay."""
    _require_matplotlib()
    create_directories()
    if Image is None:
        raise ImportError("Pillow (PIL) is required to display images.")

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path)
    info_lines = []
    if isinstance(prediction, Mapping):
        predicted_class = prediction.get("predicted_class")
        confidence = prediction.get("confidence")
        top_k = prediction.get("top_k_predictions")
        if predicted_class is not None:
            info_lines.append(f"Predicted: {predicted_class}")
        if confidence is not None:
            info_lines.append(f"Confidence: {confidence:.2f}")
        if top_k:
            info_lines.append("Top-K:")
            for item in top_k:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    info_lines.append(f"  {item[0]}: {float(item[1]):.2f}")
                else:
                    info_lines.append(f"  {item}")
    else:
        info_lines.append(str(prediction))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.axis("off")
    y_position = 0.95
    for line in info_lines:
        ax.text(0.01, y_position, line, transform=ax.transAxes, fontsize=10, color="yellow", bbox=dict(facecolor="black", alpha=0.5))
        y_position -= 0.05

    ax.set_title("Prediction Result")
    plt.tight_layout()
    out_path = _ensure_dir(save_path)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved image with prediction overlay to %s", out_path)
    return out_path


def plot_class_counts(counts: Sequence[int], class_names: Sequence[str], save_path: Path | str) -> Path:
    """Plot class count bar chart."""
    _require_matplotlib()
    create_directories()
    
    counts_arr = np.array(counts)
    labels = list(class_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts_arr, color="teal")
    
    ax.set_ylabel("Count")
    ax.set_title("Class Prediction Counts")
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
                
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_path = _ensure_dir(save_path)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved class counts plot to %s", out_path)
    return out_path


__all__ = [
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_prediction_distribution",
    "display_image_with_prediction",
    "plot_class_counts",
]


if __name__ == "__main__":
    """Command-line visualization execution."""
    print("Running Visualization Analysis...")
    
    json_path = RESULTS_DIR / "batch_predictions.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found. Please run batch prediction first.")
        sys.exit(1)
        
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Error: scikit-learn is required for confusion matrix directly.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    files = data.get("files", [])
    preds = data.get("predictions", [])
    
    if not files or not preds or len(files) != len(preds):
        print("Error: Invalid or empty prediction data.")
        sys.exit(1)
        
    y_true = []
    y_pred = []
    class_names = set()
    
    valid_count = 0
    
    for fpath, pred in zip(files, preds):
        if "predicted_class" not in pred:
            continue
            
        # Infer true class from folder structure: .../class_name/file.jpg
        p = Path(fpath)
        true_label = p.parent.name
        pred_label = pred["predicted_class"]
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        class_names.add(true_label)
        class_names.add(pred_label)
        valid_count += 1
        
    if valid_count == 0:
        print("No valid predictions found to analyze.")
        sys.exit(0)
        
    sorted_classes = sorted(list(class_names))
    print(f"Analyzed {valid_count} samples across {len(sorted_classes)} classes.")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted_classes)
    
    # Plot Confusion Matrix
    cm_path = RESULTS_DIR / "batch_confusion_matrix.png"
    plot_confusion_matrix(cm, sorted_classes, "Batch Predictions", cm_path)
    print(f"Confusion Matrix saved to: {cm_path}")
    
    # Calculate Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / valid_count
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Generate Distribution Plot (Count of predicted classes)
    pred_counts = [y_pred.count(c) for c in sorted_classes]
    
    plot_class_counts(pred_counts, sorted_classes, RESULTS_DIR / "batch_class_distribution.png")
    print(f"Distribution Plot saved to: {RESULTS_DIR / 'batch_class_distribution.png'}")
