"""
Global configuration for the Image Classification Enhancement System.

All configurable parameters are centralized here to avoid hard-coded
values scattered across the codebase.
"""
from pathlib import Path
from typing import Dict, Tuple

# ==============================
# Dataset configuration
# ==============================
# Root path to datasets; used by data preprocessing utilities.
DATASET_PATH: Path = Path("./data_samples")
# Train/validation/test split ratios; should sum to 1.0.
TRAIN_SIZE: float = 0.7
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15
# Target image resolution (height, width) for all preprocessing.
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# ==============================
# Model configuration
# ==============================
# Supported model identifiers to display in menus and for compatibility checks.
MODEL_NAMES = ["vgg16", "resnet50", "mobilenetv2"]
# Directory where model weights are stored or saved.
WEIGHTS_DIR: Path = Path("./weights")

# ==============================
# Training configuration
# ==============================
BATCH_SIZE: int = 32
EPOCHS: int = 50
LEARNING_RATE: float = 0.001
EARLY_STOP_PATIENCE: int = 5

# ==============================
# Output / results configuration
# ==============================
# Directory to store logs, plots, and serialized results.
RESULTS_DIR: Path = Path("./results")
LOG_FILE: Path = RESULTS_DIR / "app.log"
# Human-readable metadata for supported models.
MODELS_INFO: Dict[str, Dict[str, str]] = {
    "vgg16": {"description": "Classic CNN architecture", "size": "Large"},
    "resnet50": {"description": "Deep residual network", "size": "Large"},
    "mobilenetv2": {"description": "Lightweight mobile-friendly network", "size": "Small"},
}

# ==============================
# Visualization configuration
# ==============================
# Default DPI for saved matplotlib figures.
FIGURE_DPI: int = 120
# Whether to show plots interactively by default; kept False for non-blocking behavior.
SHOW_PLOTS: bool = False

# ==============================
# Prediction configuration
# ==============================
# Number of top predictions to display when available.
TOP_K: int = 3
