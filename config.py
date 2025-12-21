"""Global configuration."""
from pathlib import Path
from typing import Dict, Tuple

# Dataset
DATASET_PATH: Path = Path("./data_samples")

# Split ratios
TRAIN_SIZE: float = 0.7
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15

# Image resolution
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Models
MODEL_NAMES = ["vgg16", "resnet50", "mobilenetv2"]
WEIGHTS_DIR: Path = Path("./weights")

# Training
BATCH_SIZE: int = 32
EPOCHS: int = 50
LEARNING_RATE: float = 0.001
EARLY_STOP_PATIENCE: int = 5

# Output
RESULTS_DIR: Path = Path("./results")
LOG_FILE: Path = RESULTS_DIR / "app.log"

# Model metadata
MODELS_INFO: Dict[str, Dict[str, str]] = {
    "vgg16": {"description": "Classic CNN architecture", "size": "Large"},
    "resnet50": {"description": "Deep residual network", "size": "Large"},
    "mobilenetv2": {"description": "Lightweight mobile-friendly network", "size": "Small"},
}


# Visualization
FIGURE_DPI: int = 120
SHOW_PLOTS: bool = False

# Prediction
TOP_K: int = 3
