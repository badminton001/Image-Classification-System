"""
Configuration file for the prediction module.

This module contains configuration constants and default values used
throughout the prediction pipeline.

Author: Member 4 - Prediction & Inference Team
"""

# Default image preprocessing settings
DEFAULT_IMAGE_SIZE = (224, 224)  # Standard size for VGG16, ResNet50, MobileNetV2
DEFAULT_IMAGE_MODE = 'RGB'       # Color mode

# Prediction settings
DEFAULT_TOP_K = 3                # Default number of top predictions to return
MAX_BATCH_SIZE = 32             # Maximum batch size for processing

# Supported models
SUPPORTED_MODELS = [
    'VGG16',
    'ResNet50',
    'MobileNetV2'
]

# Supported image formats
SUPPORTED_IMAGE_FORMATS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff',
    '.webp'
]

# File paths (relative to project root)
DEFAULT_WEIGHTS_DIR = 'weights'
DEFAULT_RESULTS_DIR = 'results'

# Logging settings
LOG_LEVEL = 'INFO'              # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Validation settings
VALIDATE_INPUT = True           # Whether to validate inputs
MIN_IMAGE_SIZE = (32, 32)      # Minimum allowed image size
MAX_IMAGE_SIZE = (4096, 4096)   # Maximum allowed image size

# Performance settings
USE_GPU = True                  # Whether to use GPU if available
PREFETCH_ENABLED = True         # Enable image prefetching for batches

# Output formatting
PREDICTION_DECIMAL_PLACES = 4  # Decimal places for probabilities
DISPLAY_CONFIDENCE_AS_PERCENTAGE = True
