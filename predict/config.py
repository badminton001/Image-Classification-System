"""Configuration constants for prediction module."""

# Preprocessing
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_IMAGE_MODE = 'RGB'

# Prediction
DEFAULT_TOP_K = 3
MAX_BATCH_SIZE = 32

# Supported architectures
SUPPORTED_MODELS = [
    'VGG16',
    'ResNet50',
    'MobileNetV2'
]

# Image formats
SUPPORTED_IMAGE_FORMATS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff',
    '.webp'
]

# Paths
DEFAULT_WEIGHTS_DIR = 'weights'
DEFAULT_RESULTS_DIR = 'results'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Validation
VALIDATE_INPUT = True
MIN_IMAGE_SIZE = (32, 32)
MAX_IMAGE_SIZE = (4096, 4096)

# Performance
USE_GPU = True
PREFETCH_ENABLED = True

# Formatting
PREDICTION_DECIMAL_PLACES = 4
DISPLAY_CONFIDENCE_AS_PERCENTAGE = True
