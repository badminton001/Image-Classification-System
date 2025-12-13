"""
Model Configuration Module
Configuration constants for model architecture and training parameters.
"""

# Input specifications
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 6
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Model Architecture hyperparameters
DENSE_LAYER_UNITS = 256
DROPOUT_RATE = 0.5

# Training hyperparameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32

# Model identifiers
MODEL_VGG16 = 'vgg16'
MODEL_RESNET50 = 'resnet50'
MODEL_MOBILENETV2 = 'mobilenetv2'

AVAILABLE_MODELS = [MODEL_VGG16, MODEL_RESNET50, MODEL_MOBILENETV2]

# Callbacks settings
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7
