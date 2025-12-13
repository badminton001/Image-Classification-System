"""
Models Package
Exposes model architectures, training logic, and configurations.
"""

from .config_models import (
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASS_NAMES,
    MODEL_VGG16,
    MODEL_RESNET50,
    MODEL_MOBILENETV2,
    AVAILABLE_MODELS
)

from .model_architecture import (
    build_vgg16_model,
    build_resnet50_model,
    build_mobilenetv2_model,
    compile_model,
    unfreeze_model_layers
)

from .train import (
    train_single_model,
    train_all_models,
    save_training_history,
    load_trained_model,
    get_callbacks
)
