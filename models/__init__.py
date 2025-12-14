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

from .evaluate import (
    evaluate_single_model,
    get_predictions,
    get_predicted_classes,
    get_confusion_matrix,
    evaluate_all_models,
    compare_model_metrics
)

from .metrics_calculator import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    get_classification_report,
    get_model_performance_dict,
    save_evaluation_results
)

