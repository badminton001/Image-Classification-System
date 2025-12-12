"""
Prediction module for image classification.

This package provides functions and classes for making predictions
using pre-trained image classification models.

Modules:
    - inference: Core prediction functions
    - predictor: Predictor class for convenient prediction

Example Usage:
    # Method 1: Using individual functions
    from predict.inference import load_model, predict_single_image
    
    model = load_model('VGG16', 'weights/vgg16.h5')
    result = predict_single_image(model, 'test.jpg', class_names)
    
    # Method 2: Using Predictor class (recommended)
    from predict.predictor import Predictor
    
    predictor = Predictor('VGG16', 'weights/vgg16.h5', class_names)
    result = predictor.predict('test.jpg', top_k=3)
"""

from .inference import (
    load_model,
    preprocess_input_image,
    predict_single_image,
    predict_batch,
    format_predictions
)

from .predictor import Predictor

# Import configuration and utils for convenience
import logging
from . import config
from . import utils

# Setup module-level logger
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)

__all__ = [
    'load_model',
    'preprocess_input_image',
    'predict_single_image',
    'predict_batch',
    'format_predictions',
    'Predictor',
    'config',
    'utils'
]

__version__ = '1.0.0'
__author__ = 'Member 4 - Prediction & Inference Team'
