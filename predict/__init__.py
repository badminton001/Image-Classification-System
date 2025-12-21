"""Prediction module for image classification."""

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
