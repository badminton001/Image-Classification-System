# Prediction Module

This module manages model inference, from loading weights to generating predictions.

## Components

- **inference.py**: Low-level functions for image loading, preprocessing, and model prediction.
- **predictor.py**: High-level `Predictor` class wrapper for easy integration.
- **config.py**: Configuration constants for inference (e.g., supported formats).
- **utils.py**: Helpers for file validation and result formatting.

## Usage

```python
from predict.predictor import Predictor

# Initialize
predictor = Predictor("vgg16", "weights/vgg16_best_model.h5", class_names=["cat", "dog"])

# Predict
result = predictor.predict("test_image.jpg")
print(result)
```
