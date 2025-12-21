# Prediction Module

This module handles the inference logic, facilitating both single-image prediction and batch processing.

## Components

- **`inference.py`**:
  - Core functions for loading models and preprocessing input images.
  - Handles prediction logic and result formatting.

- **`predictor.py`**:
  - A high-level `Predictor` class that wraps the inference logic.
  - Provides a simplified API for the UI and external scripts.

- **`config.py`**:
  - Configuration constants specific to inference (e.g., supported image formats, default Top-K).

- **`utils.py`**:
  - Helper functions for results formatting and file system operations related to prediction.

## Usage

```python
from predict.predictor import Predictor

# 1. Initialize
predictor = Predictor(
    model_name="vgg16",
    model_path="weights/vgg16_best_model.h5",
    class_names=["buildings", "forest", "glacier", "mountain", "sea", "street"]
)

# 2. Predict Single Image
result = predictor.predict("test_image.jpg")
print(f"Predicted Class: {result['predicted_class']}")

# 3. Batch Predict
batch_results = predictor.predict_batch(["img1.jpg", "img2.jpg"])
```
