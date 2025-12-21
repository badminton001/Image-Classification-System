# Data Module

This module handles the entire data processing pipeline, including loading, validation, preprocessing, and augmentation.

## Components

- **`data_loader.py`**:
  - Functions to scan dataset directories and load image paths and labels.
  - Ensures data integrity by verifying file extensions.

- **`preprocessing.py`**:
  - Core logic for resizing images to target resolution (224x224).
  - Normalizes pixel values (scale 0-1).
  - Splits data into Training, Validation, and Test sets.

- **`augmentation.py`**:
  - Configures `ImageDataGenerator` for dynamic data augmentation.
  - techniques: Rotation, Zoom, Width/Height shifts, Horizontal Flip.

- **`check_real_data.py`**:
  - Utility script to verify the presence and structure of the dataset before training.

## Usage

```python
from data.preprocessing import prepare_dataset
from data.augmentation import create_augmented_train_generator

# 1. Load and Split
(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = prepare_dataset("./data_samples")

# 2. Create Generator with Augmentation
train_gen = create_augmented_train_generator(X_train, y_train, batch_size=32)
```
