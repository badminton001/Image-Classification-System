# Data Module

This module handles the data pipeline: loading, preprocessing, and augmentation.

## Components

- **data_loader.py**: Scans directories and loads image paths/labels.
- **preprocessing.py**: Core logic for resizing, normalization, and splitting datasets (Train/Val/Test).
- **augmentation.py**: Defines Keras `ImageDataGenerator` configurations for training augmentation.

## Usage

```python
from data.preprocessing import prepare_dataset

# Load and split data
(X_train, y_train), (X_val, y_val), _ = prepare_dataset("./data_samples")
```
