# Model Engineering Module Documentation

## Overview
This module (`models/`) is responsible for defining, compiling, and training the deep learning models for the Image Classification System. It implements transfer learning using three state-of-the-art architectures: VGG16, ResNet50, and MobileNetV2.

## File Structure

### `models/config_models.py`
Contains global configuration constants for the model training process.
- **Input Shape**: (224, 224, 3)
- **Hyperparameters**: Learning rate (0.001), Batch size (32), Epochs (50).
- **Classes**: 6 categories (buildings, forest, glacier, mountain, sea, street).

### `models/model_architecture.py`
Defines the factory functions for building models.
- **Method**: Transfer Learning (ImageNet weights).
- **Base Models**: VGG16, ResNet50, MobileNetV2 (Frozen).
- **HEAD (Custom Top Layers)**:
    - GlobalAveragePooling2D
    - Dense (256 units, ReLU)
    - Dropout (0.5)
    - Dense (Output, Softmax)

### `models/train.py`
Manages the training workflow.
- **Callbacks**:
    - `EarlyStopping`: Monitors validation loss (patience=5).
    - `ReduceLROnPlateau`: Decays learning rate on stagnation.
    - `ModelCheckpoint`: Saves best weights (`.h5`).
- **Persistence**: Saves training history to `results/training_history.json`.

## Usage

### Training Interface
The training process is exposed via `run_training.py` or the `train_all_models` function in `models/train.py`.

```python
from models import build_mobilenetv2_model, compile_model, train_single_model

# Example Usage
model = build_mobilenetv2_model(num_classes=6)
model = compile_model(model)
history = train_single_model(model, 'mobilenetv2', train_dataset, val_dataset)
```

## Algorithms
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
