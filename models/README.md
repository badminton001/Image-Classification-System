# Model Engineering Module

## Overview
This module defines, compiles, and trains the Deep Learning models. It leverages Transfer Learning with three pre-trained architectures: **VGG16**, **ResNet50**, and **MobileNetV2**.

## File Structure

### `config_models.py`
- Centralized configuration for model hyperparameters.
- **Settings**: Input Shape (224x224x3), Batch Size (32), Epochs (50), Learning Rate (0.001).
- **Classes**: 6 scene categories.

### `model_architecture.py`
- **Factory Functions**: `build_vgg16_model`, `build_resnet50_model`, `build_mobilenetv2_model`.
- **Architecture**:
  - **Base**: Pre-trained on ImageNet (weights frozen).
  - **Head**: Custom GlobalAveragePooling > Dense(256) > Dropout(0.5) > Output(Softmax).

### `train.py`
- Manages the training loop and callback configuration.
- **Callbacks**:
  - `EarlyStopping`: Prevents overfitting.
  - `ReduceLROnPlateau`: Optimizes learning rate dynamically.
  - `ModelCheckpoint`: Auto-saves the best model weights.
- **Output**: Saves training history to JSON for visualization.

## Usage

```python
from models import build_mobilenetv2_model, compile_model, train_single_model

# Build and Compile
model = build_mobilenetv2_model(num_classes=6)
model = compile_model(model)

# Train
history = train_single_model(model, 'mobilenetv2', train_gen, val_gen)
```
