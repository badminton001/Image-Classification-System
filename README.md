# Image Classification System
**Scene Recognition with Deep Learning (Intel Image Classification)**

This project is a comprehensive Python-based system for classifying images into 6 scene categories: **buildings, forest, glacier, mountain, sea, and street**. It utilizes Transfer Learning with state-of-the-art architectures (VGG16, ResNet50, MobileNetV2) and features a modular design for scalability.

---

## 🚀 Key Features

*   **Advanced Models**: Implements **VGG16**, **ResNet50**, and **MobileNetV2** using Transfer Learning.
*   **Robust Data Pipeline**:
    *   Automatic data loading and validation.
    *   Image processing (Resizing to 224x224, Normalization).
    *   Real-time Data Augmentation (Rotation, Zoom, Flip, Brightness).
*   **Training System**:
    *   Custom training loops with automated callbacks.
    *   Early Stopping & Learning Rate Decay.
    *   Model Checkpointing (saves best `.h5` weights).
*   **Interactive Interface**: A console-based menu system for easy navigation (Training, Evaluation, Prediction, Visualization).
*   **Modular Architecture**: Clean separation of concerns (Data, Models, UI, Utils, Prediction).

## 📂 Project Structure

```
Image-Classification-System/
├── data/                 # Data Loading & Preprocessing Module
├── models/               # Model Architecture & Training definitions
├── predict/              # Inference Engine
├── ui/                   # User Interface (Console Menu)
├── utils/                # Helper utilities (Logger, Visualization)
├── weights/              # Trained model artifacts (.h5 files)
├── results/              # Training history and logs
├── main.py               # Main Application Entry Point
├── run_training.py       # Standalone Training Script
└── config.py             # Global Configuration
```

## 🛠️ Setup & Installation

1.  **Prerequisites**:
    *   Python 3.8+
    *   TensorFlow
    *   NumPy, Pandas, Matplotlib, Scikit-learn, Pillow

2.  **Dataset**:
    *   Download the **Intel Image Classification** dataset.
    *   Ensure the dataset path is configured in `config.py` (or `main.py` entry settings).

## 💻 Usage

### 1. Main Application (Recommended)
Launch the interactive menu system to access all features:
```bash
python main.py
```
*   **Functions**: Load Data, Train Models, Evaluate, Predict (Single/Batch), Visualize Results.

### 2. Standalone Training
To run the training pipeline directly without the UI:
```bash
python run_training.py
```
*   This will train all defined models and save weights to `weights/`.

## 📊 Model Performance
The system supports training and evaluating the following architectures:
*   **MobileNetV2**: Lightweight, fast training (Recommended for CPU/Entry GPU).
*   **VGG16**: Classic deep CNN architecture.
*   **ResNet50**: Deep residual network for high accuracy.

## 📝 Configuration
*   **Global Settings**: `config.py`
*   **Model Parameters**: `models/config_models.py` (Adjust Batch Size, Epochs, Learning Rate).

---
*Developed for AIT102 - Python and TensorFlow Programming.*
