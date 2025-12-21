# Image Classification System
### Scene Recognition with Deep Learning (Intel Image Classification)

This project is a comprehensive Python-based system for classifying images into 6 scene categories: **buildings, forest, glacier, mountain, sea, and street**.

## Requirements

1. **Software Environment**:
   - **Python 3.8** or higher installed and added to PATH.
   - **Key Libraries**: TensorFlow, Keras, NumPy, Pillow, Matplotlib, Scikit-learn.
   - See `requirements.txt` for specific versions.

2. **Dataset**:
   - The system expects the 'Intel Image Classification' dataset.
   - Ensure data is placed in the `data_samples/` directory or configured path.

## User Manual

1. **Launch the Application**:
   - Run `python main.py` in your terminal.
   - A graphical interface (GUI) will open.

2. **Load a Model**:
   - Check the "**Weights file**" dropdown in the Prediction section.
   - If empty, click "**Browse**" to select a trained model (.h5) from the `weights/` folder.
   - Click "**Load Model**" to initialize the inference engine. The system supports MobileNetV2, VGG16, and ResNet50.

3. **Make Predictions (Single Image)**:
   - Click "**Browse**" inside the Prediction frame to select an image file (.jpg, .png).
   - Enter a value for "**Top-K**" (default is 3) to see the top predicted classes.
   - Click "**Predict Image**". The result and confidence score will appear in the log.
   - Click "**Visualization (Single Image)**" to generate a probability bar chart and an overlay image in the `results/` folder.

4. **Batch Prediction (Multiple Images)**:
   - Click "**Browse**" next to "Batch folder" to select a directory containing test images.
   - Click "**Batch Predict**". The system will process all images and save predictions to `results/batch_predictions.json`.
   - Click "**Batch Visualization (Batch Results)**" to generate a Confusion Matrix and Class Distribution plot.

## Project Structure

```text
Image-Classification-System/
├── data/                 # Data Loading, Preprocessing, and Augmentation
├── models/               # Model Architectures (VGG16, ResNet50, MobileNetV2) & Training
├── predict/              # Inference Engine & Prediction Logic
├── ui/                   # User Interface (Tkinter GUI)
├── utils/                # Helper Utilities (Logger, Visualization)
├── weights/              # Trained Model Artifacts
├── results/              # Output Directory (Logs, Plots, Reports)
├── main.py               # Main Application Entry Point
├── run_training.py       # Standalone Training Script
├── config.py             # Global Configuration
└── requirements.txt      # Python Dependencies
```

## Credits & Attribution

- **Dataset**: Intel Image Classification (Kaggle)
- **Deep Learning Framework**: TensorFlow & Keras
- **GUI Framework**: Tkinter
