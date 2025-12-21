# Image Classification System
### Scene Recognition with Deep Learning (Intel Image Classification)

This project is a comprehensive Python-based system for classifying images into 6 scene categories: **buildings, forest, glacier, mountain, sea, and street**. Result of a group project for [Your Course Name], [Year].

## Requirements (User Perspective)

1. **Minimum Hardware**: 
   - **CPU**: Intel Core i5 or equivalent.
   - **RAM**: 8GB minimum recommended.
   - **GPU**: Optional but recommended for faster training (NVIDIA GTX 1050 or higher).

2. **Software**:
   - **OS**: Windows 10/11, macOS, or Linux.
   - **Python 3.8** or higher installed and added to PATH.
   - **Key Libraries**: TypeScript, Keras, NumPy, Pillow, Matplotlib, Scikit-learn.
   - See `requirements.txt` for specific versions.

## How the Program Works

1. **Launch the Application**:
   - Run `python main.py` in your terminal.
   - A graphical interface (GUI) will open.

2. **Load a Model**:
   - Click "**Browse**" next to Weights file to select a trained model (.h5) from the `weights/` folder.
   - Click "**Load Model**". The system supports MobileNetV2, VGG16, and ResNet50.

3. **Make Predictions**:
   - **Single Image**: Click "**Browse**" inside the Prediction frame to select an image file (.jpg, .png). Click "**Predict Image**" to see the classification result.
   - **Visualization**: Click "**Visualization (Single Image)**" to generate probability charts and overlay images in the `results/` folder.
   - **Batch Prediction**: Select a folder of images to process them all at once. The system will save a JSON report and generate a confusion matrix.

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
- Concepts adapted from TensorFlow official tutorials and Keras documentation.
