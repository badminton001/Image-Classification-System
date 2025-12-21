# User Interface (UI) Module

This module contains the code for the desktop Graphical User Interface (GUI), built using Python's built-in **Tkinter** library.

## Components

- **`interface.py`**:
  - The main application entry point for the GUI.
  - **`ImageClassifierApp` Class**: Manages the window lifecycle, UI layout, and event handling.
  - connects the UI controls (Buttons, Entry fields) to the backend logic in `predict/` and `utils/`.

## Features

1.  **Model Management**:
    - Browse and load trained model weights (`.h5` files).
    - Auto-detects available models in the `weights/` directory.

2.  **Single Image Prediction**:
    - Load and display a single image.
    - View top-K predicted classes and probabilities.
    - **Visualization**: Generate Prediction Distribution charts and Class Activation Overlays.

3.  **Batch Prediction**:
    - Select a folder containing multiple images.
    - Process all images in bulk.
    - Generate statistical reports and Confusion Matrices.

4.  **Activity Log**:
    - Real-time logging of application status, errors, and prediction results.

## Usage

The UI is typically launched via the root `main.py` script:

```bash
python main.py
```
