# Image-Classification-System
Multi-model comparison image classification system using TensorFlow
# Image Classification System

## 1. Project Overview
This project is a Python-based application developed for the **AIT102 (Python and TensorFlow Programming)** course.

The system uses Deep Learning models (VGG16 and ResNet50) to automatically classify images into specific categories. It aims to demonstrate the practical application of Computer Vision techniques, including data processing, model training, evaluation, and GUI development.

## 2. Dataset Information
We utilize the **Intel Image Classification Dataset** for Scene Recognition.

* **Source:** Intel Image Classification (Kaggle)
* **Total Images:** ~14,000 images
* **Categories (6 Classes):**
    * `buildings`
    * `forest`
    * `glacier`
    * `mountain`
    * `sea`
    * `street`
* **Data Processing:**
    The dataset is automatically processed and split into:
    * **Training Set (70%)**: For model learning.
    * **Validation Set (15%)**: For hyperparameter tuning.
    * **Test Set (15%)**: For final evaluation.

## 3. Project Structure & Features

### Data Module (`/data`)
* **Manual Data Loading:** Implements image loading using `PIL` without relying on high-level Keras APIs.
* **Preprocessing:** Includes image resizing (224x224) and pixel normalization.
* **Augmentation:** Applies rotation, zoom, and flips to improve model generalization.

### Model Module (`/models`)
* **Transfer Learning:** Utilizes pre-trained models (VGG16, ResNet50) with custom classification layers.
* **Training:** Implements the training loop with `Adam` optimizer and `Sparse Categorical Crossentropy` loss.

### Evaluation Module
* Comparisons of model performance (Accuracy/Loss curves).
* Confusion Matrix analysis.

### Application (GUI)
* A user-friendly interface to upload images and view classification results in real-time.

## 4. Requirements
* Python 3.x
* TensorFlow
* NumPy
* Pandas
* Matplotlib
* Pillow (PIL)
* Scikit-learn

## 5. Team Members
* **Member 1:** Data Loading, Preprocessing & Augmentation
* **Member 2:** Model Architecture & Training
* **Member 3:** Model Evaluation & Comparison
* **Member 4:** Visualization & PPT
* **Member 5:** GUI Application Development
* **Group Leader:** Project Management & Coordination
