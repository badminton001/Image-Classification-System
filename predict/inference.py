"""
Inference module for image classification prediction.

This module provides functions for loading trained models, preprocessing images,
and making predictions using pre-trained models (VGG16, ResNet50, MobileNetV2).

Functions:
    - load_model: Load a trained Keras model from weights file
    - preprocess_input_image: Preprocess single image for prediction
    - predict_single_image: Predict single image and return top-k results
    - predict_batch: Batch prediction for multiple images
    - format_predictions: Format prediction results as readable string

Author: Member 4 - Prediction & Inference Team
Reference: Image Classification Enhancement System Project Guide
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
import tensorflow as tf
from tensorflow import keras

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_model(model_name: str, weights_path: str) -> keras.Model:
    """
    Load a trained Keras model from weights file.
    
    This function loads a pre-trained model (VGG16, ResNet50, or MobileNetV2)
    with custom classification layers from a saved .h5 weights file.
    
    Args:
        model_name (str): Name of the model ('VGG16', 'ResNet50', 'MobileNetV2')
        weights_path (str): Path to the model weights file (.h5 format)
        
    Returns:
        keras.Model: Loaded Keras model ready for prediction
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        Exception: If model loading fails
        
    Example:
        >>> model = load_model('VGG16', 'weights/vgg16_best_model.h5')
        >>> print(type(model))
        <class 'tensorflow.python.keras.engine.sequential.Sequential'>
        
    Reference:
        https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
    """
    # Validate weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Please ensure the model has been trained and saved."
        )
    
    try:
        # Load model from HDF5 file
        # Reference: TensorFlow documentation on model serialization
        model = keras.models.load_model(weights_path)
        print(f"✓ Successfully loaded {model_name} from {weights_path}")
        return model
        
    except Exception as e:
        raise Exception(
            f"Failed to load model {model_name}: {str(e)}\n"
            f"Ensure the weights file is a valid Keras model (.h5 format)"
        )


def preprocess_input_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for model prediction.
    
    Processing pipeline:
    1. Load image using PIL
    2. Resize to target size (224x224 for VGG16/ResNet50/MobileNetV2)
    3. Convert to numpy array
    4. Normalize pixel values to [0, 1] range
    5. Add batch dimension (1, height, width, channels)
    
    Args:
        image_path (str): Path to the image file (JPG, PNG, etc.)
        target_size (Tuple[int, int]): Target image size (height, width), default (224, 224)
        
    Returns:
        np.ndarray: Preprocessed image array with shape (1, 224, 224, 3)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image processing fails
        
    Example:
        >>> img_array = preprocess_input_image('test.jpg')
        >>> print(img_array.shape)
        (1, 224, 224, 3)
        >>> print(img_array.min(), img_array.max())
        0.0 1.0
        
    Reference:
        PIL Image Processing: https://pillow.readthedocs.io/en/stable/
        NumPy expand_dims: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    """
    # Validate image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # 1. Load image using PIL and convert to RGB
        # Reference: PIL documentation on image modes
        img = Image.open(image_path).convert('RGB')
        
        # 2. Resize image to target size using LANCZOS resampling
        # LANCZOS provides high-quality downsampling
        # Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
        img = img.resize(target_size, Image.LANCZOS)
        
        # 3. Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # 4. Normalize pixel values from [0, 255] to [0, 1]
        # This matches the preprocessing used during training
        img_array = img_array.astype('float32') / 255.0
        
        # 5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        # Models expect input shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Predict the class of a single image and return top-k results.
    
    This function performs the following steps:
    1. Preprocess the input image
    2. Run model prediction to get class probabilities
    3. Extract top-k predictions sorted by confidence
    4. Format results into a structured dictionary
    
    Args:
        model (keras.Model): Loaded Keras model
        image_path (str): Path to the image file
        class_names (List[str]): List of class names (e.g., ['buildings', 'forest', ...])
        top_k (int): Number of top predictions to return, default 3
        
    Returns:
        Dict[str, Any]: Prediction results with the following keys:
            - 'image_path' (str): Path to the input image
            - 'predicted_class' (str): Top predicted class name
            - 'confidence' (float): Confidence score [0, 1] for top prediction
            - 'top_k_predictions' (List[Tuple[str, float]]): Top-k (class, probability) pairs
            
    Example:
        >>> result = predict_single_image(model, 'forest.jpg', class_names, top_k=3)
        >>> print(result)
        {
            'image_path': 'forest.jpg',
            'predicted_class': 'forest',
            'confidence': 0.9523,
            'top_k_predictions': [
                ('forest', 0.9523),
                ('mountain', 0.0312),
                ('glacier', 0.0105)
            ]
        }
        
    Reference:
        NumPy argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        Model prediction: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
    """
    # Preprocess the image
    img_array = preprocess_input_image(image_path)
    
    # Run model prediction
    # verbose=0 suppresses progress bar for single prediction
    # Output shape: (1, num_classes)
    predictions = model.predict(img_array, verbose=0)[0]  # Get first (and only) result
    
    # Get indices of top-k predictions sorted by probability (descending)
    # argsort returns indices in ascending order, so we reverse with [::-1]
    # Reference: https://stackoverflow.com/questions/6910641/
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    
    # Create list of (class_name, probability) tuples for top-k predictions
    top_k_preds = [
        (class_names[i], float(predictions[i]))  # Convert np.float32 to Python float
        for i in top_k_indices
    ]
    
    # Get the top prediction (highest confidence)
    best_class_idx = top_k_indices[0]
    
    # Return structured prediction results
    return {
        'image_path': image_path,
        'predicted_class': class_names[best_class_idx],
        'confidence': float(predictions[best_class_idx]),
        'top_k_predictions': top_k_preds
    }


def predict_batch(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Batch prediction for multiple images.
    
    This function processes multiple images and returns predictions for each.
    It uses tqdm to display a progress bar if available.
    
    Args:
        model (keras.Model): Loaded Keras model
        image_paths (List[str]): List of image file paths
        class_names (List[str]): List of class names
        
    Returns:
        List[Dict[str, Any]]: List of prediction dictionaries, one per image
        
    Example:
        >>> paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        >>> results = predict_batch(model, paths, class_names)
        >>> print(f"Processed {len(results)} images")
        Processed 3 images
        
    Reference:
        tqdm documentation: https://tqdm.github.io/
    """
    results = []
    
    # Create iterator with progress bar if tqdm is available
    if TQDM_AVAILABLE:
        iterator = tqdm(image_paths, desc="Predicting images", unit="img")
    else:
        iterator = image_paths
        print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for i, image_path in enumerate(iterator):
        try:
            # Predict single image with top-1 result (faster for batch)
            result = predict_single_image(model, image_path, class_names, top_k=1)
            results.append(result)
            
        except Exception as e:
            # Log error but continue processing other images
            print(f"\n⚠️  Error processing {image_path}: {e}")
            # Add placeholder result for failed image
            results.append({
                'image_path': image_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'top_k_predictions': [],
                'error': str(e)
            })
        
        # Print progress if tqdm not available
        if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(image_paths)} images")
    
    if not TQDM_AVAILABLE:
        print(f"✓ Completed: {len(results)} images processed")
    
    return results


def format_predictions(prediction_dict: Dict[str, Any]) -> str:
    """
    Format prediction results as a human-readable string.
    
    This function takes a prediction dictionary and formats it into
    a nicely formatted string for display to users.
    
    Args:
        prediction_dict (Dict[str, Any]): Prediction result from predict_single_image()
        
    Returns:
        str: Formatted prediction result string
        
    Example:
        >>> result = predict_single_image(model, 'forest.jpg', class_names)
        >>> print(format_predictions(result))
        ==================
        Prediction Result
        ==================
        Image: forest.jpg
        Predicted Class: forest
        Confidence: 95.23%
        
        Top-3 Predictions:
          1. forest        - 95.23%
          2. mountain      -  3.12%
          3. glacier       -  1.05%
    """
    # Check if this is an error result
    if 'error' in prediction_dict:
        return f"""
==================
Prediction Error
==================
Image: {prediction_dict['image_path']}
Error: {prediction_dict['error']}
"""
    
    # Format header
    output = """
==================
Prediction Result
==================
"""
    
    # Add image path
    output += f"Image: {prediction_dict['image_path']}\n"
    
    # Add top prediction
    output += f"Predicted Class: {prediction_dict['predicted_class']}\n"
    output += f"Confidence: {prediction_dict['confidence']:.2%}\n\n"
    
    # Add top-k predictions if available
    top_k_preds = prediction_dict.get('top_k_predictions', [])
    if top_k_preds:
        output += f"Top-{len(top_k_preds)} Predictions:\n"
        for i, (class_name, probability) in enumerate(top_k_preds, 1):
            # Format: "  1. forest        - 95.23%"
            output += f"  {i}. {class_name:15s} - {probability:6.2%}\n"
    
    return output


# Module-level test code (can be commented out for production)
if __name__ == "__main__":
    print("=" * 50)
    print("Inference Module Test")
    print("=" * 50)
    
    # This is for testing purposes only
    # In production, this module will be imported by other scripts
    print("\n✓ Module loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_model()")
    print("  - preprocess_input_image()")
    print("  - predict_single_image()")
    print("  - predict_batch()")
    print("  - format_predictions()")
    print("\nImport this module in your main script to use these functions.")
