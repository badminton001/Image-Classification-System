"""
Inference module for image classification prediction.

Provides model loading, image preprocessing, single/batch prediction, and result formatting.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from tensorflow import keras

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_model(model_name: str, weights_path: str) -> keras.Model:

    """Load trained Keras model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    try:
        model = keras.models.load_model(weights_path)
        print(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def preprocess_input_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:

    """Preprocess single image: Load -> Resize -> Normalize."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype='float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str],
    top_k: int = 3
) -> Dict[str, Any]:

    """Predict class of single image."""
    # Preprocess image
    img_array = preprocess_input_image(image_path)
    
    # Model prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top-K results
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_preds = [(class_names[i], float(predictions[i])) for i in top_k_indices]
    
    # Return results
    best_idx = top_k_indices[0]
    return {
        'image_path': image_path,
        'predicted_class': class_names[best_idx],
        'confidence': float(predictions[best_idx]),
        'top_k_predictions': top_k_preds
    }


def predict_batch(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str]
) -> List[Dict[str, Any]]:
def predict_batch(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """Batch prediction for multiple images."""
    results = []
    
    # Create progress bar
    iterator = tqdm(image_paths, desc="Predicting", unit="img") if TQDM_AVAILABLE else image_paths
    if not TQDM_AVAILABLE:
        print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for i, image_path in enumerate(iterator):
        try:
            result = predict_single_image(model, image_path, class_names, top_k=1)
            results.append(result)
        except Exception as e:
            print(f"\nFailed {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
        
        # Progress update (without tqdm)
        if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(image_paths)}")
    
    if not TQDM_AVAILABLE:
        print(f"Completed: {len(results)} images")
    
    return results


def format_predictions(prediction_dict: Dict[str, Any]) -> str:

    """Format prediction results as readable string."""
    # Error case
    if 'error' in prediction_dict:
    if 'error' in prediction_dict:
        return f"\nPrediction Error\nImage: {prediction_dict['image_path']}\nError: {prediction_dict['error']}\n"
    
    # Normal results
    output = f"\nPrediction Result\nImage: {prediction_dict['image_path']}\nPredicted Class: {prediction_dict['predicted_class']}\nConfidence: {prediction_dict['confidence']:.2%}\n"
    
    # Add top-K results
    top_k_preds = prediction_dict.get('top_k_predictions', [])
    if top_k_preds:
        output += f"\nTop-{len(top_k_preds)} Predictions:\n"
        for i, (class_name, prob) in enumerate(top_k_preds, 1):
            output += f"  {i}. {class_name:15s} - {prob:6.2%}\n"
    
    return output


# Module test
if __name__ == "__main__":
    print("Inference Module Test")
    print("\nModule loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_model()")
    print("  - preprocess_input_image()")
    print("  - predict_single_image()")
    print("  - predict_batch()")
    print("  - format_predictions()")
