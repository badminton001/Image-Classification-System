"""
Inference module for image classification prediction.

Provides model loading, image preprocessing, single/batch prediction, and result formatting.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from tensorflow import keras
from config import TARGET_IMAGE_SIZE

try:
    from models.model_architecture import build_vgg16_model, build_resnet50_model, build_mobilenetv2_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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
        # Standard load
        model = keras.models.load_model(weights_path)
        print(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        error_str = str(e)
        if "batch_shape" in error_str or "DTypePolicy" in error_str or "Unknown dtype policy" in error_str or "InputLayer" in error_str:
            print(f"[Warning] Version mismatch detected ({error_str}). Attempting compatibility fix...")
            
            # Compatibility patches for version mismatches
            class CompatibleInputLayer(keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
                        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                    kwargs.pop('batch_shape', None)
                    super().__init__(*args, **kwargs)

            class MockDTypePolicy:
                def __init__(self, name="float32", **kwargs):
                    self._name = name
                    self._compute_dtype = name
                    self._variable_dtype = name

                @property
                def name(self):
                    return self._name
                
                @property
                def compute_dtype(self):
                    return self._compute_dtype
                
                @property
                def variable_dtype(self):
                    return self._variable_dtype

                @classmethod
                def from_config(cls, config):
                    if isinstance(config, dict):
                        return cls(**config)
                    return cls(name=str(config))
                    
                def get_config(self):
                    return {"name": self._name}
                
                # Catch specific as_list calls if necessary
                def __getattr__(self, name):
                    if name == "as_list":
                        return lambda: []
                    raise AttributeError(f"'MockDTypePolicy' object has no attribute '{name}'")
                
                def __str__(self):
                    return self._name
                    
                def __repr__(self):
                    return f"<MockDTypePolicy: {self._name}>"

            try:
                # Attempt load with patches
                model = keras.models.load_model(
                    weights_path, 
                    custom_objects={
                        'InputLayer': CompatibleInputLayer,
                        'DTypePolicy': MockDTypePolicy
                    },
                    compile=False
                )
                print(f"Successfully loaded {model_name} with compatibility patches")
                return model
            except Exception as e3:
                # Fallback: Reconstruct model structure and load weights directly
                if ARCH_AVAILABLE:
                    try:
                        print(f"[Info] Patch failed ({e3}). Switching to reconstruction strategy...")
                        name_lower = model_name.lower()
                        model_builder = None
                        if "vgg" in name_lower: model_builder = build_vgg16_model
                        elif "resnet" in name_lower: model_builder = build_resnet50_model
                        elif "mobilenet" in name_lower: model_builder = build_mobilenetv2_model
                        
                        if model_builder:
                            candidates = [2] + list(range(3, 21)) + [100, 1000]
                            
                            for n_classes in candidates:
                                try:
                                    temp_model = model_builder(num_classes=n_classes)
                                    temp_model.load_weights(weights_path)
                                    print(f"Successfully reconstructed {model_name} with {n_classes} classes")
                                    return temp_model
                                except Exception:
                                    continue
                    except Exception as e_recon:
                        print(f"[Error] Reconstruction failed: {e_recon}")

                raise Exception(f"Failed to load model (all strategies failed): {e3}")
        else:
            raise Exception(f"Failed to load model: {error_str}")


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
