"""
Image classification predictor class module.

Encapsulates prediction logic and manages model lifecycle.
"""

from typing import List, Dict, Any
from .inference import load_model, predict_single_image, predict_batch


class Predictor:
    """
    Image classification predictor.
    
    Encapsulates model loading and prediction, provides simple interface.
    
    Attributes:
        model_name: Model name
        weights_path: Path to weights file
        class_names: List of class names
        model: Keras model object
    
    Example:
        >>> predictor = Predictor("VGG16", "weights/vgg16.h5", class_names)
        >>> result = predictor.predict("test.jpg")
        >>> print(result['predicted_class'])
    """
    
    def __init__(self, model_name: str, weights_path: str, class_names: List[str]):
        """
        Initialize predictor.
        
        Args:
            model_name: Model name (VGG16/ResNet50/MobileNetV2)
            weights_path: Path to model weights file
            class_names: List of class names
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.class_names = class_names
        
        # Load model
        print(f"[Predictor] Loading {model_name}...")
        try:
            self.model = load_model(model_name, weights_path)
            print(f"[Predictor] Model loaded successfully!")
        except Exception as e:
            print(f"[Predictor] Loading failed: {e}")
            raise
    
    def predict(self, image_path: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict single image.
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions, default 3
        
        Returns:
            Prediction result dictionary
        """
        return predict_single_image(
            model=self.model,
            image_path=image_path,
            class_names=self.class_names,
            top_k=top_k
        )
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Batch predict multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction results
        """
        return predict_batch(
            model=self.model,
            image_paths=image_paths,
            class_names=self.class_names
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        total_params = self.model.count_params()
        trainable_params = sum([
            param.numpy().size
            for param in self.model.trainable_variables
        ])
        
        return {
            'model_name': self.model_name,
            'weights_path': self.weights_path,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params)
        }
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            print(f"[Predictor] {self.model_name} cleaned up")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"classes={len(self.class_names)}, "
            f"weights='{self.weights_path}')"
        )


# Module test
if __name__ == "__main__":
    print("=" * 50)
    print("Predictor Module Test")
    print("=" * 50)
    print("\n✓ Predictor class loaded successfully!")
    print("\nAvailable methods:")
    print("  - __init__(model_name, weights_path, class_names)")
    print("  - predict(image_path, top_k=3)")
    print("  - batch_predict(image_paths)")
    print("  - get_model_info()")
