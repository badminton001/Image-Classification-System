"""
Predictor module for image classification.

This module provides a Predictor class that encapsulates the prediction logic
for image classification using pre-trained models (VGG16, ResNet50, MobileNetV2).

The Predictor class manages the model lifecycle and provides convenient methods
for single and batch predictions.

Classes:
    Predictor: Main prediction class for image classification

Author: Member 4 - Prediction & Inference Team
Reference: Image Classification Enhancement System Project Guide
"""

from typing import List, Dict, Any
from .inference import (
    load_model,
    predict_single_image,
    predict_batch
)


class Predictor:
    """
    Image classification predictor class.
    
    This class encapsulates model loading and prediction logic, managing
    the model lifecycle and providing convenient prediction methods.
    
    Attributes:
        model_name (str): Name of the loaded model (e.g., 'VGG16')
        weights_path (str): Path to the model weights file
        class_names (List[str]): List of class names
        model: Loaded Keras model object
        
    Example Usage:
        >>> # Initialize predictor
        >>> predictor = Predictor(
        ...     model_name="VGG16",
        ...     weights_path="weights/vgg16_best_model.h5",
        ...     class_names=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        ... )
        >>> 
        >>> # Single prediction
        >>> result = predictor.predict("test_image.jpg", top_k=3)
        >>> print(result['predicted_class'])
        'forest'
        >>> 
        >>> # Batch prediction
        >>> results = predictor.batch_predict(['img1.jpg', 'img2.jpg'])
        >>> print(f"Processed {len(results)} images")
        Processed 2 images
        >>> 
        >>> # Get model info
        >>> info = predictor.get_model_info()
        >>> print(info['model_name'])
        'VGG16'
        
    Reference:
        Object-Oriented Programming in Python: 
        https://docs.python.org/3/tutorial/classes.html
    """
    
    def __init__(
        self,
        model_name: str,
        weights_path: str,
        class_names: List[str]
    ):
        """
        Initialize the Predictor with a trained model.
        
        This constructor loads the specified model and stores the configuration
        for later use in predictions.
        
        Args:
            model_name (str): Name of the model ('VGG16', 'ResNet50', 'MobileNetV2')
            weights_path (str): Path to the model weights file (.h5 format)
            class_names (List[str]): List of class names in order
                                    (e.g., ['buildings', 'forest', ...])
                                    
        Raises:
            FileNotFoundError: If weights file doesn't exist
            Exception: If model loading fails
            
        Example:
            >>> predictor = Predictor(
            ...     "VGG16",
            ...     "weights/vgg16_best_model.h5",
            ...     ['class1', 'class2', 'class3']
            ... )
            [Predictor] Loading VGG16...
            ✓ Successfully loaded VGG16 from weights/vgg16_best_model.h5
            [Predictor] Model loaded successfully!
        """
        # Store configuration
        self.model_name = model_name
        self.weights_path = weights_path
        self.class_names = class_names
        
        # Load the model
        print(f"[Predictor] Loading {model_name}...")
        try:
            self.model = load_model(model_name, weights_path)
            print(f"[Predictor] Model loaded successfully!")
        except Exception as e:
            print(f"[Predictor] Error loading model: {e}")
            raise
    
    def predict(
        self,
        image_path: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict the class of a single image.
        
        This method is a convenient wrapper around predict_single_image()
        that uses the loaded model and class names.
        
        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return (default: 3)
            
        Returns:
            Dict[str, Any]: Prediction results containing:
                - 'image_path': Path to the image
                - 'predicted_class': Top predicted class name
                - 'confidence': Confidence score [0, 1]
                - 'top_k_predictions': List of (class, probability) tuples
                
        Example:
            >>> result = predictor.predict("forest.jpg", top_k=3)
            >>> print(f"Predicted: {result['predicted_class']}")
            Predicted: forest
            >>> print(f"Confidence: {result['confidence']:.2%}")
            Confidence: 95.23%
            
        Reference:
            See inference.predict_single_image() for implementation details
        """
        return predict_single_image(
            model=self.model,
            image_path=image_path,
            class_names=self.class_names,
            top_k=top_k
        )
    
    def batch_predict(
        self,
        image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict classes for multiple images (batch prediction).
        
        This method processes multiple images and returns predictions for each.
        It displays a progress bar if tqdm is available.
        
        Args:
            image_paths (List[str]): List of image file paths
            
        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries, one per image
            
        Example:
            >>> paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> results = predictor.batch_predict(paths)
            Predicting images: 100%|██████████| 3/3 [00:02<00:00,  1.2img/s]
            >>> for result in results:
            ...     print(f"{result['image_path']}: {result['predicted_class']}")
            img1.jpg: forest
            img2.jpg: mountain
            img3.jpg: sea
            
        Reference:
            See inference.predict_batch() for implementation details
        """
        return predict_batch(
            model=self.model,
            image_paths=image_paths,
            class_names=self.class_names
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information:
                - 'model_name': Name of the model
                - 'weights_path': Path to weights file
                - 'num_classes': Number of classification classes
                - 'class_names': List of class names
                - 'input_shape': Expected input shape
                - 'output_shape': Model output shape
                - 'total_parameters': Total number of model parameters
                - 'trainable_parameters': Number of trainable parameters
                
        Example:
            >>> info = predictor.get_model_info()
            >>> print(f"Model: {info['model_name']}")
            Model: VGG16
            >>> print(f"Classes: {info['num_classes']}")
            Classes: 6
            >>> print(f"Input shape: {info['input_shape']}")
            Input shape: (None, 224, 224, 3)
            >>> print(f"Parameters: {info['total_parameters']:,}")
            Parameters: 14,714,688
            
        Reference:
            Keras Model API: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        """
        # Count total and trainable parameters
        # Reference: https://stackoverflow.com/questions/46160207/
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
        """
        Destructor: Clean up resources when the object is destroyed.
        
        This method is called when the Predictor object is about to be destroyed.
        It releases the model from memory to free up resources.
        
        Note:
            Python's garbage collector will automatically call this method
            when the object is no longer referenced.
            
        Example:
            >>> predictor = Predictor("VGG16", "weights/vgg16.h5", classes)
            >>> del predictor  # Explicitly delete
            [Predictor] VGG16 cleaned up.
            
        Reference:
            Python __del__ method: https://docs.python.org/3/reference/datamodel.html#object.__del__
        """
        if hasattr(self, 'model'):
            # Delete model to free memory
            del self.model
            print(f"[Predictor] {self.model_name} cleaned up.")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Predictor instance.
        
        Returns:
            str: String representation for debugging
            
        Example:
            >>> predictor = Predictor("VGG16", "weights/vgg16.h5", classes)
            >>> print(predictor)
            Predictor(model='VGG16', classes=6, weights='weights/vgg16.h5')
        """
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"classes={len(self.class_names)}, "
            f"weights='{self.weights_path}')"
        )


# Module-level test code (can be commented out for production)
if __name__ == "__main__":
    print("=" * 50)
    print("Predictor Module Test")
    print("=" * 50)
    
    # This is for testing purposes only
    # In production, this module will be imported by other scripts
    print("\n✓ Predictor class loaded successfully!")
    print("\nAvailable methods:")
    print("  - __init__(model_name, weights_path, class_names)")
    print("  - predict(image_path, top_k=3)")
    print("  - batch_predict(image_paths)")
    print("  - get_model_info()")
    print("  - __del__()")
    print("\nExample usage:")
    print("  predictor = Predictor('VGG16', 'weights/vgg16.h5', class_names)")
    print("  result = predictor.predict('test.jpg', top_k=3)")
    print("  results = predictor.batch_predict(['img1.jpg', 'img2.jpg'])")
