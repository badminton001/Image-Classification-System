"""
Test script for the predict module (inference.py and predictor.py).

This script demonstrates the usage of prediction functions and provides
basic testing for the implemented functionality.

Note: This script requires:
1. A trained model weights file (e.g., 'weights/vgg16_best_model.h5')
2. Test images in the data_samples/ directory
3. All dependencies installed (see requirements.txt)

Usage:
    python scripts/test_predict_module.py
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_inference_functions():
    """Test individual inference functions."""
    print("=" * 60)
    print("TEST 1: Inference Functions")
    print("=" * 60)
    
    from predict.inference import (
        load_model,
        preprocess_input_image,
        predict_single_image,
        predict_batch,
        format_predictions
    )
    
    # Configuration
    model_name = "VGG16"
    weights_path = "weights/vgg16_best_model.h5"
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    test_image = "data_samples/forest/test1.jpg"  # Replace with actual path
    
    try:
        # Test 1: Load model
        print("\n[Test 1.1] Loading model...")
        model = load_model(model_name, weights_path)
        print(f"✓ Model loaded: {type(model)}")
        
        # Test 2: Preprocess image
        print("\n[Test 1.2] Preprocessing image...")
        img_array = preprocess_input_image(test_image)
        print(f"✓ Image preprocessed: shape={img_array.shape}, "
              f"range=[{img_array.min():.3f}, {img_array.max():.3f}]")
        assert img_array.shape == (1, 224, 224, 3), "Wrong image shape!"
        assert 0 <= img_array.min() and img_array.max() <= 1, "Wrong normalization!"
        
        # Test 3: Single prediction
        print("\n[Test 1.3] Single image prediction...")
        result = predict_single_image(model, test_image, class_names, top_k=3)
        print("✓ Prediction result:")
        print(f"  - Predicted class: {result['predicted_class']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Top-3: {result['top_k_predictions']}")
        
        # Test 4: Format output
        print("\n[Test 1.4] Formatted output...")
        formatted = format_predictions(result)
        print(formatted)
        
        # Test 5: Batch prediction (if multiple images available)
        print("\n[Test 1.5] Batch prediction...")
        batch_paths = [test_image]  # Add more paths if available
        results = predict_batch(model, batch_paths, class_names)
        print(f"✓ Batch processed: {len(results)} images")
        
        print("\n✅ All inference function tests PASSED!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Skipping inference function tests.")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictor_class():
    """Test Predictor class."""
    print("\n" + "=" * 60)
    print("TEST 2: Predictor Class")
    print("=" * 60)
    
    from predict.predictor import Predictor
    
    # Configuration
    model_name = "VGG16"
    weights_path = "weights/vgg16_best_model.h5"
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    test_image = "data_samples/forest/test1.jpg"
    
    try:
        # Test 1: Initialize Predictor
        print("\n[Test 2.1] Initializing Predictor...")
        predictor = Predictor(model_name, weights_path, class_names)
        print(f"✓ Predictor initialized: {predictor}")
        
        # Test 2: Get model info
        print("\n[Test 2.2] Getting model info...")
        info = predictor.get_model_info()
        print("✓ Model information:")
        for key, value in info.items():
            if key != 'class_names':  # Skip long list
                print(f"  - {key}: {value}")
        
        # Test 3: Single prediction
        print("\n[Test 2.3] Single prediction with Predictor...")
        result = predictor.predict(test_image, top_k=3)
        print("✓ Prediction result:")
        print(f"  - Class: {result['predicted_class']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        
        # Test 4: Batch prediction
        print("\n[Test 2.4] Batch prediction with Predictor...")
        results = predictor.batch_predict([test_image])
        print(f"✓ Batch processed: {len(results)} images")
        
        # Test 5: Cleanup (test __del__)
        print("\n[Test 2.5] Testing cleanup...")
        del predictor
        print("✓ Predictor deleted")
        
        print("\n✅ All Predictor class tests PASSED!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Skipping Predictor class tests.")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    print("\n" + "=" * 60)
    print("PREDICT MODULE TEST SUITE")
    print("=" * 60)
    print("\nThis script tests the prediction module functionality.")
    print("Note: You need trained model weights and test images to run all tests.\n")
    
    # Check if we're in the project root
    if not os.path.exists("predict"):
        print("❌ Error: predict/ directory not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Run tests
    results = {}
    results['inference'] = test_inference_functions()
    results['predictor'] = test_predictor_class()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed or were skipped.")
        print("\nCommon issues:")
        print("  1. No trained model weights (train a model first)")
        print("  2. No test images (add images to data_samples/)")
        print("  3. Missing dependencies (run: pip install -r requirements.txt)")


if __name__ == "__main__":
    main()
