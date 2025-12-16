import os
import shutil
import numpy as np
from PIL import Image
import unittest

# Import project modules
from data.data_loader import load_dataset_from_directory, validate_dataset
from data.preprocessing import prepare_dataset
from data.augmentation import create_augmented_train_generator

class TestDataModule(unittest.TestCase):
    """
    [Functionality]: Automatically tests all functions of the data processing module.
    [Principle]: Create a temporary fake dataset -> Run code -> Check results -> Delete fake data.
    """

    def setUp(self):
        """Preparation before tests: create fake data"""
        self.test_dir = "temp_test_data"
        self.classes = ["cat", "dog"]
        
        # 1. Create temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # 2. Create 10 fake images per class
        for class_name in self.classes:
            class_path = os.path.join(self.test_dir, class_name)
            os.makedirs(class_path)
            for i in range(10):
                # Create a 100x100 random RGB image
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(class_path, f"img_{i}.jpg"))
                
        print(f"\n[Setup] Created temporary dataset at {self.test_dir}")

    def tearDown(self):
        """Cleanup after tests: remove fake data"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("[Teardown] Removed temporary dataset")

    def test_1_data_loader(self):
        """Test data loader"""
        print("Testing Data Loader...")
        paths, labels, class_names = load_dataset_from_directory(self.test_dir)
        
        # Assert checks: Should have 20 images (10 cat + 10 dog)
        self.assertEqual(len(paths), 20)
        # Assert checks: Should have 2 classes
        self.assertEqual(len(class_names), 2)
        print("Data Loader Test Passed!")

    def test_2_preprocessing(self):
        """Test preprocessing capability (split + normalize + resize)"""
        print("Testing Preprocessing...")
        # Run core function
        (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = prepare_dataset(
            self.test_dir, target_size=(50, 50), test_size=0.2, val_size=0.2
        )
        
        # Check return
        self.assertIsNotNone(X_train)
        
        # Check shape: X should be (count, 50, 50, 3)
        self.assertEqual(X_train.shape[1:], (50, 50, 3))
        
        # Check normalization: max value should not exceed 1.0
        self.assertTrue(X_train.max() <= 1.0)
        
        print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print("Preprocessing Test Passed!")

    def test_3_augmentation(self):
        """Test data augmentation generator"""
        print("Testing Augmentation...")
        # Create dummy data for generator
        X_dummy = np.random.rand(10, 50, 50, 3).astype('float32')
        y_dummy = np.random.randint(0, 2, 10)
        
        # Call augmentation function
        generator = create_augmented_train_generator(X_dummy, y_dummy, batch_size=4)
        
        # Try to get a batch from generator
        batch_x, batch_y = next(generator)
        
        # Check batch size
        self.assertEqual(batch_x.shape[0], 4)
        print("Augmentation Generator Test Passed!")

    def test_cleanup(self):
         """Ensure clean up works"""
         pass

if __name__ == '__main__':
    unittest.main()