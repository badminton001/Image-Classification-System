import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator configuration for data augmentation

def get_train_augmentation():
    """
    Configure training data augmentation settings.
    """
    # Note: rescaling is handled in preprocessing.py
    # Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    return train_datagen

def get_test_augmentation():
    """
    Configure testing/validation data settings (no augmentation).
    """
    test_datagen = ImageDataGenerator()
    return test_datagen

def create_augmented_train_generator(X_train, y_train, batch_size=32):
    """
    Create a data generator for training from in-memory numpy arrays.
    """
    train_datagen = get_train_augmentation()
    
    generator = train_datagen.flow(
        X_train, 
        y_train, 
        batch_size=batch_size,
        shuffle=True
    )
    return generator

def create_validation_generator(X_val, y_val, batch_size=32):
    """
    Create a data generator for validation.
    """
    val_datagen = get_test_augmentation()
    
    generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )
    return generator