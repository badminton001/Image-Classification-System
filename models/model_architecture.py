"""
Model Architecture Module
Functions to build and compile VGG16, ResNet50, and MobileNetV2 models for transfer learning.
"""

import tensorflow as tf
# Reference: https://keras.io/api/applications/
# Transfer learning implementation adapted from Keras documentation
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .config_models import INPUT_SHAPE, DENSE_LAYER_UNITS, DROPOUT_RATE

def build_vgg16_model(num_classes, input_shape=INPUT_SHAPE):
    """Builds VGG16 model with frozen base and custom classification head."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_LAYER_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='VGG16_Transfer')
    return model

def build_resnet50_model(num_classes, input_shape=INPUT_SHAPE):
    """Builds ResNet50 model with frozen base and custom classification head."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_LAYER_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='ResNet50_Transfer')
    return model

def build_mobilenetv2_model(num_classes, input_shape=INPUT_SHAPE):
    """Builds MobileNetV2 model with frozen base and custom classification head."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_LAYER_UNITS, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='MobileNetV2_Transfer')
    return model

def compile_model(model, learning_rate=0.001):
    """Compiles the model with Adam optimizer and Categorical Crossentropy loss."""
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model_layers(model, num_layers_to_unfreeze=50):
    """Unfreezes the last N layers of the model for fine-tuning."""
    num_layers = len(model.layers)
    if num_layers_to_unfreeze > num_layers:
        num_layers_to_unfreeze = num_layers
        
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
        
    print(f"Unfrozen the last {num_layers_to_unfreeze} layers.")
    return model
